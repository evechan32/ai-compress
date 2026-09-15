"""Prompt 侧分块 KV 驱逐。

模式（PE_MODE）：
- off      : 旁路
- position : 位置式 sink+window（复用 R-SWA 掩码，真实物理释放）
- free     : 只释放块、不加掩码（诊断用）
- chunkkv  : 注意力打分选块（ChunkKV 式）+ 自定义 backend 压实 block_table

chunkkv 不打 R-SWA 掩码，而是由自定义 metadata builder 把被驱逐块从
attention 的 block_table 压掉并缩小 seq_lens，kernel 只 gather 保留 KV；
KV 写入仍走 slot_mapping，不受影响。
"""
from __future__ import annotations

import dataclasses
import importlib
import os

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

_LOG = os.environ.get("PE_LOG", "1") == "1"
_MODE = os.environ.get("PE_MODE", "off")

_DEFAULTS = {
    "sink": int(os.environ.get("PE_SINK", "64")),
    "window": int(os.environ.get("PE_WINDOW", "1024")),
    "budget": int(os.environ.get("PE_BUDGET", "256")),
    "chunk": int(os.environ.get("PE_CHUNK", "16")),
    "obs": int(os.environ.get("PE_OBS", "64")),
    "ratio": float(os.environ.get("PE_RATIO", "0")),
    "agg": os.environ.get("PE_AGG", "sum"),
    "vote_layers": int(os.environ.get("PE_VOTE_LAYERS", "1")),
    "score_mode": os.environ.get("PE_SCORE_MODE", "window").lower(),
    "ctx_queries": int(os.environ.get("PE_CTX_QUERIES", "256")),
    "ctx_cap": int(os.environ.get("PE_CTX_CAP", "16384")),
    "use_covariance": os.environ.get("PE_USE_COV", "1") == "1",
    "use_vnorm": os.environ.get("PE_USE_VNORM", "1") == "1",
    "win_agg": os.environ.get("PE_WIN_AGG", "sum").lower(),
}

_PE_REQ_IDS: list = []
_PE_RETAINED: dict[str, tuple[int, tuple[int, ...]]] = {}
_PE_LAYER_SEEN: set = set()
_PE_CFG: dict = {}
_PE_VERSION: int = 0
_PE_ATTACHED: int = -1


def _ingest(out) -> None:
    try:
        pe = getattr(out, "pe_retained", None)
        if isinstance(pe, dict):
            _PE_RETAINED.update(pe)
    except Exception:
        pass


def _patch_output_transport() -> None:
    """TP>1：仅 rank0 的输出会回 scheduler；把 worker 的保留集附在输出上，
    在 engine 进程落回 _PE_RETAINED（manager 读的那张表）。TP=1 同进程时是 no-op。"""
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.outputs import ModelRunnerOutput

    if not getattr(ModelRunnerOutput, "_pe_out_patched", False):
        orig_init = ModelRunnerOutput.__init__

        def init(self, *a, **kw):
            global _PE_ATTACHED
            orig_init(self, *a, **kw)
            self.pe_retained = None
            if _PE_RETAINED and _PE_ATTACHED != _PE_VERSION:
                _PE_ATTACHED = _PE_VERSION
                if len(_PE_RETAINED) > 4096:
                    for k in list(_PE_RETAINED)[: len(_PE_RETAINED) - 1024]:
                        _PE_RETAINED.pop(k, None)
                self.pe_retained = dict(_PE_RETAINED)

        ModelRunnerOutput.__init__ = init
        ModelRunnerOutput._pe_out_patched = True

    if not getattr(Executor, "_pe_exec_patched", False):
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        def _wrap(cls, name):
            orig = getattr(cls, name)

            def f(self, *a, **kw):
                out = orig(self, *a, **kw)
                _ingest(out)
                return out

            setattr(cls, name, f)

        for cls in (MultiprocExecutor, UniProcExecutor):
            for name in ("execute_model", "sample_tokens"):
                if hasattr(cls, name):
                    _wrap(cls, name)
        Executor._pe_exec_patched = True
_PE_PROMPT_LEN: dict[str, int] = {}
_PE_PREFILLING: dict[str, bool] = {}
_PE_Q_BUF: dict[tuple, torch.Tensor] = {}
_PE_Q_LASTL: dict[tuple, int] = {}
_PE_IMP_ACC: dict[str, torch.Tensor] = {}
_PE_VOTES: dict[str, int] = {}
_PE_MAX_LAYER: int = -1
_PE_VALIDATED: bool = False
_IN_TARGET_LAYER: bool = False


def _tp_all_reduce(x):
    """把各 rank 的局部重要性求和，保证所有 rank 得到同一份全局分数。"""
    try:
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        if g is not None and getattr(g, "world_size", 1) > 1:
            return g.all_reduce(x)
    except Exception:
        pass
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
    except Exception:
        pass
    return x


def _check_process_mode(tp: int) -> None:
    mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    on = str(mp).lower() not in ("0", "false")
    if tp > 1:
        if not on:
            raise RuntimeError(
                "pevict: TP>1 需要 VLLM_ENABLE_V1_MULTIPROCESSING=1"
                "（保留集经 ModelRunnerOutput 回传）。"
            )
    elif on:
        raise RuntimeError(
            "pevict: TP=1 需要 VLLM_ENABLE_V1_MULTIPROCESSING=0"
            "（保留集经进程内表直传）。"
        )


def _validate_config(vllm_config) -> None:
    """一次性启动校验：把不支持的部署配置挡在启动期，而不是静默出错。"""
    global _PE_VALIDATED
    if _PE_VALIDATED:
        return
    _PE_VALIDATED = True
    pc = getattr(vllm_config, "parallel_config", None)
    if pc is not None:
        tp = int(getattr(pc, "tensor_parallel_size", 1) or 1)
        pp = int(getattr(pc, "pipeline_parallel_size", 1) or 1)
        if pp != 1:
            raise RuntimeError(
                f"pevict 暂不支持 PP>1（当前 pp={pp}）：打分发生在最后一层，跨 stage 无对应机制。"
            )
        _check_process_mode(tp)
        if tp > 1:
            sc0 = getattr(vllm_config, "scheduler_config", None)
            if sc0 is not None and getattr(sc0, "async_scheduling", False):
                raise RuntimeError(
                    "pevict: TP>1 需关闭 async_scheduling（保留集回传依赖调度/执行步对齐）。"
                    "LLM 侧传 async_scheduling=False，serve 侧加 --no-async-scheduling。"
                )
            _patch_output_transport()
    sc = getattr(vllm_config, "scheduler_config", None)
    if sc is not None and getattr(sc, "async_scheduling", False):
        print("[PE] warn: async_scheduling 已开启；_PE_REQ_IDS 依赖调度与执行步对齐。"
              "单进程下实测正确，但若出现请求串扰请传 async_scheduling=False。",
              flush=True)
    cc = getattr(vllm_config, "compilation_config", None)
    if cc is not None:
        cm = getattr(cc, "cudagraph_mode", None)
        if cm is not None and "NONE" not in str(getattr(cm, "name", cm)).upper():
            raise RuntimeError(
                f"pevict 需要关闭 CUDA graph（打分含动态 shape 与 host 同步），"
                f"当前 cudagraph_mode={cm}；请设 enforce_eager=True。"
            )


def _layer_idx(layer) -> int:
    name = getattr(layer, "layer_name", "") or ""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


@dataclasses.dataclass(frozen=True, kw_only=True)
class PromptEvictSpec(FullAttentionSpec):
    sink: int = 64
    window: int = 1024
    budget: int = 256
    chunk: int = 16
    obs: int = 64
    ratio: float = 0.0
    agg: str = "sum"
    vote_layers: int = 1
    score_mode: str = "window"
    ctx_queries: int = 256
    ctx_cap: int = 16384
    use_covariance: bool = True
    use_vnorm: bool = True
    win_agg: str = "sum"

    @classmethod
    def merge(cls, specs):
        base = FullAttentionSpec.merge(specs)
        keys = tuple(_DEFAULTS)
        params = {k: getattr(specs[0], k) for k in keys}
        for s in specs:
            for k, v in params.items():
                assert getattr(s, k) == v, f"PromptEvictSpec {k} mismatch"
        return build_spec(base, params)


def build_spec(base, params):
    kw = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    kw.update({k: params.get(k, v) for k, v in _DEFAULTS.items()})
    return PromptEvictSpec(**kw)


def _block_size_of(layer) -> int:
    try:
        return int(layer.kv_cache.shape[2])
    except Exception as e:
        raise RuntimeError(
            "pevict: 无法从 layer.kv_cache 读取 block_size（可能处于 direct-call 路径）。"
            "拒绝回退到默认 16，以免静默算错保留集。请用 PE_MODE=chunkkv（从 kv_cache 取块大小）。"
        ) from e


class PromptEvictAttention(Attention):
    def __init__(self, *args, pe_params=None, **kwargs) -> None:
        global _PE_MAX_LAYER, _IN_TARGET_LAYER
        _IN_TARGET_LAYER = True
        try:
            super().__init__(*args, **kwargs)
        finally:
            _IN_TARGET_LAYER = False
        self._pe_params = pe_params or dict(_DEFAULTS)
        _PE_MAX_LAYER = max(_PE_MAX_LAYER, _layer_idx(self))

    def get_kv_cache_spec(self, vllm_config):
        _validate_config(vllm_config)
        base = super().get_kv_cache_spec(vllm_config)
        if base is None:
            return None
        return build_spec(base, dict(self._pe_params))

    def _maybe_score(self, query, key) -> None:
        from vllm.forward_context import get_forward_context
        md_map = get_forward_context().attn_metadata
        if not isinstance(md_map, dict):
            return
        md = md_map.get(self.layer_name)
        if md is None or md.block_table is None or getattr(md, "seq_lens", None) is None:
            return
        qsl = md.query_start_loc.tolist()
        seqlens = md.seq_lens.tolist()
        bs = _block_size_of(self)
        cfg = self._pe_params
        for r in range(len(seqlens)):
            q_len = qsl[r + 1] - qsl[r]
            L = int(seqlens[r])
            if q_len != L or L <= 0:
                continue
            req_id = _PE_REQ_IDS[r] if r < len(_PE_REQ_IDS) else None
            if req_id is None:
                continue
            self._score_one(query, key, qsl[r], qsl[r + 1], L, req_id, bs, cfg)

    def _score_one(self, query, key, qs, qe, L, req_id, bs, cfg):
        Hq, Hkv, D = self.num_heads, self.num_kv_heads, self.head_size
        rep = Hq // Hkv
        q3 = query[qs:qe].detach().reshape(-1, Hq, D).float()
        k3 = key[qs:qe].detach().reshape(-1, Hkv, D).float()
        w = min(int(cfg["obs"]), q3.shape[0])
        kk = k3.repeat_interleave(rep, dim=1)
        sc = torch.einsum("whd,lhd->hwl", q3[-w:], kk) * (D ** -0.5)
        imp = torch.softmax(sc, dim=-1).sum(dim=1).mean(dim=0)
        nblk = (L + bs - 1) // bs
        bscore = torch.zeros(nblk, device=imp.device)
        bscore.index_add_(0, torch.arange(L, device=imp.device) // bs, imp)
        order = torch.argsort(bscore, descending=True).tolist()
        ratio = float(cfg.get("ratio", 0) or 0)
        budget = int(ratio * L) if ratio > 0 else int(cfg["budget"])
        keep = set(range(min(int(cfg["sink"]) // bs, nblk)))
        kept = len(keep) * bs
        for b in order:
            keep.add(b)
            kept += bs
            if kept >= budget:
                break
        wb = (w + bs - 1) // bs
        keep.update(range(max(0, nblk - wb), nblk))
        _PE_RETAINED[req_id] = (nblk, tuple(sorted(keep)))
        global _PE_VERSION
        _PE_VERSION += 1
        if _LOG and nblk > 4 and self.layer_name not in _PE_LAYER_SEEN:
            _PE_LAYER_SEEN.add(self.layer_name)
            print(f"[PE] score layer={self.layer_name} keep={len(keep)}/{nblk}",
                  flush=True)

    def forward(self, query, key, value, output_shape=None, output_dtype=None):
        if _MODE == "mask" and key is not None:
            try:
                self._maybe_score(query, key)
            except Exception as e:
                print("[PE] score warn:", type(e).__name__, str(e)[:160], flush=True)
        return super().forward(query, key, value, output_shape, output_dtype)


class PromptEvictManager(FullAttentionManager):
    def __init__(self, kv_cache_spec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self._pe_spec = kv_cache_spec
        self._pe_logged: set = set()

    def _free_idx(self, request_id, idxs) -> int:
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return 0
        n = 0
        for i in idxs:
            if i >= len(blocks) or blocks[i] == self._null_block:
                continue
            self.block_pool.free_blocks([blocks[i]])
            blocks[i] = self._null_block
            n += 1
        return n

    def _evict_position(self, request_id, processed):
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        bs = self.block_size
        sink, window = self._pe_spec.sink, self._pe_spec.window
        first = (sink + bs - 1) // bs
        last = min(max(sink, processed - window) // bs, len(blocks))
        if first >= last:
            return
        before = self.block_pool.get_num_free_blocks()
        n = self._free_idx(request_id, range(first, last))
        if _LOG and (request_id, "ev") not in self._pe_logged:
            self._pe_logged.add((request_id, "ev"))
            print(f"[PE] evict position req={request_id} freed={n} "
                  f"pool {before}->{self.block_pool.get_num_free_blocks()}",
                  flush=True)

    def _evict_chunkkv(self, request_id):
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        ent = _PE_RETAINED.get(request_id)
        if ent is None:
            return
        prompt_nblk, keep = ent
        keep_set = set(keep)
        nblk = min(prompt_nblk, len(blocks))
        drop = [i for i in range(nblk) if i not in keep_set]
        before = self.block_pool.get_num_free_blocks()
        n = self._free_idx(request_id, drop)
        _PE_IMP_ACC.pop(request_id, None)
        _PE_VOTES.pop(request_id, None)
        for k in [k for k in _PE_Q_BUF if k[1] == request_id]:
            _PE_Q_BUF.pop(k, None)
        for k in [k for k in _PE_Q_LASTL if k[1] == request_id]:
            _PE_Q_LASTL.pop(k, None)
        if _LOG and (request_id, "ev") not in self._pe_logged:
            self._pe_logged.add((request_id, "ev"))
            print(f"[PE] evict chunkkv req={request_id} prompt_nblk={prompt_nblk} "
                  f"keep={len(keep)} freed={n} "
                  f"pool {before}->{self.block_pool.get_num_free_blocks()}",
                  flush=True)

    def free(self, request_id):
        if _LOG and request_id in _PE_RETAINED:
            print(f"[PE] free req={request_id} retained={len(_PE_RETAINED)} "
                  f"prompt_len={len(_PE_PROMPT_LEN)} qbuf={len(_PE_Q_BUF)}",
                  flush=True)
        _PE_RETAINED.pop(request_id, None)
        _PE_PROMPT_LEN.pop(request_id, None)
        _PE_IMP_ACC.pop(request_id, None)
        _PE_VOTES.pop(request_id, None)
        for k in [k for k in _PE_Q_BUF if k[1] == request_id]:
            _PE_Q_BUF.pop(k, None)
        for k in [k for k in _PE_Q_LASTL if k[1] == request_id]:
            _PE_Q_LASTL.pop(k, None)
        for k in [k for k in _PE_LAYER_SEEN if k[1] == request_id]:
            _PE_LAYER_SEEN.discard(k)
        self._pe_logged = {k for k in self._pe_logged
                           if not (isinstance(k, tuple) and k[0] == request_id)}
        return super().free(request_id)

    def remove_skipped_blocks(self, request_id, processed_computed_tokens,
                              num_prompt_tokens=None):
        if num_prompt_tokens is not None:
            _PE_PROMPT_LEN[request_id] = int(num_prompt_tokens)
            done = processed_computed_tokens >= num_prompt_tokens
            if _LOG and done and (request_id, "done") not in self._pe_logged:
                self._pe_logged.add((request_id, "done"))
                blocks = self.req_to_blocks.get(request_id)
                print(f"[PE] prefilled req={request_id} prompt={num_prompt_tokens} "
                      f"blocks={len(blocks) if blocks else None}", flush=True)
            if done and _MODE == "position":
                self._evict_position(request_id, processed_computed_tokens)
            elif done and _MODE in ("chunkkv", "free", "mask"):
                self._evict_chunkkv(request_id)
        return super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )


def _patch_registry() -> None:
    if getattr(KVCacheSpecRegistry, "_pe_patched", False):
        return
    orig = KVCacheSpecRegistry.get_manager_class.__func__

    def patched(cls, kvcache_spec):
        cls._ensure_registered()
        KVCacheSpecRegistry.register(
            PromptEvictSpec, PromptEvictManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        return orig(cls, kvcache_spec)

    KVCacheSpecRegistry.get_manager_class = classmethod(patched)
    KVCacheSpecRegistry._pe_patched = True


def _patch_rswa_window(window: int) -> None:
    from vllm.config.model import ModelConfig
    if getattr(ModelConfig, "_pe_window_patched", False):
        return
    orig = ModelConfig.rswa_window.fget

    def prop(self):
        v = orig(self)
        return window if v is None else v

    ModelConfig.rswa_window = property(prop)
    ModelConfig._pe_window_patched = True


def _patch_prefix_clamp(sink: int) -> None:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    if getattr(CommonAttentionMetadata, "_pe_clamp_patched", False):
        return
    orig = CommonAttentionMetadata.__init__

    def init(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        rp = self.rswa_prefix_lens
        if rp is not None:
            self.rswa_prefix_lens = torch.clamp(rp, max=sink)

    CommonAttentionMetadata.__init__ = init
    CommonAttentionMetadata._pe_clamp_patched = True


def _patch_req_ids():
    import importlib
    mod = importlib.import_module("vllm.v1.worker.gpu.model_states.default")
    cls = mod.DefaultModelState
    if getattr(cls, "_pe_reqids_patched", False):
        return
    orig = cls.prepare_attn

    def prepare(self, input_batch, *args, **kwargs):
        try:
            ids = list(input_batch.req_ids)[: input_batch.num_reqs]
            _PE_REQ_IDS[:] = ids
            pl = getattr(input_batch, "prompt_lens", None)
            if pl is not None:
                vals = pl.tolist() if hasattr(pl, "tolist") else list(pl)
                for i, rid in enumerate(ids):
                    if i < len(vals):
                        _PE_PROMPT_LEN[rid] = int(vals[i])
            ip = getattr(input_batch, "is_prefilling_np", None)
            if ip is not None:
                for i, rid in enumerate(ids):
                    if i < len(ip):
                        _PE_PREFILLING[rid] = bool(ip[i])
        except Exception:
            pass
        return orig(self, input_batch, *args, **kwargs)

    cls.prepare_attn = prepare
    cls._pe_reqids_patched = True


def _select_blocks(imp, L, bs, cfg, req_id):
    nblk = (L + bs - 1) // bs
    idx = torch.arange(L, device=imp.device) // bs
    agg = str(cfg.get("agg", "sum")).lower()
    if agg == "max":
        bscore = torch.full((nblk,), float("-inf"), device=imp.device)
        bscore.scatter_reduce_(0, idx, imp, reduce="amax", include_self=False)
    elif agg == "mean":
        s = torch.zeros(nblk, device=imp.device)
        s.index_add_(0, idx, imp)
        c = torch.zeros(nblk, device=imp.device)
        c.index_add_(0, idx, torch.ones_like(imp))
        bscore = s / c.clamp_min(1.0)
    else:
        bscore = torch.zeros(nblk, device=imp.device)
        bscore.index_add_(0, idx, imp)
    order = torch.argsort(bscore, descending=True).tolist()
    ratio = float(cfg.get("ratio", 0) or 0)
    budget = int(ratio * L) if ratio > 0 else int(cfg["budget"])
    keep = set(range(min(int(cfg["sink"]) // bs, nblk)))
    kept = len(keep) * bs
    for b in order:
        keep.add(b)
        kept += bs
        if kept >= budget:
            break
    wb = (int(cfg["obs"]) + bs - 1) // bs
    keep.update(range(max(0, nblk - wb), nblk))
    _PE_RETAINED[req_id] = (nblk, tuple(sorted(keep)))
    global _PE_VERSION
    _PE_VERSION += 1
    return nblk, keep


def _avg_rope(D, theta, n_future, device, dtype):
    """未来 n_future 个位置的 RoPE 平均旋转矩阵 R_avg = mean_Δ R(Δ)。"""
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=device).float() / D))
    f = torch.outer(torch.arange(1, n_future + 1, device=device).float(), inv)
    emb = torch.cat([f, f], dim=-1)
    cos, sin = emb.cos().mean(0), emb.sin().mean(0)
    eye = torch.eye(D, device=device, dtype=dtype)
    P = torch.zeros(D, D, device=device, dtype=dtype)
    P[D // 2:, : D // 2] = torch.eye(D // 2, device=device, dtype=dtype)
    P[: D // 2, D // 2:] = -torch.eye(D // 2, device=device, dtype=dtype)
    return cos.unsqueeze(1) * eye + sin.unsqueeze(1) * P


def _expected_imp(buf, k_all, kv_cache, blk, nblk, bs, Hq, Hkv, D, rep, cfg):
    """ExpectedAttention 式打分（query-agnostic）。

    score_j = softmax_j(k_j·μ/√d + ½·k_jΣk_jᵀ/d) · ‖v_j‖，μ/Σ 为 query 的均值/协方差。
    未来位置用 n_future 个位置的 RoPE 平均旋转近似（RoPE 分块旋转可交换，故直接作用于
    post-RoPE query 与 kvpress 的 avg-RoPE 等价）。
    """
    q = buf[min(int(cfg.get("sink", 0)), max(0, buf.shape[0] - 1)):].float()
    R = _avg_rope(D, float(cfg.get("rope_theta", 1e6)),
                  int(cfg.get("n_future", 512)), q.device, q.dtype)
    q = q @ R.T
    mu = q.mean(dim=0)
    ke = k_all.repeat_interleave(rep, dim=1)
    logits = torch.einsum("lhd,hd->lh", ke, mu) * (D ** -0.5)
    if cfg.get("use_covariance", True) and q.shape[0] > 1:
        cen = q - mu
        cov = torch.einsum("qhd,qhe->hde", cen, cen) / q.shape[0]
        logits = logits + 0.5 * torch.einsum("lhd,hde,lhe->lh", ke, cov, ke) / D
    p = torch.softmax(logits, dim=0)
    p_grp = p.view(k_all.shape[0], Hkv, rep).mean(dim=2)
    if cfg.get("use_vnorm", True):
        vc = kv_cache.transpose(1, 2)[..., D:]
        v_all = vc[blk].reshape(nblk * bs, Hkv, D)[:k_all.shape[0]].float()
        p_grp = p_grp * v_all.norm(dim=-1)
    return p_grp.mean(dim=1)


def _score_from_cache(layer, query, key, kv_cache, md) -> None:
    """从 KV 池 gather 全序列 K 后打分（支持 chunked prefill + 多层投票）。

    只在"本 chunk 走完整个 prompt"（seq_lens == prompt_len）时打分；
    此时池中已包含全部 prompt KV（本函数在 super().forward() 之后调用）。
    最后 vote_layers 层的逐 token 重要性求平均后选块。
    """
    li = _layer_idx(layer)
    cfg = _PE_CFG
    k_vote = max(1, int(cfg.get("vote_layers", 1) or 1))
    if li < _PE_MAX_LAYER - k_vote + 1 or li > _PE_MAX_LAYER:
        return
    pl = getattr(md, "pe_prompt_lens", None)
    qsl = md.query_start_loc.tolist()
    seqlens = md.seq_lens.tolist()
    bt = md.block_table
    bs = int(kv_cache.shape[2])
    Hq, Hkv, D = layer.num_heads, layer.num_kv_heads, layer.head_size
    rep = max(1, Hq // Hkv)
    pl_list = pl.tolist() if pl is not None else None
    kc = kv_cache.transpose(1, 2)[..., :D]
    obs = int(cfg["obs"])
    mode = str(cfg.get("score_mode", "window")).lower()
    cap = int(cfg.get("ctx_cap", 16384)) if mode in ("context", "expected") else obs
    for r in range(len(seqlens)):
        L = int(seqlens[r])
        req_id = _PE_REQ_IDS[r] if r < len(_PE_REQ_IDS) else None
        if req_id is None:
            continue
        qs, qe = qsl[r], qsl[r + 1]
        bkey = (li, req_id)
        if _PE_PREFILLING.get(req_id, True):
            if _PE_Q_LASTL.get(bkey) != L:
                _PE_Q_LASTL[bkey] = L
                q_chunk = query[qs:qe].detach().reshape(-1, Hq, D)
                buf = _PE_Q_BUF.get(bkey)
                buf = q_chunk if buf is None else torch.cat([buf, q_chunk], dim=0)
                _PE_Q_BUF[bkey] = buf[-cap:]
            continue
        if req_id in _PE_RETAINED:
            continue
        buf = _PE_Q_BUF.get(bkey)
        if buf is None:
            continue
        nblk = (L + bs - 1) // bs
        blk = bt[r, :nblk].to(torch.long)
        k_all = kc[blk].reshape(nblk * bs, Hkv, D)[:L].float()
        if mode == "expected":
            imp = _expected_imp(buf, k_all, kv_cache, blk, nblk, bs,
                                Hq, Hkv, D, rep, cfg)
            w = buf.shape[0]
        else:
            q3 = buf.float()
            if mode == "context":
                nq = min(int(cfg.get("ctx_queries", 256)), q3.shape[0])
                idxq = torch.linspace(0, q3.shape[0] - 1, nq).long().to(q3.device)
                q_sel = q3[idxq]
                w = nq
            else:
                w = min(obs, q3.shape[0])
                q_sel = q3[-w:]
            kk = k_all.repeat_interleave(rep, dim=1)
            sc = torch.einsum("whd,lhd->hwl", q_sel, kk) * (D ** -0.5)
            prob = torch.softmax(sc, dim=-1)
            if mode == "context" or str(cfg.get("win_agg", "sum")).lower() == "max":
                imp = prob.amax(dim=1).mean(dim=0)
            else:
                imp = prob.sum(dim=1).mean(dim=0)
        acc = _PE_IMP_ACC.get(req_id)
        _PE_IMP_ACC[req_id] = imp if acc is None else acc + imp
        _PE_VOTES[req_id] = _PE_VOTES.get(req_id, 0) + 1
        if li != _PE_MAX_LAYER:
            continue
        imp_avg = _PE_IMP_ACC.pop(req_id) / max(1, _PE_VOTES.pop(req_id, 1))
        imp_avg = _tp_all_reduce(imp_avg)
        _select_blocks(imp_avg, L, bs, cfg, req_id)
        for k in range(_PE_MAX_LAYER - k_vote + 1, _PE_MAX_LAYER + 1):
            _PE_Q_BUF.pop((k, req_id), None)
            _PE_Q_LASTL.pop((k, req_id), None)
        key_seen = (layer.layer_name, req_id)
        if _LOG and nblk > 4 and key_seen not in _PE_LAYER_SEEN:
            _PE_LAYER_SEEN.add(key_seen)
            print(f"[PE] score layer={layer.layer_name} at_L={L} "
                  f"win={w} votes={k_vote} keep={len(_PE_RETAINED[req_id][1])}/{nblk}",
                  flush=True)


def _build_flex_backend():
    from vllm.v1.attention.backend import subclass_attention_backend
    from vllm.v1.attention.backends.flex_attention import (
        FlexAttentionBackend,
        FlexAttentionMetadataBuilder,
        physical_to_logical_mapping,
    )

    class PromptEvictFlexBuilder(FlexAttentionMetadataBuilder):
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            md = super().build(common_prefix_len, common_attn_metadata, fast_build)
            if not _PE_RETAINED:
                return md
            bt, sl = md.block_table, md.seq_lens
            if bt is None or sl is None or bt.numel() == 0:
                return md
            bs = md.block_size
            nreq = common_attn_metadata.num_reqs
            new_bt = bt.clone()
            changed = False
            for r in range(nreq):
                req_id = _PE_REQ_IDS[r] if r < len(_PE_REQ_IDS) else None
                ent = _PE_RETAINED.get(req_id)
                if ent is None:
                    continue
                prompt_nblk, keep = ent
                kset = set(keep)
                L = int(sl[r].item())
                nblk = (L + bs - 1) // bs
                for i in range(min(prompt_nblk, nblk)):
                    if i not in kset:
                        new_bt[r, i] = 0
                        changed = True
            if changed:
                inv = physical_to_logical_mapping(
                    new_bt, sl, bs, self.cache_config.num_gpu_blocks
                )
                md.physical_to_logical[:nreq].copy_(inv[:nreq])
                md.block_table = new_bt
            return md

    return subclass_attention_backend("PromptEvictFlex", FlexAttentionBackend,
                                      PromptEvictFlexBuilder)


def _build_backend():
    from vllm.v1.attention.backend import subclass_attention_backend_with_overrides
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionImpl,
        TritonAttentionMetadataBuilder,
    )

    class PromptEvictImpl(TritonAttentionImpl):
        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output, *args, **kwargs):
            out = super().forward(layer, query, key, value, kv_cache,
                                  attn_metadata, output, *args, **kwargs)
            if _MODE == "chunkkv" and attn_metadata is not None:
                try:
                    _score_from_cache(layer, query, key, kv_cache, attn_metadata)
                except Exception as e:
                    print("[PE] score warn:", type(e).__name__, str(e)[:160],
                          flush=True)
            return out

    class PromptEvictBuilder(TritonAttentionMetadataBuilder):
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            md = super().build(common_prefix_len, common_attn_metadata, fast_build)
            try:
                md.pe_prompt_lens = common_attn_metadata.rswa_prefix_lens
            except Exception:
                md.pe_prompt_lens = None
            if not _PE_RETAINED:
                return md
            bt, sl = md.block_table, md.seq_lens
            if bt is None or sl is None or bt.numel() == 0:
                return md
            bs = self.kv_cache_spec.block_size
            nreq = sl.shape[0]
            new_bt, new_sl = bt.clone(), sl.clone()
            changed = False
            for r in range(nreq):
                req_id = _PE_REQ_IDS[r] if r < len(_PE_REQ_IDS) else None
                ent = _PE_RETAINED.get(req_id)
                if ent is None:
                    continue
                prompt_nblk, keep = ent
                L = int(sl[r].item())
                nblk = (L + bs - 1) // bs
                idx = sorted(set(keep) | set(range(prompt_nblk, nblk)))
                if len(idx) == nblk and idx == list(range(nblk)):
                    continue
                src = bt[r][torch.tensor(idx, device=bt.device)]
                new_bt[r, :len(idx)] = src
                new_bt[r, len(idx):] = 0
                new_sl[r] = sum(min(bs, L - i * bs) for i in idx)
                changed = True
            if changed:
                md.block_table, md.seq_lens = new_bt, new_sl
                md.max_seq_len = int(new_sl.max().item())
            return md

    return subclass_attention_backend_with_overrides(
        "PromptEvict", TritonAttentionBackend,
        {"get_builder_cls": lambda: PromptEvictBuilder,
         "get_impl_cls": lambda: PromptEvictImpl},
    )


def _patch_custom_backend():
    import vllm.model_executor.layers.attention.attention as attn_mod
    if getattr(attn_mod, "_pe_backend_patched", False):
        return
    backend = _build_flex_backend() if _MODE == "mask" else _build_backend()
    want = "FlexAttentionBackend" if _MODE == "mask" else "TritonAttentionBackend"
    orig = attn_mod.get_attn_backend

    def sel(*args, **kwargs):
        cls = orig(*args, **kwargs)
        if not _IN_TARGET_LAYER:
            return cls
        if cls.__name__ == want:
            return backend
        raise RuntimeError(
            f"PE_MODE={_MODE} 需要 {want}，但选中 {cls.__name__}。"
            f"请加对应的 --attention-backend。"
        )

    attn_mod.get_attn_backend = sel
    attn_mod._pe_backend_patched = True


def install(params=None, arch_module="vllm.model_executor.models.qwen2") -> None:
    _patch_registry()
    pe_params = dict(_DEFAULTS)
    if params:
        pe_params.update(params)
    if _MODE == "position":
        _patch_rswa_window(pe_params["window"])
        _patch_prefix_clamp(pe_params["sink"])
    elif _MODE in ("chunkkv", "free", "mask", "passthrough"):
        _PE_CFG.update(pe_params)
        _patch_req_ids()
        _patch_custom_backend()
        _patch_output_transport()
    mod = importlib.import_module(arch_module)
    if not hasattr(mod, "Attention"):
        raise ValueError(f"{arch_module} 未导出 Attention")
    if getattr(mod, "_pe_installed", False):
        return

    def factory(*args, **kwargs):
        return PromptEvictAttention(*args, pe_params=pe_params, **kwargs)

    mod.Attention = factory
    mod._pe_installed = True
    print(f"[PE] installed mode={_MODE} params={pe_params} arch={arch_module}",
          flush=True)
