"""vLLM **0.11.2** 端口 —— S2：管路 + 注意力打分（chunkkv）+ 生成段窗口。

与 0.28 的差异（均实测确认）：
  - Attention：`vllm.attention.layer.Attention`
  - spec→manager：`spec_manager_map[type(spec)]` 精确类型查表 → 自己注册
  - 无 `KVCacheSpecRegistry`：override `Attention.get_kv_cache_spec()`
  - `remove_skipped_blocks(req, num_computed_tokens)` 无 num_prompt_tokens
    → patch `KVCacheManager.allocate_slots` 记录 `request.num_prompt_tokens`
  - 无 `subclass_attention_backend`：手动子类化 + patch `layer.get_attn_backend`
  - 行号→req_id：patch `GPUModelRunner._prepare_inputs`
  - `TritonAttentionMetadata` 自带 query_start_loc / seq_lens / block_table

与 0.28 的功能差异：**本版含生成段窗口**（0.28 的 chunkkv 模式完全不压生成段）。
保留集统一以**块下标**（非块 id）描述，避免块复用导致的误伤。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_LOG = os.environ.get("PE_LOG", "0") == "1"
_MODE = os.environ.get("PE_MODE", "off")

_DEFAULTS = {
    "sink": int(os.environ.get("PE_SINK", "64")),
    "window": int(os.environ.get("PE_WINDOW", "1024")),
    "budget": int(os.environ.get("PE_BUDGET", "256")),
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
    "gen_window": int(os.environ.get("PE_GEN_WINDOW", "0")),
    "rope_theta": float(os.environ.get("PE_ROPE_THETA", "1e6")),
    "n_future": int(os.environ.get("PE_N_FUTURE", "512")),
}

_CFG: dict = {}
_MAX_LAYER: int = -1
_PROMPT_LEN: dict[str, int] = {}
_REQ_IDS: list[str] = []
_RETAINED: dict[str, tuple[int, tuple[int, ...]]] = {}
_Q_BUF: dict = {}
_Q_LASTL: dict = {}
_IMP_ACC: dict = {}
_VOTES: dict = {}
_LAYER_SEEN: set = set()
_DBG_SEEN: set = set()
_VER: int = 0
_BUILT_VER: int = -1
_PROMPT_DONE: set = set()


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


def _make_spec_cls():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    @dataclass(frozen=True)
    class _PromptEvictSpec(FullAttentionSpec):
        sink: int = 64
        window: int = 1024

    return _PromptEvictSpec


def _select_blocks(imp, L, bs, cfg, req_id):
    import torch

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
    _RETAINED[req_id] = (nblk, tuple(sorted(keep)))
    global _VER
    _VER += 1
    return nblk, keep


def _avg_rope(D, theta, n_future, device, dtype):
    import torch

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
    import torch

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
        v_all = kv_cache[blk, 1].reshape(nblk * bs, Hkv, D)[:k_all.shape[0]].float()
        p_grp = p_grp * v_all.norm(dim=-1)
    return p_grp.mean(dim=1)


def _score_from_cache(layer, query, key, kv_cache, md) -> None:
    """从 KV 池 gather 全序列 K 后打分（支持 chunked prefill + 多层投票）。

    prompt 长度来自 _PROMPT_LEN（0.11.2 无 rswa_prefix_lens）；
    `L < prompt_len` 即视为仍在 prefill。
    """
    import torch

    li = _layer_idx(layer)
    cfg = _CFG
    if _LOG and ("shape",) not in _DBG_SEEN:
        _DBG_SEEN.add(("shape",))
        print(f"[PE11] kv_cache.shape={tuple(kv_cache.shape)} "
              f"query={tuple(query.shape)} key={None if key is None else tuple(key.shape)} "
              f"Hq={layer.num_heads} Hkv={layer.num_kv_heads} D={layer.head_size}", flush=True)
    if li < 0 or li > _MAX_LAYER:
        return
    bs = int(kv_cache.shape[2])
    if bs <= 1:
        return
    k_vote = max(1, int(cfg.get("vote_layers", 1) or 1))
    if li < _MAX_LAYER - k_vote + 1:
        return
    live = [r for r in _REQ_IDS if r is not None]
    if live and all(r in _RETAINED for r in live):
        return
    qsl = md.query_start_loc.tolist()
    seqlens = md.seq_lens.tolist()
    bt = md.block_table
    Hq, Hkv, D = layer.num_heads, layer.num_kv_heads, layer.head_size
    rep = max(1, Hq // Hkv)
    kc = kv_cache[:, 0]
    obs = int(cfg["obs"])
    mode = str(cfg.get("score_mode", "window")).lower()
    cap = int(cfg.get("ctx_cap", 16384)) if mode in ("context", "expected") else obs
    for r in range(len(seqlens)):
        req_id = _REQ_IDS[r] if r < len(_REQ_IDS) else None
        if req_id is None:
            continue
        L = int(seqlens[r])
        plen = _PROMPT_LEN.get(req_id)
        if plen is None:
            continue
        qs, qe = qsl[r], qsl[r + 1]
        bkey = (li, req_id)
        if _LOG and li == _MAX_LAYER and (req_id, L) not in _DBG_SEEN:
            _DBG_SEEN.add((req_id, L))
            blen = None if _Q_BUF.get(bkey) is None else int(_Q_BUF[bkey].shape[0])
            print(f"[PE11] dbg2 L={L} plen={plen} qs={qs} qe={qe} buf={blen} "
                  f"retained={req_id in _RETAINED}", flush=True)
        if L <= plen:
            if _Q_LASTL.get(bkey) != L:
                _Q_LASTL[bkey] = L
                q_chunk = query[qs:qe].detach().reshape(-1, Hq, D)
                buf = _Q_BUF.get(bkey)
                buf = q_chunk if buf is None else torch.cat([buf, q_chunk], dim=0)
                _Q_BUF[bkey] = buf[-cap:]
            if L < plen:
                continue
        if req_id in _RETAINED:
            continue
        buf = _Q_BUF.get(bkey)
        if buf is None:
            continue
        nblk = (L + bs - 1) // bs
        blk = bt[r, :nblk].to(torch.long)
        k_all = kc[blk].reshape(nblk * bs, Hkv, D)[:L].float()
        if mode == "expected":
            imp = _expected_imp(buf, k_all, kv_cache, blk, nblk, bs, Hq, Hkv, D, rep, cfg)
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
        acc = _IMP_ACC.get(req_id)
        _IMP_ACC[req_id] = imp if acc is None else acc + imp
        _VOTES[req_id] = _VOTES.get(req_id, 0) + 1
        if li != _MAX_LAYER:
            continue
        imp_avg = _IMP_ACC.pop(req_id) / max(1, _VOTES.pop(req_id, 1))
        _select_blocks(imp_avg, L, bs, cfg, req_id)
        for k in range(_MAX_LAYER - k_vote + 1, _MAX_LAYER + 1):
            _Q_BUF.pop((k, req_id), None)
            _Q_LASTL.pop((k, req_id), None)
        key_seen = (getattr(layer, "layer_name", "?"), req_id)
        if _LOG and nblk > 4 and key_seen not in _LAYER_SEEN:
            _LAYER_SEEN.add(key_seen)
            print(f"[PE11] score layer={layer.layer_name} at_L={L} win={w} "
                  f"votes={k_vote} keep={len(_RETAINED[req_id][1])}/{nblk}", flush=True)


def _prompt_nblk(req_id, bs):
    plen = _PROMPT_LEN.get(req_id)
    return None if plen is None else (plen + bs - 1) // bs


def _make_manager_cls():
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    class _PromptEvictManager(FullAttentionManager):
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

        def remove_skipped_blocks(self, request_id, num_computed_tokens):
            blocks = self.req_to_blocks.get(request_id)
            plen = _PROMPT_LEN.get(request_id)
            if blocks and plen is not None:
                bs = self.block_size
                pnb = (plen + bs - 1) // bs
                if num_computed_tokens >= plen and request_id not in _PROMPT_DONE:
                    _PROMPT_DONE.add(request_id)
                    if _MODE == "chunkkv":
                        ent = _RETAINED.get(request_id)
                        if ent is not None:
                            prompt_nblk, keep = ent
                            nblk = min(prompt_nblk, len(blocks))
                            keepset = set(keep)
                            drop = [i for i in range(nblk) if i not in keepset]
                            if drop:
                                before = self.block_pool.get_num_free_blocks()
                                n = self._free_idx(request_id, drop)
                                if _LOG:
                                    print(f"[PE11] evict chunkkv req={request_id} "
                                          f"prompt_nblk={prompt_nblk} keep={len(keep)} "
                                          f"freed={n} pool {before}->"
                                          f"{self.block_pool.get_num_free_blocks()}", flush=True)
                    elif _MODE == "position":
                        sink = int(getattr(self.kv_cache_spec, "sink", _DEFAULTS["sink"]))
                        window = int(getattr(self.kv_cache_spec, "window", _DEFAULTS["window"]))
                        first = (sink + bs - 1) // bs
                        last = min(max(sink, num_computed_tokens - window) // bs, len(blocks))
                        if first < last:
                            self._free_idx(request_id, range(first, last))
                gw = int(_CFG.get("gen_window", 0) or 0)
                if gw > 0:
                    nblk_now = (num_computed_tokens + bs - 1) // bs
                    keep_from = max(pnb, (num_computed_tokens - gw) // bs)
                    tail = min(keep_from, len(blocks))
                    if pnb < tail:
                        before = self.block_pool.get_num_free_blocks()
                        n = self._free_idx(request_id, range(pnb, tail))
                        if _LOG and n:
                            print(f"[PE11] evict genwin req={request_id} "
                                  f"blk[{pnb},{tail}) nblk={nblk_now} freed={n} pool {before}->"
                                  f"{self.block_pool.get_num_free_blocks()}", flush=True)
            return super().remove_skipped_blocks(request_id, num_computed_tokens)

        def free(self, request_id):
            _PROMPT_LEN.pop(request_id, None)
            _PROMPT_DONE.discard(request_id)
            _RETAINED.pop(request_id, None)
            _IMP_ACC.pop(request_id, None)
            _VOTES.pop(request_id, None)
            for k in [k for k in _Q_BUF if k[1] == request_id]:
                _Q_BUF.pop(k, None)
            for k in [k for k in _Q_LASTL if k[1] == request_id]:
                _Q_LASTL.pop(k, None)
            for k in [k for k in _LAYER_SEEN if k[1] == request_id]:
                _LAYER_SEEN.discard(k)
            return super().free(request_id)

    return _PromptEvictManager


def _make_attention_cls(spec_cls, pe_params):
    from vllm.attention.layer import Attention

    class _PromptEvictAttention(Attention):
        def get_kv_cache_spec(self, vllm_config):
            global _MAX_LAYER
            try:
                n = int(vllm_config.model_config.hf_config.num_hidden_layers)
                _MAX_LAYER = max(_MAX_LAYER, n - 1)
            except Exception:
                pass
            base = super().get_kv_cache_spec(vllm_config)
            return spec_cls(
                block_size=base.block_size,
                num_kv_heads=base.num_kv_heads,
                head_size=base.head_size,
                dtype=base.dtype,
                sink=int(pe_params["sink"]),
                window=int(pe_params["window"]),
            )

    return _PromptEvictAttention


def _build_backend(torch):
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionImpl,
        TritonAttentionMetadataBuilder,
    )

    class PromptEvictImpl(TritonAttentionImpl):
        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output, *args, **kwargs):
            global _MAX_LAYER
            li = _layer_idx(layer)
            if li > _MAX_LAYER:
                _MAX_LAYER = li
            out = super().forward(layer, query, key, value, kv_cache,
                                  attn_metadata, output, *args, **kwargs)
            if _MODE == "chunkkv" and attn_metadata is not None:
                try:
                    _score_from_cache(layer, query, key, kv_cache, attn_metadata)
                except Exception as e:
                    print("[PE11] score warn:", type(e).__name__, str(e)[:160], flush=True)
            return out

    class PromptEvictBuilder(TritonAttentionMetadataBuilder):
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            md = super().build(common_prefix_len, common_attn_metadata, fast_build)
            global _BUILT_VER
            gw = int(_CFG.get("gen_window", 0) or 0)
            if not _RETAINED and gw == 0:
                return md
            if gw == 0 and _BUILT_VER == _VER:
                return md
            _BUILT_VER = _VER
            bt, sl = getattr(md, "block_table", None), getattr(md, "seq_lens", None)
            if bt is None or sl is None or bt.numel() == 0:
                return md
            bs = self.kv_cache_spec.block_size
            new_bt, new_sl = bt.clone(), sl.clone()
            changed = False
            for r in range(sl.shape[0]):
                req_id = _REQ_IDS[r] if r < len(_REQ_IDS) else None
                if req_id is None:
                    continue
                pnb = _prompt_nblk(req_id, bs)
                if pnb is None:
                    continue
                L = int(sl[r].item())
                nblk = (L + bs - 1) // bs
                ent = _RETAINED.get(req_id)
                prompt_keep = set(ent[1]) if ent is not None else set(range(min(pnb, nblk)))
                gen_from = pnb
                if gw > 0:
                    gen_from = max(pnb, min((L - gw) // bs, nblk))
                idx = sorted({i for i in prompt_keep if i < nblk}
                             | set(range(gen_from, nblk)))
                if len(idx) == nblk and idx == list(range(nblk)):
                    continue
                if idx:
                    src = bt[r][torch.tensor(idx, device=bt.device, dtype=torch.long)]
                    new_bt[r, :len(idx)] = src
                    new_sl[r] = sum(min(bs, L - i * bs) for i in idx)
                else:
                    new_sl[r] = 0
                new_bt[r, len(idx):] = 0
                changed = True
            if changed:
                md.block_table, md.seq_lens = new_bt, new_sl
                try:
                    md.max_seq_len = int(new_sl.max().item())
                except Exception:
                    pass
            return md

    class PromptEvictBackend(TritonAttentionBackend):
        @staticmethod
        def get_name() -> str:
            # 必须沿用已有枚举名：Attention.__init__ 会做 AttentionBackendEnum[get_name()]
            return "TRITON_ATTN"

        @classmethod
        def get_impl_cls(cls):
            return PromptEvictImpl

        @classmethod
        def get_builder_cls(cls):
            return PromptEvictBuilder

    return PromptEvictBackend


def install(params=None) -> None:
    import torch

    pe_params = dict(_DEFAULTS)
    if params:
        pe_params.update(params)
    _CFG.update(pe_params)

    from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map
    spec_cls = _make_spec_cls()
    spec_manager_map[spec_cls] = _make_manager_cls()

    import vllm.attention.layer as layer_mod
    backend_cls = _build_backend(torch)
    if not getattr(layer_mod, "_pe11_patched", False):
        orig_get = layer_mod.get_attn_backend
        layer_mod.get_attn_backend = lambda *a, **k: backend_cls
        layer_mod._pe11_orig_get = orig_get
        layer_mod._pe11_patched = True

    from vllm.v1.core.kv_cache_manager import KVCacheManager
    if not getattr(KVCacheManager, "_pe11_patched", False):
        orig_alloc = KVCacheManager.allocate_slots

        def patched_allocate_slots(self, request, *args, **kwargs):
            try:
                _PROMPT_LEN[request.request_id] = int(request.num_prompt_tokens)
            except Exception:
                pass
            return orig_alloc(self, request, *args, **kwargs)

        KVCacheManager.allocate_slots = patched_allocate_slots
        KVCacheManager._pe11_patched = True

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if not getattr(GPUModelRunner, "_pe11_patched", False):
        orig_prep = GPUModelRunner._prepare_inputs

        def patched_prepare_inputs(self, *args, **kwargs):
            out = orig_prep(self, *args, **kwargs)
            try:
                global _REQ_IDS
                _REQ_IDS = list(self.input_batch.req_ids)
            except Exception:
                pass
            return out

        GPUModelRunner._prepare_inputs = patched_prepare_inputs
        GPUModelRunner._pe11_patched = True

    import importlib
    arch = os.environ.get("PE_ARCH", "vllm.model_executor.models.qwen2")
    mod = importlib.import_module(arch)
    if getattr(mod, "_pe11_installed", False):
        return
    mod.Attention = _make_attention_cls(spec_cls, pe_params)
    mod._pe11_installed = True
    print(f"[PE11] installed mode={_MODE} params={pe_params} arch={arch}", flush=True)
