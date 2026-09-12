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
}

_PE_REQ_IDS: list = []
_PE_RETAINED: dict[str, tuple[int, tuple[int, ...]]] = {}
_PE_LAYER_SEEN: set = set()


@dataclasses.dataclass(frozen=True, kw_only=True)
class PromptEvictSpec(FullAttentionSpec):
    sink: int = 64
    window: int = 1024
    budget: int = 256
    chunk: int = 16
    obs: int = 64
    ratio: float = 0.0

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
    except Exception:
        return int(os.environ.get("PE_BLOCK", "16"))


class PromptEvictAttention(Attention):
    def __init__(self, *args, pe_params=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pe_params = pe_params or dict(_DEFAULTS)

    def get_kv_cache_spec(self, vllm_config):
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
        if _LOG and nblk > 4 and self.layer_name not in _PE_LAYER_SEEN:
            _PE_LAYER_SEEN.add(self.layer_name)
            print(f"[PE] score layer={self.layer_name} keep={len(keep)}/{nblk}",
                  flush=True)

    def forward(self, query, key, value, output_shape=None, output_dtype=None):
        if _MODE in ("chunkkv", "mask") and key is not None:
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
        if _LOG and (request_id, "ev") not in self._pe_logged:
            self._pe_logged.add((request_id, "ev"))
            print(f"[PE] evict chunkkv req={request_id} prompt_nblk={prompt_nblk} "
                  f"keep={len(keep)} freed={n} "
                  f"pool {before}->{self.block_pool.get_num_free_blocks()}",
                  flush=True)

    def remove_skipped_blocks(self, request_id, processed_computed_tokens,
                              num_prompt_tokens=None):
        if num_prompt_tokens is not None:
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
            _PE_REQ_IDS[:] = list(input_batch.req_ids)[: input_batch.num_reqs]
        except Exception:
            pass
        return orig(self, input_batch, *args, **kwargs)

    cls.prepare_attn = prepare
    cls._pe_reqids_patched = True


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
    from vllm.v1.attention.backend import subclass_attention_backend
    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionMetadataBuilder,
    )

    class PromptEvictBuilder(TritonAttentionMetadataBuilder):
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            md = super().build(common_prefix_len, common_attn_metadata, fast_build)
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
                if not idx:
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

    return subclass_attention_backend("PromptEvict", TritonAttentionBackend,
                                      PromptEvictBuilder)


def _patch_custom_backend():
    import vllm.model_executor.layers.attention.attention as attn_mod
    if getattr(attn_mod, "_pe_backend_patched", False):
        return
    backend = _build_flex_backend() if _MODE == "mask" else _build_backend()
    want = "FlexAttentionBackend" if _MODE == "mask" else "TritonAttentionBackend"
    orig = attn_mod.get_attn_backend

    def sel(*args, **kwargs):
        cls = orig(*args, **kwargs)
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
    elif _MODE in ("chunkkv", "free", "mask"):
        _patch_req_ids()
        _patch_custom_backend()
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
