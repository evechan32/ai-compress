"""vLLM **0.11.2** 端口 —— S1：管路验证（真释放 + 隐藏；暂不含注意力打分）。

与 0.28 的差异（均已实测确认）：
  - Attention 路径：`vllm.attention.layer.Attention`
  - spec→manager：0.11.2 用 `spec_manager_map[type(spec)]` **精确类型**查表 → 必须自己注册
  - 无 `KVCacheSpecRegistry`：直接 override `Attention.get_kv_cache_spec()`
  - `remove_skipped_blocks(request_id, num_computed_tokens)` **无 num_prompt_tokens**
    → patch `KVCacheManager.allocate_slots`，从 `request.num_prompt_tokens` 记录
  - 无 `subclass_attention_backend`：手动子类化 + patch `vllm.attention.layer.get_attn_backend`
  - 行号→请求 id：patch `GPUModelRunner._prepare_inputs`（0.28 用的是 model_states.default）

S1 只做 **position 模式（sink+window）**：验证能真释放块、且注意力通过压实 block_table
看不到被释放的块。打分（chunkkv）留到 S2。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_LOG = os.environ.get("PE_LOG", "0") == "1"

_DEFAULTS = {
    "sink": int(os.environ.get("PE_SINK", "64")),
    "window": int(os.environ.get("PE_WINDOW", "1024")),
}

# 引擎侧状态（单进程 / TP=1）
_PE_PROMPT_LEN: dict[str, int] = {}        # req_id -> prompt token 数
_PE_DROP: dict[str, tuple[int, int]] = {}  # req_id -> (first_blk, last_blk) 被释放的块区间
_PE_REQ_IDS: list[str] = []                # 行号 -> req_id（由 _prepare_inputs 填充）


def _make_spec_cls():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    @dataclass(frozen=True)
    class _PromptEvictSpec(FullAttentionSpec):
        sink: int = 64
        window: int = 1024

    return _PromptEvictSpec


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
            """0.11.2 签名无 num_prompt_tokens（prompt 长度由 allocate_slots 侧记录）。"""
            plen = _PE_PROMPT_LEN.get(request_id)
            if (plen is not None and num_computed_tokens >= plen
                    and request_id not in _PE_DROP):
                blocks = self.req_to_blocks.get(request_id)
                if blocks:
                    bs = self.block_size
                    sink = int(getattr(self.kv_cache_spec, "sink", _DEFAULTS["sink"]))
                    window = int(getattr(self.kv_cache_spec, "window", _DEFAULTS["window"]))
                    first = (sink + bs - 1) // bs
                    last = min(max(sink, num_computed_tokens - window) // bs, len(blocks))
                    if first < last:
                        before = self.block_pool.get_num_free_blocks()
                        n = self._free_idx(request_id, range(first, last))
                        _PE_DROP[request_id] = (first, last)
                        if _LOG:
                            print(f"[PE11] evict req={request_id} blk[{first},{last}) "
                                  f"freed={n} pool {before}->"
                                  f"{self.block_pool.get_num_free_blocks()}", flush=True)
            return super().remove_skipped_blocks(request_id, num_computed_tokens)

        def free(self, request_id):
            _PE_PROMPT_LEN.pop(request_id, None)
            _PE_DROP.pop(request_id, None)
            return super().free(request_id)

    return _PromptEvictManager


def _make_attention_cls(spec_cls, pe_params):
    from vllm.attention.layer import Attention

    class _PromptEvictAttention(Attention):
        def get_kv_cache_spec(self, vllm_config):
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
        pass  # S1：只验证管路，不打分

    class PromptEvictBuilder(TritonAttentionMetadataBuilder):
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            md = super().build(common_prefix_len, common_attn_metadata, fast_build)
            if not _PE_DROP:
                return md
            bt, sl = getattr(md, "block_table", None), getattr(md, "seq_lens", None)
            if bt is None or sl is None or bt.numel() == 0:
                return md
            bs = self.kv_cache_spec.block_size
            new_bt, new_sl = bt.clone(), sl.clone()
            changed = False
            for r in range(sl.shape[0]):
                req_id = _PE_REQ_IDS[r] if r < len(_PE_REQ_IDS) else None
                span = _PE_DROP.get(req_id) if req_id else None
                if span is None:
                    continue
                L = int(sl[r].item())
                nblk = (L + bs - 1) // bs
                first = min(span[0], nblk)
                last = min(span[1], nblk)
                idx = list(range(first)) + list(range(last, nblk))
                if len(idx) == nblk:
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

    # 1) spec -> manager（0.11.2 用精确类型查表）
    from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map
    spec_cls = _make_spec_cls()
    spec_manager_map[spec_cls] = _make_manager_cls()

    # 2) 后端注入
    import vllm.attention.layer as layer_mod
    backend_cls = _build_backend(torch)

    if not getattr(layer_mod, "_pe11_patched", False):
        orig_get = layer_mod.get_attn_backend

        def patched_get_attn_backend(*args, **kwargs):
            return backend_cls

        layer_mod.get_attn_backend = patched_get_attn_backend
        layer_mod._pe11_orig_get = orig_get
        layer_mod._pe11_patched = True

    # 3) 记录 prompt 长度（0.11.2 不再通过 remove_skipped_blocks 传）
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    if not getattr(KVCacheManager, "_pe11_patched", False):
        orig_alloc = KVCacheManager.allocate_slots

        def patched_allocate_slots(self, request, *args, **kwargs):
            try:
                _PE_PROMPT_LEN[request.request_id] = int(request.num_prompt_tokens)
            except Exception:
                pass
            return orig_alloc(self, request, *args, **kwargs)

        KVCacheManager.allocate_slots = patched_allocate_slots
        KVCacheManager._pe11_patched = True

    # 4) 行号 -> 请求 id（元数据构建期要用）
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    if not getattr(GPUModelRunner, "_pe11_patched", False):
        orig_prep = GPUModelRunner._prepare_inputs

        def patched_prepare_inputs(self, *args, **kwargs):
            out = orig_prep(self, *args, **kwargs)
            try:
                global _PE_REQ_IDS
                _PE_REQ_IDS = list(self.input_batch.req_ids)
            except Exception:
                pass
            return out

        GPUModelRunner._prepare_inputs = patched_prepare_inputs
        GPUModelRunner._pe11_patched = True

    # 5) 换掉模型模块的 Attention
    import importlib
    arch = os.environ.get("PE_ARCH", "vllm.model_executor.models.qwen2")
    mod = importlib.import_module(arch)
    if getattr(mod, "_pe11_installed", False):
        return
    mod.Attention = _make_attention_cls(spec_cls, pe_params)
    mod._pe11_installed = True
    print(f"[PE11] installed params={pe_params} arch={arch}", flush=True)
