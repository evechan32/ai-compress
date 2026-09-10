"""SGLang 散点 KV 驱逐 backend 原型。

在 decode 元数据构建后，把每个请求的 KV 索引过滤为 [头 head 段 + 末尾 window 段]，
丢弃中段，从而验证"散点 KV 注意力"机制（SnapKV/H2O 的必要前提）。

限制：当前仅限制注意力可见范围，未释放 KV 池槽位（内存未回收）；选择策略为位置式，
非 attention 重要性打分（SnapKV 二期）。
"""
import os

import torch

from sglang.srt.layers.attention.attention_registry import register_attention_backend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend


class ScatterTritonBackend(TritonAttnBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kvx_head = int(os.environ.get("KVX_HEAD", "64"))
        self._kvx_window = int(os.environ.get("KVX_WINDOW", "32"))
        self._kvx_last_ptr = None

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        try:
            mode = getattr(forward_batch, "forward_mode", None)
            if mode is None or not mode.is_decode():
                return
            fm = self.forward_metadata
            if fm is None or fm.kv_indptr is None or fm.kv_indices is None:
                return
            bs = int(forward_batch.seq_lens.shape[0])
            seqlens = forward_batch.seq_lens[:bs].tolist()
            old_indptr = fm.kv_indptr[: bs + 1].tolist()
            old_idx = fm.kv_indices
            keep_rows = []
            new_indptr = [0]
            for i in range(bs):
                s = int(seqlens[i])
                start, end = old_indptr[i], old_indptr[i + 1]
                rows = old_idx[start:end]
                head = min(self._kvx_head, s)
                win = min(self._kvx_window, max(0, s - head))
                parts = [rows[:head]]
                if win > 0:
                    parts.append(rows[end - start - win:])
                keep = torch.cat(parts)
                keep_rows.append(keep)
                new_indptr.append(new_indptr[-1] + int(keep.numel()))
            fm.kv_indices = torch.cat(keep_rows) if keep_rows else old_idx[:0]
            fm.kv_indptr = torch.tensor(
                new_indptr, dtype=fm.kv_indptr.dtype, device=fm.kv_indptr.device
            )
            if self._kvx_last_ptr != new_indptr:
                self._kvx_last_ptr = new_indptr
                print(
                    f"KVX decode filter: old_ptr={old_indptr} new_ptr={new_indptr} "
                    f"head={self._kvx_head} window={self._kvx_window}",
                    flush=True,
                )
        except Exception as e:
            print("KVX ERR:", type(e).__name__, str(e)[:200], flush=True)


@register_attention_backend("kvx_scatter")
def create_kvx_scatter(runner):
    return ScatterTritonBackend(runner)
