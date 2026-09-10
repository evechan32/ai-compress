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


def _install_allocator_dedupe(allocator) -> None:
    """让 allocator.free 幂等：忽略已在空闲集的槽，避免中途释放 + 请求结束重复释放。"""
    if getattr(allocator, "_kvx_dedupe", False):
        return
    free_set: set[int] = set()
    orig_free = allocator.free
    orig_alloc = allocator.alloc

    def free(idx):
        lst = idx.tolist() if hasattr(idx, "tolist") else list(idx)
        new = [i for i in lst if i not in free_set]
        for i in new:
            free_set.add(i)
        if new:
            t = torch.tensor(new, dtype=idx.dtype, device=idx.device)
            orig_free(t)

    def alloc(n):
        out = orig_alloc(n)
        if out is not None:
            for i in (out.tolist() if hasattr(out, "tolist") else out):
                free_set.discard(i)
        return out

    allocator.free = free
    allocator.alloc = alloc
    allocator._kvx_dedupe = True


class ScatterTritonBackend(TritonAttnBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kvx_head = int(os.environ.get("KVX_HEAD", "64"))
        self._kvx_window = int(os.environ.get("KVX_WINDOW", "32"))
        self._kvx_free = os.environ.get("KVX_FREE", "0") == "1"
        self._kvx_importance = os.environ.get("KVX_IMPORTANCE", "0") == "1"
        self._kvx_budget = int(os.environ.get("KVX_BUDGET", "256"))
        self._kvx_last_ptr = None
        self._kvx_freed = False

    def _importance_positions(self, rows, seq_len: int, budget: int, win: int):
        try:
            v = self.token_to_kv_pool.v_buffer[0][rows]
            score = v.float().norm(dim=-1).norm(dim=-1)
            k = min(budget, seq_len)
            top = torch.topk(score, k).indices.tolist()
            keep = set(top)
            start = max(0, seq_len - win)
            keep.update(range(start, seq_len))
            return sorted(keep)
        except Exception as e:
            print("KVX importance warn:", type(e).__name__, str(e)[:150], flush=True)
            return list(range(min(budget, seq_len))) + list(range(max(0, seq_len - win), seq_len))

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
                win = min(self._kvx_window, max(0, s - self._kvx_budget))
                if self._kvx_importance:
                    sel = self._importance_positions(rows, s, self._kvx_budget, win)
                    keep = rows[torch.tensor(sel, device=rows.device)]
                else:
                    head = min(self._kvx_head, s)
                    win = min(self._kvx_window, max(0, s - head))
                    parts = [rows[:head]]
                    if win > 0:
                        parts.append(rows[end - start - win:])
                    keep = torch.cat(parts)
                keep_rows.append(keep)
                new_indptr.append(new_indptr[-1] + int(keep.numel()))
                if self._kvx_free and i == 0 and not self._kvx_freed:
                    allocator = self.token_to_kv_pool_allocator
                    before = allocator.available_size()
                    _install_allocator_dedupe(allocator)
                    dropped = rows[head:end - start - win]
                    allocator.free(dropped)
                    try:
                        row = int(forward_batch.req_pool_indices[i].item())
                        self.req_to_token_pool.req_to_token[
                            row, head:end - start - win
                        ] = 0
                    except Exception as e:
                        print("KVX zero-row warn:", type(e).__name__, str(e)[:120], flush=True)
                    self._kvx_freed = True
                    print(f"KVX free: dropped={int(dropped.numel())} slots "
                          f"avail {before}->{allocator.available_size()}", flush=True)
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
