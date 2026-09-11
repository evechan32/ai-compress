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
        self._kvx_obs = int(os.environ.get("KVX_OBS", "64"))
        self._kvx_headagg = os.environ.get("KVX_HEADAGG", "mean")
        self._kvx_mode = os.environ.get("KVX_MODE", "position")
        self._kvx_page = int(os.environ.get("KVX_PAGE", "16"))
        self._kvx_topk_pages = int(os.environ.get("KVX_TOPK_PAGES", "16"))
        self._kvx_sink = int(os.environ.get("KVX_SINK", "64"))
        self._kvx_last_q = None
        self._kvx_scores = None
        self._kvx_sum = None
        self._kvx_cnt = 0
        self._kvx_sel = {}
        self._kvx_last_ptr = None
        self._kvx_freed = False

    def _quest_positions(self, rows, seq_len: int, win: int):
        try:
            kbuf, _ = self.token_to_kv_pool.get_kv_buffer(0)
            kk = kbuf[rows].float()
            hkv, d = kk.shape[1], kk.shape[2]
            page = self._kvx_page
            n_pages = (seq_len + page - 1) // page
            pad = n_pages * page - seq_len
            if pad:
                kk = torch.cat([kk, kk[-1:].expand(pad, hkv, d)], dim=0)
            kk = kk.reshape(n_pages, page, hkv, d)
            kmax = kk.max(dim=1).values
            kmin = kk.min(dim=1).values
            q = self._kvx_last_q
            if q is None:
                raise ValueError("no query captured")
            qf = q.float()
            rep = max(1, qf.shape[0] // hkv)
            score = torch.zeros(n_pages, device=kk.device)
            for h in range(qf.shape[0]):
                kh_max = kmax[:, h // rep, :]
                kh_min = kmin[:, h // rep, :]
                up = torch.maximum(qf[h] * kh_max, qf[h] * kh_min).sum(dim=-1)
                score += up
            k = min(self._kvx_topk_pages, n_pages)
            top_pages = torch.topk(score, k).indices.tolist()
            keep = set(range(min(self._kvx_sink, seq_len)))
            for pg in top_pages:
                keep.update(range(pg * page, min((pg + 1) * page, seq_len)))
            keep.update(range(max(0, seq_len - win), seq_len))
            return sorted(keep)
        except Exception as e:
            print("KVX quest warn:", type(e).__name__, str(e)[:150], flush=True)
            return list(range(min(self._kvx_topk_pages * self._kvx_page, seq_len))) + \
                list(range(max(0, seq_len - win), seq_len))

    def _importance_positions(self, rows, seq_len: int, budget: int, win: int):
        try:
            key = None
            try:
                key = int(self._kvx_cur_key)
            except Exception:
                key = None
            cached = self._kvx_sel.get(key) if key is not None else None
            if cached is None:
                score = self._kvx_scores
                if score is None or score.numel() == 0:
                    raise ValueError("no attention scores captured")
                t = min(score.numel(), seq_len)
                k = min(budget, t)
                cached = sorted(set(torch.topk(score[:t], k).indices.tolist()))
                if key is not None:
                    self._kvx_sel[key] = cached
                self._kvx_sum = None
                self._kvx_cnt = 0
                self._kvx_scores = None
            keep = set(p for p in cached if p < seq_len)
            keep.update(range(max(0, seq_len - win), seq_len))
            return sorted(keep)
        except Exception as e:
            print("KVX importance warn:", type(e).__name__, str(e)[:150], flush=True)
            return list(range(min(budget, seq_len))) + list(range(max(0, seq_len - win), seq_len))

    def _kvx_maybe_score(self, q, k, layer, forward_batch):
        if not self._kvx_importance or q is None:
            return
        try:
            fm = forward_batch.forward_mode
            if getattr(fm, "is_decode", None) and fm.is_decode():
                return
            if k is None:
                kbuf, _ = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                k = kbuf[forward_batch.out_cache_loc]
                global _KVX_KFIX_LOGGED
                if not globals().get("_KVX_KFIX_LOGGED"):
                    globals()["_KVX_KFIX_LOGGED"] = True
                    print("KVX score: k from pool path", flush=True)
            lens = getattr(forward_batch, "extend_seq_lens", None)
            if lens is not None and len(lens) > 1:
                return
            t = q.shape[0]
            hq, hk, d = layer.tp_q_head_num, layer.tp_k_head_num, layer.qk_head_dim
            q3 = q.reshape(t, hq, d).float()
            k3 = k.reshape(k.shape[0], hk, d).float()
            w = min(self._kvx_obs, t)
            rep = max(1, hq // hk)
            if self._kvx_headagg == "max":
                imp = torch.zeros(t, device=q.device, dtype=torch.float32)
                for h in range(hq):
                    kh = k3[:, h // rep, :]
                    sc = (q3[t - w:t, h, :] @ kh.t()) / (d ** 0.5)
                    imp = torch.maximum(imp, torch.softmax(sc, dim=-1).sum(dim=0))
            else:
                imp = torch.zeros(t, device=q.device, dtype=torch.float32)
                for h in range(hq):
                    kh = k3[:, h // rep, :]
                    sc = (q3[t - w:t, h, :] @ kh.t()) / (d ** 0.5)
                    imp += torch.softmax(sc, dim=-1).sum(dim=0)
                imp /= max(1, hq)
            if self._kvx_sum is None or self._kvx_sum.numel() != t:
                self._kvx_sum = imp
                self._kvx_cnt = 1
            else:
                self._kvx_sum = self._kvx_sum + imp
                self._kvx_cnt += 1
            self._kvx_scores = self._kvx_sum / self._kvx_cnt
        except Exception as e:
            print("KVX score warn:", type(e).__name__, str(e)[:150], flush=True)

    def _kvx_discover(self, name, a, kw):
        seen = globals().setdefault("_KVX_SEEN", set())
        if name in seen:
            return
        seen.add(name)
        k = kw.get("k", a[1] if len(a) > 1 else "?")
        print(f"KVX discover: method={name} nargs={len(a)} k_is_none={k is None if not isinstance(k, str) else k}", flush=True)

    def forward_extend(self, *args, **kwargs):
        self._kvx_discover("forward_extend", args, kwargs)
        if len(args) >= 5:
            self._kvx_maybe_score(args[0], args[1], args[3], args[4])
            self._kvx_capture_q(args[0], args[3])
        return super().forward_extend(*args, **kwargs)

    def _forward_extend_unified(self, *args, **kwargs):
        self._kvx_discover("_forward_extend_unified", args, kwargs)
        return super()._forward_extend_unified(*args, **kwargs)

    def _kvx_capture_q(self, q, layer):
        try:
            if self._kvx_mode == "quest" and getattr(layer, "layer_id", -1) == 0 \
                    and q is not None:
                hq, d = layer.tp_q_head_num, layer.qk_head_dim
                self._kvx_last_q = q.reshape(-1, hq, d)[-1].detach()
        except Exception:
            pass

    def forward_decode(self, q, k, v, layer, forward_batch, *args, **kwargs):
        self._kvx_capture_q(q, layer)
        return super().forward_decode(q, k, v, layer, forward_batch, *args, **kwargs)

    def forward_mixed(self, *args, **kwargs):
        self._kvx_discover("forward_mixed", args, kwargs)
        return super().forward_mixed(*args, **kwargs)

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
                try:
                    self._kvx_cur_key = int(forward_batch.req_pool_indices[i].item())
                except Exception:
                    self._kvx_cur_key = i
                win = min(self._kvx_window, max(0, s - self._kvx_budget))
                if self._kvx_mode == "quest":
                    sel = self._quest_positions(rows, s, self._kvx_window)
                    keep = rows[torch.tensor(sel, device=rows.device)]
                elif self._kvx_importance:
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
