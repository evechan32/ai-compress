# Chunk-Level KV Cache Eviction with Real Physical Block Release in vLLM

**A zero-fork, plugin-only integration of attention-scored prompt eviction**

*Draft v0.1 — 2026-09. 中文对照见每节 ZH 部分。*

> ⚠️ Draft status. Experiments are on a single model (Qwen2.5-1.5B-Instruct) with a
> distribution-based metric; task-accuracy benchmarks (LongBench F1), larger models,
> and throughput numbers are TODO. Engineering guards (TP>1, async scheduling,
> CUDA graph) are documented as limitations, not yet enforced at startup.

---

## Abstract

KV cache eviction is the standard way to bound long-context memory without retraining,
and chunk-level, attention-scored selection (ChunkKV/SnapKV-style) is empirically the
most accurate family. Yet in production serving frameworks such eviction is usually
*logical*: attention is masked so evicted positions are not attended, but the
underlying KV blocks are never returned to the pool, so memory is not actually saved.
Existing systems that do release memory (e.g. R-KV) fork the serving framework.

We present a **zero-fork** design that performs attention-scored, block-granular prompt
eviction with **real physical block release** inside vLLM 0.28, using only the public
extension points (`KVCacheSpecRegistry`, `subclass_attention_backend`,
`register_backend(AttentionBackendEnum.CUSTOM)`). Our contributions are architectural,
not algorithmic: (i) a solution to the *worker-scores / scheduler-frees* split via a
process-local, request-id-keyed rendezvous; (ii) a custom attention backend that hides
evicted blocks by compacting `block_table` and `seq_lens` at metadata-build time,
leaving `slot_mapping` untouched; (iii) support for **chunked prefill**, where the
observation window spans chunks and the prompt boundary is recovered from the KV
manager rather than the attention layer; and (iv) a **near-tie-robust evaluation
metric** (per-step distribution distance) because greedy token agreement is dominated
by numerical noise, not information loss. Under capacity-bound, decode-heavy serving we
measure **1.12×–1.26× throughput** from the released blocks. We additionally report a
silent-correctness hazard: vLLM's auto-selected FlashAttention-2 backend ignores the
R-SWA mask while the KV manager still frees blocks, causing attention to read freed
memory; we make this fail loudly.

**ZH 摘要**：KV 驱逐通常只是"逻辑驱逐"——掩掉但不释放块，显存没省；真正释放内存的系统
（如 R-KV）需要 fork 框架。本文给出一种**零 fork**方案，在 vLLM 0.28 上用公开扩展点实现
"注意力打分选块 + 真实物理释放"。贡献在架构而非算法：解决"worker 打分 / scheduler 释放"
的跨组件问题；用自定义 backend 在 metadata 构建期压实 `block_table`/`seq_lens`（不动
`slot_mapping`）来隐藏被驱逐块；支持 chunked prefill；并提出对 near-tie 鲁棒的分步分布
距离指标。另报告一个静默正确性隐患：vLLM 自动选的 FlashAttention-2 会忽略 R-SWA 掩码但
KV 管理器仍释放块，导致注意力读取已释放显存；我们让它显式报错。

---

## 1. Introduction

Long-context inference is memory-bound on the KV cache. Two families avoid retraining:
**eviction** (drop tokens) and **quantization** (fewer bits). Within eviction,
attention-scored, *chunk/block-level* selection — score tokens by the attention of a
trailing observation window, aggregate to blocks, keep the top blocks — is the most
accurate practical choice (SnapKV, PyramidKV, ChunkKV), and our own measurements agree.

Serving frameworks, however, expose only two shapes of eviction:

- **Contiguous** (streaming/sliding-window/R-SWA): expressible by a built-in mask; can
  release blocks. But its selection is positional and, as we measure, catastrophic for
  quality (Table 2).
- **Scattered token-level** (H2O/SnapKV): typically implemented by *masking* in a custom
  kernel — attention is corrected, but the pool still holds every block.

The gap we fill is **attention-scored, scattered, block-granular eviction that actually
returns blocks to the pool**, without forking vLLM.

The main obstacle is architectural. The eviction *decision* requires attention scores,
which are produced inside the model runner (worker); the *release* of KV blocks is
performed by the scheduler's KV cache manager. These live in different components and,
under multiprocessing, different processes. Contiguous R-SWA sidesteps this because its
rule is purely positional and can be recomputed by the scheduler.

**Contributions.**
1. A zero-fork architecture for attention-scored block eviction with physical release
   in vLLM 0.28, bridging the worker/scheduler split with a request-id-keyed process
   rendezvous (§3.2).
2. A custom attention backend that hides evicted blocks by compacting the attention
   `block_table` and `seq_lens` (§3.3) — no scattered mask, no custom kernel.
3. Chunked-prefill support: scoring moves to the backend `Impl` (which owns the full KV
   pool), the prompt boundary is recovered from the KV manager, and the observation
   window is buffered across chunks (§3.4).
4. A near-tie-robust metric that isolates eviction's information loss from numerical
   reduction-order noise, with a zero-noise floor (§3.5, §5).
5. A silent-correctness finding and guard for R-SWA under FlashAttention-2 (§5.4).

**ZH 引言**：长上下文推理受 KV 显存限制。两族免训练方法：驱逐与量化。驱逐中"注意力打分 +
块级选择"（SnapKV/PyramidKV/ChunkKV）最准，我们的实测也一致。但框架只暴露两种形状：
连续（滑窗/R-SWA，可释放但位置式、质量差）与散点 token 级（一般只掩不释放）。本文填补的
空白是：**注意力打分、散点、块级、且真正归还块的驱逐**，且不 fork vLLM。核心障碍是架构：
决策需要 worker 里的注意力分数，释放发生在 scheduler 的 KV manager，二者跨组件（多进程时
跨进程）；R-SWA 靠纯位置规则绕过。贡献见上（1–5）。

---

## 2. Related Work

**Eviction / sparse attention.** StreamingLLM (sink + window) is positional.
H2O and Scissorhands score by accumulated attention; TOVA by the last query.
SnapKV votes with a trailing observation window over the prompt; PyramidKV allocates
budget across layers; AdaKV per head; ChunkKV aggregates token scores to chunks and
evicts whole chunks, preserving semantic contiguity. Quest/InfLLM are query-aware
block retrieval. R-KV subtracts redundancy for long reasoning traces.
**Quantization.** KIVI (2/4-bit, per-channel K / per-token V), KVQuant (3-bit with
dense-and-sparse outliers). **Reconstruction.** KVzip scores a KV pair by its ability
to reconstruct the context, enabling query-agnostic reuse.
**Structural low-rank.** MLA (trained from scratch), Palu (post-hoc weight
decomposition + rank search + quantization; reports 50% low-rank as already "difficult
to fully preserve accuracy").

**Positioning.** We do not propose a new scoring rule; we show that a known rule
(ChunkKV-style block selection) can be made to *physically* release memory inside vLLM
without a fork, and we quantify it with a noise-robust metric. Prior integrations either
fork (R-KV's SGLang patch) or stay logical (NVIDIA kvpress masks with fake keys).

**ZH 相关工作**：驱逐（StreamingLLM/H2O/Scissorhands/TOVA/SnapKV/PyramidKV/AdaKV/
ChunkKV/Quest/InfLLM/R-KV）、量化（KIVI/KVQuant）、重建（KVzip）、结构低秩（MLA/Palu）。
定位：不提出新打分规则，而是让已知规则（ChunkKV 式块选择）在 vLLM 内**物理释放**且不 fork，
并用抗噪指标量化。已有集成要么 fork（R-KV），要么停在逻辑层（kvpress 假 key 掩码）。

---

## 3. Method

### 3.1 Setting and constraints

vLLM 0.28 pools KV in fixed-size blocks (default 16 tokens). All full-attention layers of
a model share one KV cache group and one block table: evicting a logical block removes it
from **all** layers. Consequently (i) block granularity is the natural unit, and (ii)
per-layer or per-head budgets are **not** expressible without separate KV groups — a
constraint that also rules out the "PyramidKV-style" layer budgets in our setting.

### 3.2 Architecture: bridging worker and scheduler

We register (all publicly):
- a `KVCacheSpec` (`PromptEvictSpec`) and manager (`PromptEvictManager`) via
  `KVCacheSpecRegistry.register`;
- a custom attention backend via `subclass_attention_backend` over `TritonAttentionBackend`;
- the model's `Attention` class is swapped for `PromptEvictAttention`, which returns our
  spec from `get_kv_cache_spec`.

The decision is written by the **worker** into a process-local dict
`_PE_RETAINED[request_id] = (prompt_nblocks, kept_blocks)`; the **scheduler** manager reads
it in `remove_skipped_blocks(...)` once prefill completes (`processed >= num_prompt_tokens`)
and frees every non-kept prompt block. This requires the scheduler and worker to share a
process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), which we adopt.

The observation window's queries (`obs` vectors per layer) and per-request state are
keyed by request id and cleaned on `manager.free(request_id)`.

### 3.3 Hiding evicted blocks without a scattered mask

The KV manager replaces freed blocks with `null_block` in `req_to_blocks`, but — we
verified — the **worker's `block_table` is not updated with these nulls**; masking, not
nulling, is what makes evicted positions invisible (this is also how R-SWA works). vLLM
0.28 provides no scattered mask.

Instead, our backend's metadata builder, after `super().build()`, **compacts** the
attention `block_table` to contain only retained (plus generated) blocks and recomputes
`seq_lens` as the true token count of the retained blocks. KV writes are unaffected
because they use `slot_mapping`, not the attention block table. When nothing is dropped
we skip compaction entirely, which keeps the plugin numerically neutral.

### 3.4 Chunked prefill

A long prompt is split into chunks; scoring from the layer's current-chunk `key` and
requiring `q_len == L` silently disables eviction for prompts longer than
`max_num_batched_tokens` — i.e. exactly the target regime. We therefore:

1. score inside the backend `Impl.forward`, which receives the whole KV pool
   (`kv_cache`) and runs *after* `super().forward()` (so the current chunk's KV is
   already written);
2. recover the prompt boundary from the manager's `num_prompt_tokens` (the attention
   layer does not see it; we verified the metadata's `rswa_prefix_lens` is not the
   prompt length);
3. buffer the last `obs` query vectors **across chunks** per layer, so a final chunk
   shorter than `obs` does not truncate the window.

### 3.5 Scoring and selection

For the prompt-final chunk, each voting layer gathers all `L` keys from the pool
(`kv_cache[block_table]`), computes the observation-window attention
`softmax(q_w K^T/√d)` for the last `w = min(obs, q_len)` queries, sums over `w` and means
over heads to obtain a per-token importance, aggregates to blocks, and keeps the top
blocks until a token budget (`budget`, or `ratio·L`) is reached, unioned with `sink` and
the window blocks. With `vote_layers = K`, the per-token importances of the last `K`
layers are averaged before selection (default `K=1`).

### 3.6 A near-tie-robust metric

Greedy token agreement is unusable here: enabling scoring (even with no eviction)
deterministically flips ≈7% of greedy tokens on a small degenerate model, because the
extra GPU ops shift allocator/reduction order and the model has many near-ties. We
instead compare, at each decode step, the top-k log-probability distributions of two
configurations, and only while their token prefixes still agree. We report the divergence
rate, the mean first-divergence step, and the mean KL. Running the same configuration
twice yields **divergence rate 0 and KL 0** — a zero noise floor.

**ZH 方法**：3.1 约束：vLLM 所有 full-attn 层共享一个 KV group/block table → 块粒度是
自然单位，逐层/逐头预算不可表达。3.2 架构：注册自定义 spec+manager（`KVCacheSpecRegistry`）
与 backend（`subclass_attention_backend`），替换模型 `Attention` 返回自定义 spec；worker 打分写入
进程内按 request_id 键的表，scheduler 的 manager 在 prefill 完成后释放非保留块（需单进程）。
3.3 隐藏被驱逐块：manager 只把 `req_to_blocks` 换成 null，**worker 的 block_table 不同步**；
我们在 builder 里压实 attention 的 `block_table` 并重算 `seq_lens`，不动 `slot_mapping`；无驱逐
时跳过，保持数值中性。3.4 chunked prefill：打分移到 backend `Impl`（拿得到整池、且在 super 之后），
prompt 边界取自 manager 的 `num_prompt_tokens`，观测窗 query 跨 chunk 缓存。3.5 打分与选择：
末 chunk 用观测窗注意力→按块聚合→取 top 至预算，并上 sink/窗口块；可选末 K 层平均。
3.6 指标：贪心 token 一致率不可用（打分本身即确定性翻转 ~7% near-tie）；改用分步 top-k
分布距离，同配置两次 → div_rate 0、KL 0。

---

## 4. Implementation on vLLM 0.28

All extension points are public; the plugin is ~450 lines (`kvcompress/pevict.py`). A
`passthrough` mode (backsend+spec installed, no scoring/eviction) is numerically
identical to vanilla vLLM, which we use as an isolation control.

Configuration: single process, TP=PP=1, `enforce_eager=True` (scoring performs
dynamic-shape GPU work and host syncs inside the forward), and `TRITON_ATTN`. The
backend guard fails loudly if a non-mask-capable backend is selected.

---

## 5. Evaluation

**Setup.** Qwen2.5-1.5B-Instruct (28 layers, 2 KV heads, head dim 128, block 16) on
2×RTX 5070; 5 LongBench tasks × 30 samples; reference = the same plugin with no eviction
(`budget ≥ prompt`). Metric §3.6.

**5.1 Neutrality / isolation.** `passthrough` = divergence 0 vs vanilla. No-eviction
scoring shifts ≈7% of greedy tokens but is deterministic run-to-run — the reason we
abandoned greedy agreement.

**5.2 Attention-scored vs positional (Table 2).** At comparable retention, positional
sink+window diverges almost always (div_rate 1.000 at sink=0/window=1024; 0.933 at
sink=64), while ChunkKV-style block selection diverges far less (0.147 at 80% kept,
0.373 at 50%). Attention-scored selection is what makes eviction usable.

**5.3 Ablations (Table 1).** Divergence rate rises monotonically with compression
(0.147/0.373/0.527 at keep 80/50/30%). Observation window: **smaller is better**
(16→0.207 vs 256→0.373) because the window is always retained, so a large window steals
budget from top-scored blocks and dilutes the vote with pre-question context. Block
aggregation (`sum`/`mean`/`max`) is a wash (`mean≡sum` for uniform blocks). Multi-layer
voting is marginal (K=2/4: 0.267/0.253) and costs linearly (K=8 OOM).

**5.4 A silent-correctness finding.** vLLM auto-selects FlashAttention-2, which sets but
never consumes the R-SWA mask; under FA2, R-SWA produces byte-identical output to
baseline (SHA1 match) while the manager still frees blocks — attention reads freed
memory. Under `TRITON_ATTN` the eviction is real and deterministic (window=64 differs
from baseline; window=2048 is neutral). We add a startup guard that refuses non-mask
backends (or substitutes TritonAttentionBackend).

**5.5 Throughput under capacity pressure (Table 3).** A decode-heavy workload
(M=128 concurrent requests, ~3 000-token prompts, 512 generated tokens; pool = 119 632
tokens, ≈39 concurrent 3k-token requests) is capacity-bound: without release, vLLM
queues requests; with release, more run concurrently. Throughput rises monotonically
with compression — **1.12× at keep 50%, 1.26× at keep 30%** over no-eviction, stable
across repeated runs (3130/3133 tok/s). Under a prefill-dominated workload (T=8 or 256)
there is **no gain**, because eviction saves KV memory but not prefill compute, and
scoring adds ≈5% overhead. This delineates the regime where physical release pays off:
long decode under KV-capacity pressure.

**ZH 实验**：模型 Qwen2.5-1.5B，5 任务 × 30 样本，参考=同插件不驱逐。5.1 中性：passthrough
与原生逐字一致；不驱逐的打分本身翻转 ~7% 贪心 token（确定性），故弃用贪心指标。5.2 结构化
对比：位置式几乎总是分歧（1.000 / 0.933），块级注意力选择显著更低（0.147 / 0.373）。5.3 消融：
压缩越高分歧越多（0.147/0.373/0.527）；**观测窗越小越好**（16→0.207 vs 256→0.373，因窗口恒被
保留、大窗稀释投票且占预算）；块内聚合 sum/mean/max 基本等价；多层投票边际（K=2/4 →
0.267/0.253）且成本线性（K=8 OOM）。5.4 静默隐患：FA2 设了 R-SWA 元数据但从不消费掩码，
输出与基线逐字相同却已释放块 → 读已释放显存；TRITON_ATTN 下驱逐真实且确定。已加启动护栏。
5.5 吞吐（容量受限 + decode 为主）：M=128、512 生成，池 119,632 token（约 39 并发）；释放后
吞吐随压缩比单调上升——keep50% 1.12×、keep30% 1.26×（重复运行稳定）。prefill 为主（T=8/256）
时无收益：驱逐省显存不省 prefill 计算，且打分开销约 5%。这界定了物理释放的收益区间：**KV
容量受限 + 长 decode**。

### Tables

**Table 1. Ablations (div_rate ↓, vs no-eviction ref; lower is better).**

| compression | keep 80% | keep 50% | keep 30% |
|---|---|---|---|
| div_rate | 0.147 | 0.373 | 0.527 |

| obs | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| div_rate | **0.207** | 0.247 | 0.280 | 0.360 | 0.373 |

| vote_layers | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| div_rate | 0.280 | 0.267 | 0.253 | OOM |

**Table 2. Attention-scored vs positional (div_rate, vs no-eviction ref).**

| method | setting | div_rate |
|---|---|---|
| positional | sink 0, window 1024 | 1.000 |
| positional | sink 64, window 1024 | 0.933 |
| ChunkKV-style | keep 80% | 0.147 |
| ChunkKV-style | keep 50% | 0.373 |

**Table 3. Throughput under capacity pressure (M=128, T=512, ~3k-token prompts).**

| config | retained | tok/s | speedup |
|---|---|---|---|
| no eviction (passthrough) | 100% | 3130 / 3133 | 1.00× |
| ChunkKV-style, keep 50% | ~50% | 3496 | 1.12× |
| ChunkKV-style, keep 30% | ~30% | 3956 | 1.26× |

*(Prefill-dominated workloads show no gain: T=8 → 144.4 vs 142.1 tok/s.)*

---

## 6. Limitations and Future Work

- **Scale.** One 1.5B model; no task-accuracy benchmark (LongBench F1/EM), no larger
  models. Throughput gains are demonstrated only in a capacity-bound, decode-heavy
  regime; prefill-dominated workloads show none (eviction saves memory, not prefill
  compute).
- **Metric.** Distribution distance measures deviation from no-eviction, not task
  correctness; a large KL may still be a valid alternative continuation.
- **Deployment guards not yet enforced**: TP>1, pipeline parallelism, async scheduling,
  CUDA graphs, and multi-KV-group models are unsupported and currently only documented.
  We are adding startup assertions.
- **Prefix caching.** Blocks shared via the prefix cache are ref-counted; our early free
  decrements the request's reference. Single-config tests pass, but the accounting under
  interleaved multi-request sharing needs proof.
- **Per-layer/per-head budgets are inexpressible** with a single shared KV group — a
  structural constraint of the framework, worth stating explicitly as the reason
  PyramidKV/AdaKV-style allocation cannot be reproduced zero-fork.
- **Query-agnostic** (prefix-caching/multi-turn) scoring is not implemented; KVzip-style
  reconstruction is a natural next step.

## 7. Conclusion

Attention-scored, block-granular KV eviction can be integrated into vLLM 0.28 with real
physical block release, without forking, by bridging the worker/scheduler split with a
process-local rendezvous and compacting the attention metadata in a custom backend.
Chunked prefill is supported by moving scoring into the backend and buffering the
observation window across chunks. We also show that greedy-token evaluation is unreliable
for this class of change and provide a noise-robust alternative, and we surface a silent
mask-ignoring hazard in FlashAttention-2.

---

## Appendix A. Reproduction (server)

```bash
# no-eviction reference
PE_MODE=chunkkv PE_BACKEND=TRITON_ATTN PE_BUDGET=4096 PE_OBS=256 PE_N=30 \
  PE_TAG=ref python -m bench.p3_metric
# eviction at 50%
PE_MODE=chunkkv PE_BACKEND=TRITON_ATTN PE_RATIO=0.5 PE_OBS=16 PE_N=30 \
  PE_TAG=evict python -m bench.p3_metric
# compare
PE_REF_TAG=ref PE_TAG=evict python -m bench.p3_metric compare
```
