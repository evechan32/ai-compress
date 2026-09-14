# vLLM prompt 侧分块 KV 驱逐（P1–P3）

> 目标：长 prompt 的 KV 在 prefill 后**真实释放显存**，且近无损。
> 全部实现为 vLLM 0.28 插件（零 fork），代码在 `kvcompress/pevict.py`。

## 结论速览

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1 | 自定义 KVCacheSpec/Manager/Attention 注册 + 运行时接口观测 | ✅ |
| P2 | 位置式 sink+window 驱逐（复用 R-SWA 掩码）+ 物理释放 | ✅ 机制正确 |
| P3-B | 注意力打分选块（ChunkKV 式）+ 自定义 backend 压制 block_table | ✅ **99.84% / 100% token 一致** |
| P3-A | 同上，改走 FlexAttention `physical_to_logical` 掩码 | ⚠️ 93% 一致性，劣于 B |

质量指标为 **decode 侧逐 token 一致率**（baseline vs 驱逐，贪心 16 token，5 任务 × 60 条 = 300 条）。
位置式（P2）为对照：**22.4%**，说明"注意力选块"才是近无损的关键。

## 关键机制

### 物理释放与掩码是两件事
`RSWAManager._remove_blocks_in_range` 把被驱逐块在 `req_to_blocks` 里换成 `null_block` 并归还池子
（真释放）。但 **worker 侧的 block_table 不会同步这些 null**（实测仍为原始块 id）。
因此"看不到被驱逐 KV"必须靠 **attention 掩码/压制**，不能依赖 null 块。

### B：自定义 backend 压制 block_table
- `subclass_attention_backend("PromptEvict", TritonAttentionBackend, PromptEvictBuilder)`
- builder 在 `super().build()` 后，把被驱逐块从 attention 用的 `block_table` 压掉，
  并重算 `seq_lens`（保留块的真实 token 数）；KV 写入走 `slot_mapping`，不受影响。
- triton 统一 kernel 只 gather 保留 KV → 被驱逐位置读不到。

### A：FlexAttention 掩码
- `physical_to_logical[req, block] = -1` 会被 flex 判为 invalid 并掩掉。
- 实现为把被驱逐逻辑块在**克隆的 block_table** 里置 0、重算映射并写回。
- **缺陷**：flex 的 block mask 在 `super().build()` 中已用旧映射构建，事后改映射不触发重建，
  导致部分驱逐未生效（93% vs B 的 99.8%）。

## 打分（ChunkKV 式）

- 在 `PromptEvictAttention.forward`（prefill）取最后 `obs` 个 query，对全序列 key 做注意力，
  按 block 聚合，取 top 块直到累计 token 达 `budget`，再并上 sink 与观察窗。
- 逐层 overwrite，最终由**最后一层**决定（SnapKV 式）。
- `_PE_REQ_IDS` 由 patch `DefaultModelState.prepare_attn` 从 `input_batch.req_ids` 注入，
  保留集按 **request_id** 键（不能用块签名——prefix caching 会让不同请求共享前缀块，
  实测 `pc=1` 时签名碰撞导致 1/4 正确）。

## 验证

| 项 | 结果 |
|---|---|
| 中性性（budget ≥ prompt） | 输出逐字等于基线，freed=0 |
| 物理释放 | 202 块中释放 186–196（`pool` 实增） |
| needle 强驱逐（obs=8,budget=0） | 基线 `74829`，B 与 A 均丢失该信息 |
| 多请求（4 并发，needle 位置各异） | 4/4 |
| prefix caching 开启 | 4/4（修掉签名碰撞后） |
| 128-token 长 decode | 4/4 |
| 质量一致率 | B: 99.84%@0.5 / 100%@0.8；A: 93.2%/93.4% |

## 环境要求 / 坑

- **必须 `--attention-backend TRITON_ATTN`**（B）或 `FLEX_ATTENTION`（A）。
- 单进程：`VLLM_ENABLE_V1_MULTIPROCESSING=0`，TP=1，`enforce_eager=True`
  （保留集经进程内共享表传递）。
- **R-SWA 掩码只有 TRITON_ATTN / FlexAttention / FA4 实现**；vLLM 默认自动选 FLASH_ATTN(FA2)，
  FA2 会**静默忽略掩码**却仍释放块 → 读已释放显存。见 `kvcompress/adapter.py` 的 backend guard。

## 复现

```bash
# B：注意力选块，50% 保留，物理释放
PE_MODE=chunkkv PE_BACKEND=TRITON_ATTN PE_RATIO=0.5 PE_OBS=256 PE_SINK=0 \
  python -m bench.p3_agree

# needle 机制验证
PE_MODE=chunkkv PE_BACKEND=TRITON_ATTN PE_BUDGET=64 PE_OBS=64 python -m bench.p2_needle
```

脚本：`bench/p1_pevict_run.py`（P1）、`bench/p2_needle.py`（机制）、
`bench/p3_validate.py`（多请求/prefix caching）、`bench/p3_agree.py`（质量一致率）、
`bench/p3_quality.py`（teacher-forced NLL，**注意**：prompt_logprobs 在 prefill 计算，
不受 post-prefill 驱逐影响，故此指标无效，仅留档）。

## 附：kvcompress RSWA 的 FA2 问题（已修）

`kvcompress` 的 RSWA 在 FA2 下输出与基线逐字相同（SHA1 相同）——掩码被忽略，
注意力仍在读已释放块。已在 `kvcompress/adapter.py` 加 `_patch_backend_guard`：
RSWA 启用时若 backend 不支持掩码则**启动即报错**，`AI_COMPRESS_FORCE_BACKEND=1` 可自动换
`TritonAttentionBackend`。**旧有的"无损 24/24"结论在 FA2 下不成立，需在 TRITON_ATTN 下重测。**

## 附：三种打分模式对比 + 指标局限（2026-09，冻结版）

| 打分模式 | 依赖 query | 额外前向 | LongBench F1 (ratio0.5, n=300) | div_rate |
|---|---|---|---|---|
| `window`（问题末尾窗口） | ✅ | ❌ | 0.2299 (−0.009) | **0.193** |
| `context`（全 prompt 采样 max） | ❌ | ❌ | **0.2361 (−0.003)** | 0.400 |
| `expected`（ExpectedAttention 式） | ❌ | ❌ | 0.2217 (−0.017) | 0.480 |

（无驱逐基线 F1 = 0.2391；SE≈0.023，故三者差值都在噪声内。）

**关键结论：`div_rate` 与 F1 的跨方法排序不一致。**
`div_rate` 说 window 最好、context 最差；F1 说 context 最好、expected 最差。
原因：`div_rate` 衡量的是"分布偏移"，而一个小模型在高熵/退化位置上的贪心 token
对数值扰动极敏感——**分布偏移 ≠ 任务损失**。所以：

- `div_rate` 只适合**同一方法族内的压缩比扫描**（我们实测它随压缩单调）；
- **跨方法比较必须用 F1**（或其它任务指标）。

另外：**`context` 模式在 F1 上近无损（−0.003）且是 query-agnostic（可复用）**——这比
`div_rate` 给出的印象好得多，说明"廉价 query-agnostic"这条路的实际可用性被低估了。
`expected` 的 F1 最差，说明按 (μ,Σ) 的解析期望注意力在这个小模型上并不比朴素采样更准。


## 附：query-agnostic（只用原内容）的压缩率-精度曲线

设置：`PE_SCORE_MODE=context`（不看问题，只用 context 自身采样 256 个 query + max），
LongBench 5 任务 × 40 条（n=200），max_new=32，无驱逐基线 F1 = **0.246**。

| 请求保留率 | 实测保留 | F1 | Δ | 释放块比例 |
|---|---|---|---|---|
| 0.8 | 78–79% | **0.2539** | **+0.008** | 21% |
| 0.6 | 59% | 0.2483 | +0.002 | 41% |
| 0.4 | 39–40% | 0.2349 | −0.011 | 60% |
| 0.3 | 29–30% | 0.2321 | −0.014 | 70% |
| **0.2** | **19–20%** | **0.2213** | **−0.025** | **80%** |

**结论**
1. **比率控制准确**：请求 0.2–0.8 → 实测保留率基本吻合。
2. **保留 ≥40% 时损失在噪声内**（n=200 时 F1 的 SE≈0.028，所有 |Δ|<1 SE）；
   保留 40–60% 基本持平，**60% 以上甚至略升**。
3. **压到 20% 保留**（prompt KV 显存 5×）→ F1 −0.025（≈1 SE，趋势上开始可见衰退）。
4. 复现：
   ```bash
   PE_MODE=chunkkv PE_BACKEND=TRITON_ATTN PE_RATIO=0.4 \
     PE_SCORE_MODE=context PE_CTX_QUERIES=256 PE_LOG=1 PE_N=40 \
     python -m bench.p3_longbench_f1
   ```

> ⚠️ 统计强度：若要判定"20% 保留的 −0.025 是否显著"，需把 n 提到 200/任务
> （SE 降到 ~0.012）。当前 n=200 总样本只能给趋势。

## 附：Needle-in-a-haystack（retrieval 场景）—— 便宜方法就够了

设置：`bench/p3_needle_bench.py`，针（唯一 code，如 `Z12345`）放在深度 {0.1,0.3,0.5,0.7,0.9}，
每深度 8 条（共 40），context 2048 token，问 "What is the secret vault code?"，
判据是输出**逐字包含**该 code。**无 sink**（否则 sink 就超过小预算）。

| 保留率 | `context`（query-agnostic） | `window+max`（query-aware） |
|---|---|---|
| 1.0（无驱逐） | 40/40 | 40/40 |
| 0.5 / 0.3 / 0.2 / 0.1 | 40/40 | 40/40 |
| **0.05** | 36/40 | **40/40** |
| **0.03** | **36/40** | 16/40 |
| **0.02** | **36/40** | 16/40 |
| **0.01** | **28/40** | 0/40 |

**结论**
1. **天花板在 ~5% 保留**：≥0.05 时两种便宜方法都近乎满分——因为针的 rank 只有 6–16（<1% 长度），
   任何 ≥5% 预算都保得住。**说明在 NIAH 上"免费方法已经够了"。**
2. **崩溃点不同、互有胜负**：`window+max` 在 0.05 满分但 0.03 崩；`context` 能撑到 0.02 但 0.05 略差。
   没有谁全面占优。
3. **最实际的结论**：**第二个前向（+2–3×）换来的 rank 6 vs 16，在 ≥5% 保留率上完全体现不出来**
   （两者都满分）；只有把 prompt 压到 **≤1%**（100×）时 rank 才有意义——那已是不现实的预算。
   ⇒ **在这个 retrieval 基准上，KVzip 式复述的额外成本买不到东西。**

> ⚠️ 边界：每深度仅 8 条（per-depth 噪声大，只看总趋势）；针是高显著性句子（与标准 NIAH 一致）；
> 单模型（1.5B）、单填充文本。复现：
> ```bash
> PE_MODE=chunkkv PE_BACKEND=TRITON_ATTN PE_RATIO=0.05 PE_SINK=0 \
>   PE_SCORE_MODE=window PE_OBS=32 PE_WIN_AGG=max \
>   python -m bench.p3_needle_bench
> ```

## 附：配对精度评测（n=950 + bootstrap CI）—— 统计上站得住的版本

`bench/p3_pair_eval.py`：5 任务 × 200 条（**n=950**，逐样本 F1 存盘），
对 baseline 做**配对 bootstrap（10000 次重采样）**给 ΔF1 的 95% CI。
CI 宽度 ±0.006–0.012 → **能检出 ~0.01 的差异**，所以"不显著"是有信息量的结论。

| 配置 | F1 | ΔF1 | 95% CI | 显著？ |
|---|---|---|---|---|
| baseline（无驱逐） | 0.2389 | — | — | — |
| `window+max` @保留 50% | 0.2405 | +0.0016 | [−0.0043, +0.0071] | ❌ |
| **`window+max` @保留 20%（KV 5×）** | 0.2365 | −0.0025 | [−0.0116, +0.0066] | ❌ |
| `context`（query-agnostic，可复用）@保留 50% | 0.2330 | −0.0059 | [−0.0151, +0.0036] | ❌ |
| **`window+max` @保留 10%（KV 10×）** | 0.2273 | **−0.0116** | **[−0.0227, −0.0006]** | ✅ |
| 位置式（sink64+win1024） | 0.1434 | **−0.0955** | [−0.1153, −0.0757] | ✅ |

**结论**
1. **保留 ≥20% → 损失不显著**，CI 上界 **< 1.2 分**；即 **KV 压 5× 在统计上无损**。
2. **保留 10%（压 10×）→ 首次显著**，−0.0116（相对 ~5%）。
3. **query-agnostic（可复用）在 50% 保留也不显著**（−0.0059，CI 含 0）。
4. **位置式 −0.0955**，比我们压到 10% 还差 8 倍。

复现：
```bash
P(){ env $2 PE_TAG=$1 PE_BACKEND=TRITON_ATTN PE_LOG=0 PE_N=200 \
      python -m bench.p3_pair_eval; }
P base   "PE_MODE=passthrough"
P wmax20 "PE_MODE=chunkkv PE_RATIO=0.2 PE_SCORE_MODE=window PE_OBS=32 PE_WIN_AGG=max"
PE_COMPARE=base PE_TAG=wmax20 python -m bench.p3_pair_eval   # 只看最后一行的 CI
```
