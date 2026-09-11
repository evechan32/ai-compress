# FP8 KV 量化对比报告（SGLang + triton）

> 日期：2026-09-11 ｜ 模型：Qwen2.5-1.5B-Instruct ｜ 框架：SGLang 0.5.19（triton 后端）
> 说明：vLLM 的 fp8 KV 走 FlashInfer（本机 sm_120 不可用），故量化实验在 SGLang 完成。

## 1. 容量（显存）——约 2×

固定 `mem_fraction_static=0.30`（缩小 GPU KV 池以暴露容量墙），输入 17778 token：

| kv_cache_dtype | 结果 |
|---|---|
| bf16 | ❌ `Input length (17778 tokens) exceeds the maximum allowed length (11052 tokens)` |
| **fp8_e4m3** | ✅ 通过（wall 2.1s） |

→ FP8 KV 把可服务 KV 容量从 11052 提升到 **>17778（≈2×）**，符合 8bit 减半预期。

## 2. 质量（LongBench F1，n=30/子集，temp=0）

| 子集 | bf16 | fp8_e4m3 | Δ |
|---|---|---|---|
| qasper | 0.3059 | 0.2253 | -0.081 |
| 2wikimqa | 0.1521 | 0.1148 | -0.037 |
| multifieldqa_en | 0.4176 | 0.3347 | -0.083 |
| hotpotqa | 0.1751 | 0.1613 | -0.014 |
| triviaqa | 0.1969 | 0.2089 | +0.012 |

- 平均绝对下降 ≈ **-0.04**（相对 ~-12%），1.5B 上**可测**（非纯噪声）。
- 补充（n=20）：**fp8_e5m2 崩坏**（qasper 0.0104 / multifieldqa 0.0784），尾数位不足，不可用。
- 定性：fp8_e4m3 属"近无损但有代价"；更大模型通常损失更小。

## 3. 时延（prompt≈4000 token，生成 256，decode）

| 配置 | decode tok/s |
|---|---|
| bf16 | 156.5 |
| fp8_e4m3 | 147.6 |

- 1.5B 短中上下文下并非显存受限，反量化开销使 fp8 略慢（~6%）；FP8 的收益在**容量/显存受限**场景，而非此处。

## 4. 结论

| 维度 | fp8_e4m3 |
|---|---|
| KV 显存 | **~2× 容量**（实测） |
| 质量 | 小幅下降（F1 平均 -0.04，1.5B 可测） |
| 时延 | 非容量受限下略慢（~6%） |
| e5m2 | 不可用（质量崩坏） |
| 无损性 | **非逐字无损**；度量级"近无损"需按任务容忍度评估 |

**实践建议**：显存/容量受限且能接受小幅质量波动的场景可用 fp8_e4m3；否则用 bf16（或按 §前文 RSWA 做 KV 有界）。
**与 RSWA 插件对比**：RSWA 保留全 prompt → 常规任务逐字无损，但收益仅在生成段；FP8 压全部 KV → 容量 ~2×，但引入质量波动。两者可叠加（fp8 + RSWA）。

## 5. 复现

```bash
PY=/root/sglang-venv/bin/python
# 质量
$PY bench/sgl_longbench.py --data /hy-tmp/longbench/data \
  --files qasper.jsonl 2wikimqa.jsonl multifieldqa_en.jsonl hotpotqa.jsonl triviaqa.jsonl \
  --n 30 --tag final-fp8e4 --attention-backend triton --kv-cache-dtype fp8_e4m3
# 容量：mem_fraction_static=0.30 + 20k 输入（见 /tmp/sgl_cap.py）
```
