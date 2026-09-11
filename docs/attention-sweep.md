# 注意力 / KV 分布：跨模型 × 跨长度扫描

> 日期：2026-09-12 ｜ 脚本：`bench/attn_sweep.py`
> 模型：Qwen2.5-1.5B-Instruct（GQA 12/2）、Llama-3.2-1B-Instruct（GQA 32/8）、SmolLM2-1.7B-Instruct（MHA 32/32）
> 长度：1024 / 2048 / 4096 / 8192 ｜ 输入：needle-in-the-middle（needle 放在 50%），取**最后一个 query** 对各 KV 位置的注意力。
> 方法：HF transformers 5.16.1 + **query-chunked eager 注意力**（峰值显存 `O(H·chunk·L)` 而非 `O(H·L²)`，长序列不 OOM；不生成，只做一次前向）。原始数据：`docs/figs/attn-sweep/`（图 + `stats.json`/`summary.json`）。

## 0. 主表（最后一个 query，层/头平均）

| 模型 | L | sink(64) | win(256) | middle | top10% | needle 质量 | needle rank | H/Hkv | grp | hdim | KV (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1024 | 0.314 | 0.479 | 0.207 | 0.775 | 0.0351 | 13 | 12/2 | 6 | 128 | 28 |
| Qwen2.5-1.5B | 2048 | 0.333 | 0.399 | 0.268 | 0.785 | 0.0400 | 8 | 12/2 | 6 | 128 | 56 |
| Qwen2.5-1.5B | 4096 | 0.330 | 0.368 | 0.302 | 0.832 | 0.0209 | 9 | 12/2 | 6 | 128 | 112 |
| Qwen2.5-1.5B | 8192 | 0.320 | 0.414 | 0.266 | 0.837 | 0.0100 | 24 | 12/2 | 6 | 128 | 224 |
| Llama-3.2-1B | 1024 | 0.482 | 0.345 | 0.173 | 0.808 | 0.0437 | 5 | 32/8 | 4 | 64 | 32 |
| Llama-3.2-1B | 2048 | 0.515 | 0.276 | 0.210 | 0.821 | 0.0282 | 7 | 32/8 | 4 | 64 | 64 |
| Llama-3.2-1B | 4096 | 0.500 | 0.271 | 0.228 | 0.849 | 0.0146 | 7 | 32/8 | 4 | 64 | 128 |
| Llama-3.2-1B | 8192 | 0.457 | 0.280 | 0.263 | 0.822 | 0.0148 | 10 | 32/8 | 4 | 64 | 256 |
| SmolLM2-1.7B | 1024 | 0.576 | 0.286 | 0.138 | 0.830 | 0.0270 | 11 | 32/32 | 1 | 64 | 192 |
| SmolLM2-1.7B | 2048 | 0.588 | 0.245 | 0.168 | 0.850 | 0.0227 | 10 | 32/32 | 1 | 64 | 384 |
| SmolLM2-1.7B | 4096 | 0.551 | 0.295 | 0.155 | 0.897 | 0.0119 | 16 | 32/32 | 1 | 64 | 768 |
| SmolLM2-1.7B | 8192 | 0.568 | 0.257 | 0.175 | 0.894 | 0.0052 | 19 | 32/32 | 1 | 64 | 1536 |

每 token KV 字节（@8192）：**Qwen 28 KB ＜ Llama 32 KB ＜＜ SmolLM2 192 KB（6.9×）**。

## 1. 关键发现

1. **sink 是"模型属性"，不是"长度属性"。** 同一模型内 sink 占比几乎不随 L 变：Qwen ≈0.32、Llama ≈0.46–0.52、SmolLM2 ≈0.55–0.59。→ 统一 sink 预算（如固定 64）不合理，应按模型校准。
2. **注意力高度集中，且越长越集中。** top10% 位置覆盖 **0.78（1k）→ 0.90（8k）**。→ 长上下文下 KV 冗余度更高，压缩空间变大。
3. **固定"最近窗口"占比随 L 被稀释。** win(256) 从 0.48→0.41（Qwen）、0.35→0.28（Llama）、0.29→0.26（SmolLM2）。→ 越长越不能只靠位置启发式（StreamingLLM/RSWA），必须补内容驱动的中段选择。
4. **中段占比随 L 上升。** middle 从 0.21→0.30（Qwen）、0.17→0.26（Llama）、0.14→0.18（SmolLM2）。→ "中段筛选"问题在长上下文下**更重要也更难**。
5. **KV 大小差异几乎完全由 GQA / 层数决定：** SmolLM2（MHA 32/32、24 层）每 token KV 是 Qwen（GQA 12/2、28 层）的 **6.9×**、Llama（GQA 32/8、16 层）的 6×。→ **架构（GQA）的收益远大于任何推理期插件压缩**；反过来，头级/层间压缩对 MHA 类模型收益最大。
6. **needle 并不在高注意力位置：rank 5–24，质量占比仅 0.5–4%。** 且随 L 变差（Qwen 13→24、SmolLM2 11→19）。→ 再次证实 **"注意力质量 ≠ 信息价值"**：纯按注意力质量做驱逐会漏掉唯一答案 token。

## 2. 对 KV 压缩方向的含义

- **位置先验（sink+window）天花板约 0.7–0.85，且随 L 稀释** → 我们的 RSWA 在长上下文下收益稳定但有限，必须叠加内容/语义选择才有进一步提升。
- **sink 跨模型差异大** → sink 长度应作为 per-model 超参（我们插件默认 64 对 SmolLM2 偏小）。
- **GQA 是最大的"免费压缩"** → 若目标是显存，换 GQA 模型比推理期压缩更彻底；若目标是研究推理期压缩，**MHA 大 KV 模型（如 SmolLM2）才是最有价值的试验台**。
- **needle 低 rank** → 选择信号要基于"模型如何使用"（检索头/重建/期望注意力），而非注意力质量本身；这与 `survey-latest-2026.md` 里 CompressKV(检索头+层预算)、KVzip(重建) 的动机一致。

## 3. 复现

```bash
# 服务器；输出写到 /root 避免 /hy-tmp 压力
/usr/local/bin/python3 bench/attn_sweep.py \
  --models /models/qwen2.5-1.5b-instruct \
           /hy-tmp/models/Llama-3.2-1B-Instruct \
           /hy-tmp/models/SmolLM2-1.7B-Instruct \
  --lengths 1024 2048 4096 8192 --out /root/attn_sweep --max-full 1024
```

说明：`attention_matrix` 用 query-chunked 实现，`attn.npz` 只对 `L ≤ --max-full` 存完整 `[H,Lq,Lk]`；`usage.npy` 是每个 KV 位置收到的注意力质量（列和）。

## 4. 产出图（`docs/figs/attn-sweep/`）

- `aggregate_coverage.png` — sink/window/middle/top10%/needle 占比 vs 长度（3 模型）
- `kv_bytes_vs_length.png`、`kv_per_layer.png`、`kv_usage_distribution.png`
- `<model>/L<n>/attn_positions.png`（逐层 attention vs 位置）、`attn_cumulative.png`（累积覆盖）
- `<model>/L1024/attn_heads_L*.png`（头 × 位置热力图）

## 5. 局限

- 每 (模型, L) **仅 1 个样本**、needle 固定在 50%、单次前向取最后 query 行；非统计结论。
- 只取最后 query 行 + 全序列列和，**不等价于自回归生成时的逐步分布**。
- chunked 注意力与 HF 原生 eager 数学等价（同一 softmax），但未逐元素数值校验。
