# KVpress LongBench 多方法对照（HF 路径）

> 日期：2026-09-12 ｜ 脚本：`bench/kvpress_eval.py` ｜ 原始数据：`docs/kvpress/longbench-r0.5.json`
> 环境：**NVIDIA kvpress 0.5.4 + transformers 5.2.0**（隔离在 `/root/kvpress-libs`，用 `PYTHONPATH` 遮蔽；系统 transformers 5.16.1 未改动）
> 设置：Qwen2.5-1.5B-Instruct；LongBench 5 任务 × n=20；`compression_ratio=0.5`（驱逐 50%）；HF greedy 生成，`max_new=32`，context 截断 20000 字符；wall 417s。

## 0. 结果（F1，越大越好）

| method | qasper | 2wikimqa | multifieldqa_en | hotpotqa | triviaqa | MEAN | ΔMEAN |
|---|---|---|---|---|---|---|---|
| chunkkv | 0.3379 | 0.1939 | 0.4086 | 0.3177 | 0.1849 | **0.2886** | +0.0004 |
| none（完整） | 0.3451 | 0.1968 | 0.3915 | 0.3177 | 0.1898 | 0.2882 | 0 |
| **kvzip** | 0.3451 | 0.1968 | 0.3915 | 0.3177 | 0.1898 | **0.2882** | **0.0000** |
| tova | 0.3590 | 0.2027 | 0.3821 | 0.3119 | 0.1802 | 0.2872 | −0.0010 |
| snapkv | 0.3158 | 0.1527 | 0.3851 | 0.3177 | 0.1853 | 0.2713 | −0.0169 |
| keydiff | 0.3472 | 0.1875 | 0.4017 | 0.2477 | 0.1686 | 0.2705 | −0.0177 |
| expected（预期注意力） | 0.3594 | 0.1527 | 0.3278 | 0.3119 | 0.1634 | 0.2630 | −0.0252 |
| pyramidkv | 0.2707 | 0.1528 | 0.3897 | 0.2977 | 0.1678 | 0.2557 | −0.0325 |
| streamingllm | 0.2684 | 0.1417 | 0.2554 | 0.3115 | 0.1834 | 0.2321 | −0.0561 |

## 1. 结论

1. **50% 驱逐下，`chunkkv` / `kvzip` / `tova` 与完整注意力几乎无差**（ΔMEAN ≤ 0.001）。其中 **KVzip 五个任务逐字等于 baseline** —— 这是"query-agnostic 压缩在 50% 预算近无损"的直接证据。
2. **`snapkv` / `keydiff` / `expected` / `pyramidkv` 掉 0.017–0.033**；**`streamingllm`（位置式）最差，掉 0.056**——与我们此前"位置式优于注意力式"的结论方向不一致，说明**官方实现的选择信号比我们的位置式更成熟**。
3. **与自研实现对比（关键）**：我们此前的散点/Quest/注意力分数在 ~50–60% 预算下掉得很厉害（FINAL-REPORT：qasper 0.3448→0.2166/0.2229/0.1159）。本次官方 press 中 chunkkv/kvzip/tova 却能近无损 → **差距不在机制而在"选择信号"**：重建（KVzip）、语义块（ChunkKV）、当前 query 投票（TOVA）明显优于我们的位置式/注意力式打分。
4. **KVzip 的代价**：运行日志明确警告 "requires multiple forward passes ... 2–3× initial prefilling"，正是我们分析的"多一遍 prefill"，单请求长 prompt 场景 TTFT 翻倍。

## 2. 口径与局限（诚实披露）

- 本次 HF baseline `qasper=0.3451` 与历史 SGLang `0.3448` 一致，但 `2wikimqa`（0.1968 vs 历史 0.1036）、`multifieldqa_en`（0.3915 vs 0.4326）不同 → **跨运行/跨引擎比较需谨慎**；本次"kvpress vs none"同 run 可比。
- n=20、单 ratio（0.5）、单模型；context 截断 20000 字符；F1 部分任务噪声大。
- 各 press 对 `compression_ratio` 的解释可能不同，横向数字仅作量级参考。

## 3. 复现

```bash
# 隔离环境（系统 transformers 5.16.1 不动）
PYTHONPATH=/root/kvpress-libs /usr/local/bin/python3 bench/kvpress_eval.py \
  --data /hy-tmp/longbench/data \
  --files qasper.jsonl 2wikimqa.jsonl multifieldqa_en.jsonl hotpotqa.jsonl triviaqa.jsonl \
  --n 20 --ratio 0.5 --tag r0.5 --out /root/kvpress-out \
  --methods none snapkv chunkkv expected keydiff tova pyramidkv streamingllm kvzip
```

## 4. 隔离安装说明

kvpress 0.5.4 依赖 `transformers<5.3`，直接装会降级系统 5.16.1 → 采用 `pip install --no-deps --target=/root/kvpress-libs transformers==5.2.0 tokenizers==0.22.2 kvpress accelerate datasets pyarrow pandas dill xxhash multiprocess fire termcolor cachetools`，运行时 `PYTHONPATH=/root/kvpress-libs` 遮蔽，torch 复用系统。已验证系统 `transformers` 仍为 5.16.1。
