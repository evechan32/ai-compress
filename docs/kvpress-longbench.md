# KVpress LongBench 多方法对照（HF 路径）

> 日期：2026-09-12 ｜ 脚本：`bench/kvpress_eval.py` ｜ 原始数据：`docs/kvpress/longbench-r0.5.json`、`docs/kvpress/longbench-r0.8.json`
> 环境：**NVIDIA kvpress 0.5.4 + transformers 5.2.0**（隔离在 `/root/kvpress-libs`，`PYTHONPATH` 遮蔽；系统 transformers 5.16.1 未改动）
> 设置：Qwen2.5-1.5B-Instruct；LongBench 5 任务 × n=20；HF greedy，`max_new=32`，context 截断 20000 字符。

## 0. 重要更正：KVzip 在本路径下**未生效**（retraction）

初版曾把"KVzip F1 与完整注意力逐任务完全相同"解读为"50%/80% 驱逐近无损"，**该结论已撤回**。复核证据：

- 缓存长度探针（同 prompt，`generate(max_new=4)` 后读第一层 K 的 seq 长度）：

  | 配置 | KV seq 长度 |
  |---|---|
  | 完整注意力 | 1203 |
  | snapkv ratio=0.5 | **603**（≈减半，压缩真实生效） |
  | tova ratio=0.5 | **603** |
  | chunkkv ratio=0.5 | **603** |
  | kvzip ratio=0.5 / 0.8 | 异常（读数 1，缓存结构与其它 press 不同） |

- KVzip 在 r0.5 与 r0.8 两个比例下、5 个任务的 F1 **全部逐位等于 baseline**（10 个数完全相同）。若真的驱逐了 50–80%，greedy 输出几乎不可能完全不变。

- **官方 pipeline 复核（决定性证据）**：用 `KVPressTextGenerationPipeline` + 自备 `DynamicCache`（context 15025 token），测得：

  | 配置 | 压缩后 cache_len | 耗时 |
  |---|---|---|
  | none | 15025 | 1.6s |
  | snapkv ratio=0.5 | **7512**（真实减半） | 1.3s |
  | kvzip ratio=0.5 | **15025（未压缩）** | **5.8s** |
  | kvzip ratio=0.8 | **15025（未压缩）** | 5.7s |

  → KVzip **确实执行了重建打分**（耗时 ≈3.6×，与其 2–3× 警告一致），但**最终没有真正缩小 KV cache**。手动 `with` 与官方 pipeline 两种路径结果一致，故这是 **kvpress 0.5.4 + transformers 5.2.0 的版本兼容问题，而非调用姿势错误**。

因此 **KVzip 一行不计入下方结论**。可行替代：① 在 transformers 4.x 环境跑 kvpress；② 用官方仓库 `snu-mllab/KVzip`；③ 用 `FastKVzipPress`（需下载 gate，HF 不可达时走镜像）。

## 1. 结果（F1，越大越好；baseline none 在 r0.5 测得 = 0.2882）

| method | r0.5 MEAN | Δ0.5 | r0.8 MEAN | Δ0.8 |
|---|---|---|---|---|
| chunkkv | 0.2886 | **+0.0004** | 0.2745 | −0.0137 |
| tova | 0.2872 | −0.0010 | 0.2533 | −0.0349 |
| snapkv | 0.2713 | −0.0169 | 0.2761 | −0.0121 |
| keydiff | 0.2705 | −0.0177 | 0.2082 | −0.0800 |
| expected（预期注意力） | 0.2630 | −0.0252 | 0.2438 | −0.0444 |
| pyramidkv | 0.2557 | −0.0325 | 0.2743 | −0.0139 |
| streamingllm | 0.2321 | −0.0561 | 0.2373 | −0.0509 |
| ~~kvzip~~ | ~~0.2882~~ | 未生效 | ~~0.2882~~ | 未生效 |

逐任务（r0.5）：`chunkkv` 与 baseline 几乎重合（qasper 0.3379/0.3451、mfqa 0.4086/0.3915、hotpotqa 完全持平）；`streamingllm` 在 mfqa 掉到 0.2554（vs 0.3915）。

## 2. 结论

1. **50% 驱逐下，`chunkkv` 与 `tova` 近无损**（Δ ≤ 0.001）；`snapkv`/`keydiff`/`expected`/`pyramidkv` 掉 0.017–0.033；**位置式 `streamingllm` 最差（−0.056）**。
2. **80% 驱逐下全部退化但幅度不大**：最好的是 `snapkv`（−0.012）、`chunkkv`/`pyramidkv`（−0.014）；`tova`/`expected` 掉 0.035–0.044；`keydiff` 最差（−0.080）。→ **块级/位置混合（chunkkv、pyramidkv）与当前-query 投票（tova）在激进预算下更稳**。
3. **与自研实现对比（关键）**：我们此前的散点/Quest/注意力分数在 ~50–60% 预算下掉得很厉害（FINAL-REPORT：qasper 0.3448→0.2166/0.2229/0.1159）。官方 press 中 chunkkv/tova 却能近无损 → **差距不在机制而在选择信号**：语义块（ChunkKV）、当前 query 投票（TOVA）明显优于我们的位置式/注意力式打分。
4. **实证了"两遍 prefill"代价**：KVzip 运行日志明确警告 2–3× prefill 开销——正是我们分析的"多一遍 prefill"，单请求长 prompt 场景 TTFT 翻倍。

## 3. 口径与局限（诚实披露）

- 本次 HF baseline `qasper=0.3451` 与历史 SGLang `0.3448` 一致，但 `2wikimqa`（0.1968 vs 历史 0.1036）、`multifieldqa_en`（0.3915 vs 0.4326）不同 → **跨运行/跨引擎比较需谨慎**；本次"press vs none"同 run 可比。
- n=20、单模型；context 截断 20000 字符；F1 部分任务噪声大。
- 各 press 对 `compression_ratio` 的解释可能不同，横向数字仅作量级参考。
- KVzip 未生效（见 §0），不计入结论。

## 4. 复现

```bash
# 隔离环境（系统 transformers 5.16.1 不动）
PYTHONPATH=/root/kvpress-libs /usr/local/bin/python3 bench/kvpress_eval.py \
  --data /hy-tmp/longbench/data \
  --files qasper.jsonl 2wikimqa.jsonl multifieldqa_en.jsonl hotpotqa.jsonl triviaqa.jsonl \
  --n 20 --ratio 0.5 --tag r0.5 --out /root/kvpress-out \
  --methods none snapkv chunkkv expected keydiff tova pyramidkv streamingllm kvzip
```

## 5. 隔离安装说明

kvpress 0.5.4 依赖 `transformers<5.3`，直接装会降级系统 5.16.1 → 采用 `pip install --no-deps --target=/root/kvpress-libs transformers==5.2.0 tokenizers==0.22.2 kvpress accelerate datasets pyarrow pandas dill xxhash multiprocess fire termcolor cachetools`，运行时 `PYTHONPATH=/root/kvpress-libs` 遮蔽，torch 复用系统。已验证系统 `transformers` 仍为 5.16.1。
