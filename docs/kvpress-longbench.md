# KVpress LongBench 多方法对照（HF 路径）

> 日期：2026-09-12 ｜ 脚本：`bench/kvpress_eval.py` ｜ 原始数据：`docs/kvpress/longbench-r0.5.json`、`docs/kvpress/longbench-r0.8.json`
> 环境：**NVIDIA kvpress 0.5.4 + transformers 5.2.0**（隔离在 `/root/kvpress-libs`，`PYTHONPATH` 遮蔽；系统 transformers 5.16.1 未改动）
> 设置：Qwen2.5-1.5B-Instruct；LongBench 5 任务 × n=20；HF greedy，`max_new=32`，context 截断 20000 字符。

## 0. KVzip 的真相：假 key 掩码式压缩（手动路径作废；pipeline 有效数字见 §0.1）

关于 KVzip 曾出现两个错误解读，均已推翻。最终结论（读源码 + 探针）：

- **机制：KVzip 在 kvpress 里是"假 key 掩码式"压缩，不是物理驱逐。** `KVzipPress.compress_post` 只写入 `module.masked_key_indices`；`attention_patch.attention_patch` 装饰器在 **decode 时**把被驱逐位置替换成满足 `exp(<q,k>)=0` 的假 key，作用于注意力计算（`kvpress/__init__.py:51` 在 import 时自动 `patch_attention_functions()`）。其文档字符串明确写着 **"does not reduce peak memory"**。
  → **因此用 cache 长度判断 KVzip 是否压缩是错的**：此前"未压缩 / 版本不兼容"的定性作废（§0 旧版据此得出的 15025→15025 属正常现象）。
- **真正的问题在调用协议**：`KVzipPress.__call__` 把打分与压缩放在 `with` 块**退出之后**执行。而我们的 `bench/kvpress_eval.py` 是在 `with` 内 `generate`，压缩发生在生成**之后** → 对本次生成完全无影响。所以 KVzip 的 F1 与 baseline 逐位相同是**协议错误造成的空操作**，既不能说明"不能压缩"，也不能说明"近无损"。
- **正确协议**（官方 `KVPressTextGenerationPipeline`）：`with` 内只 prefill **context** → 退出 `with` 时压缩（设置 `masked_key_indices`）→ **之后**再对 question 生成。
### 0.1 有效数字（pipeline 协议，`docs/kvpress/pipeline-r0.5.json`）

用 `bench/kvpress_pipeline_eval.py`（官方 pipeline：`with` 内 prefill context → 退出压缩 → 再生成 question），2 任务 × n=10、ratio=0.5：

| method | qasper | multifieldqa_en | MEAN | Δv​s none |
|---|---|---|---|---|
| none | 0.1834 | 0.2906 | **0.2370** | 0 |
| snapkv | 0.1282 | 0.2788 | 0.2035 | −0.0335 |
| **kvzip** | 0.1804 | 0.2683 | **0.2243** | **−0.0127** |

**决定性证据**：KVzip 的 F1 **不再等于 baseline**（0.2243 vs 0.2370）→ 掩码压缩**确实生效**。在 50% 预算下 KVzip 掉约 0.013、优于同协议下 snapkv 的 −0.034；即 **KVzip 能压缩，且在此预算下损失很小**。
（注：pipeline 协议用 chat template + 原始问题，baseline 与 §1 手动协议的 0.2882 不可直接比；本表内可比。）

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
| ~~kvzip~~ | ~~0.2882~~ | 手动协议作废 | ~~0.2882~~ | 手动协议作废；**pipeline 有效值见 §0.1** |

逐任务（r0.5）：`chunkkv` 与 baseline 几乎重合（qasper 0.3379/0.3451、mfqa 0.4086/0.3915、hotpotqa 完全持平）；`streamingllm` 在 mfqa 掉到 0.2554（vs 0.3915）。

## 2. 结论

1. **50% 驱逐下，`chunkkv` 与 `tova` 近无损**（Δ ≤ 0.001）；`snapkv`/`keydiff`/`expected`/`pyramidkv` 掉 0.017–0.033；**位置式 `streamingllm` 最差（−0.056）**。
2. **80% 驱逐下全部退化但幅度不大**：最好的是 `snapkv`（−0.012）、`chunkkv`/`pyramidkv`（−0.014）；`tova`/`expected` 掉 0.035–0.044；`keydiff` 最差（−0.080）。→ **块级/位置混合（chunkkv、pyramidkv）与当前-query 投票（tova）在激进预算下更稳**。
3. **与自研实现对比（关键）**：我们此前的散点/Quest/注意力分数在 ~50–60% 预算下掉得很厉害（FINAL-REPORT：qasper 0.3448→0.2166/0.2229/0.1159）。官方 press 中 chunkkv/tova 却能近无损 → **差距不在机制而在选择信号**：语义块（ChunkKV）、当前 query 投票（TOVA）明显优于我们的位置式/注意力式打分。
4. **实证了"额外 prefill"代价**：KVzip 运行日志明确警告 2–3× prefill 开销，且实测耗时 1.6s→5.8s（≈3.6×），确认其重建打分确实执行。

## 3. 口径与局限（诚实披露）

- 本次 HF baseline `qasper=0.3451` 与历史 SGLang `0.3448` 一致，但 `2wikimqa`（0.1968 vs 历史 0.1036）、`multifieldqa_en`（0.3915 vs 0.4326）不同 → **跨运行/跨引擎比较需谨慎**；本次"press vs none"同 run 可比。
- n=20、单模型；context 截断 20000 字符；F1 部分任务噪声大。
- 各 press 对 `compression_ratio` 的解释可能不同，横向数字仅作量级参考。
- KVzip 手动协议行作废（调用协议错误，见 §0）；**pipeline 协议有效值：0.2243（−0.0127），见 §0.1**。

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

## 6. 附：kvpress 全库广扫（ratio 0.5 / 0.8，5 任务 × n=20）

Baseline `none = 0.2882`。0.5 与 0.8 两次广扫合并（原始 JSON 见本机 `server-export/kvpress-out/`）：

| method | mean@0.5 | Δ0.5 | mean@0.8 | Δ0.8 |
|---|---|---|---|---|
| compactor | 0.2918 | **+0.0036** | 0.2474 | −0.0408 |
| chunkkv | 0.2886 | +0.0004 | 0.2745 | −0.0137 |
| tova | 0.2872 | −0.0010 | 0.2533 | −0.0349 |
| cur | 0.2857 | −0.0025 | 0.2456 | −0.0426 |
| leverage | 0.2833 | −0.0049 | 0.2498 | −0.0384 |
| block | 0.2793 | −0.0089 | 0.2199 | −0.0683 |
| ada（头级预算） | 0.2750 | −0.0132 | 0.2674 | −0.0208 |
| lagkv | 0.2744 | −0.0138 | 0.2372 | −0.0510 |
| merging | 0.2718 | −0.0164 | 0.2714 | −0.0168 |
| snapkv | 0.2713 | −0.0169 | 0.2761 | **−0.0121** |
| keydiff | 0.2705 | −0.0177 | 0.2082 | −0.0800 |
| think（通道） | 0.2671 | −0.0211 | 0.1165 | −0.1717 |
| expected | 0.2630 | −0.0252 | 0.2438 | −0.0444 |
| pyramidkv | 0.2557 | −0.0325 | 0.2743 | −0.0139 |
| cap | 0.2475 | −0.0407 | 0.2203 | −0.0679 |
| knorm | 0.2341 | −0.0541 | 0.1120 | −0.1762 |
| streamingllm | 0.2321 | −0.0561 | 0.2373 | −0.0509 |
| chunk | 0.1812 | −0.1070 | 0.1141 | −0.1741 |

结论：

1. **`chunkkv` 最稳**：50% 近无损（+0.0004）、80% 仍居第一梯队（−0.014）。
2. **`snapkv` / `pyramidkv` 在 80% 更稳**（−0.012 / −0.014）——"观察窗口 + 位置"组合抗激进压缩。
3. **`ada`（头级预算）在 80% 明显优于 50%**（−0.021 vs −0.013）→ 头级重分配在极端预算下更重要。
4. **`compactor` / `cur` / `leverage` 只适合温和预算**（50% ≈/优于 baseline，80% 掉 ~0.04）。
5. **通道/范数类在激进预算崩**：`think` −0.17、`knorm` −0.18、`chunk` −0.17。
6. 环境失败（非算法结论）：`observed`（需 eager）、`kvzap`/`finch`/`duo`/`critical`/`dms`（需额外资产）、`qfilter`（需 HF 下载）。

> `kvzip` 不在此表（手动协议无效，见 §0/§0.1）。
