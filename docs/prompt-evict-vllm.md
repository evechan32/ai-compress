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
