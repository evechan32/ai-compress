# kvcompress — vLLM 连续区间 KV 驱逐插件（零 fork）

面向长上下文 / 长多轮对话的 KV cache 显存优化。通过运行时注入把 **vLLM 原生 RSWA（Reference Sliding Window Attention）语义**应用到标准 MHA/GQA 模型（Qwen2.5 / Qwen3），**不修改 vLLM 源码**。

## 它做什么

RSWA 语义 = **prompt(prefill) KV 全部保留** + **生成段只保留最近 `window` 个 token**，更早的生成 KV 被逐 decode 步驱逐（gap 释放 + null 块位置对齐）。效果：每请求 KV 有界于 `O(prompt + window)`，不再随生成长度线性增长。

```
序列:  [==== prompt (始终保留) ====][.. 被驱逐 ..][= 最近 window (保留) =]
                                      ↑ 超窗后逐步释放
```

## 安装

```bash
pip install -e .        # 注册 vllm.general_plugins 入口点，vLLM 各进程自动加载
```

## 使用

```bash
# 默认关闭（零副作用，行为与原生 vLLM 一致）
AI_COMPRESS_ENABLE=1 AI_COMPRESS_RSWA_WINDOW=256 vllm serve <model>
```

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `AI_COMPRESS_ENABLE` | `0` | 总开关 |
| `AI_COMPRESS_POLICY` | `rswa` | `rswa`（v1 支持）；`sink_window` 为 v1.x，当前显式报错 |
| `AI_COMPRESS_RSWA_WINDOW` | `1024` | 生成段保留窗口（自动向上对齐到 block_size 16 的倍数） |
| `AI_COMPRESS_SINK_LEN` | `64` | sink 长度（sink_window 用，v1 未实现） |
| `AI_COMPRESS_TARGET_ARCHS` | `Qwen2ForCausalLM` | 支持：`Qwen2ForCausalLM`、`Qwen3ForCausalLM` |

RTX 50 系（sm_120）注意：flashinfer 采样在此环境不可用，插件入口已统一设置 `VLLM_USE_FLASHINFER_SAMPLER=0`。

## 支持的模型架构

- `Qwen2ForCausalLM`（如 Qwen2.5-1.5B/7B）
- `Qwen3ForCausalLM`（如 Qwen3-8B）

机制：将目标模型模块命名空间中的 `Attention` 替换为 `RSWAAttention` 工厂，并覆盖 `ModelConfig.rswa_window`。未知架构 / 未实现的 policy 会在启动期显式报错（拒绝"以为在压缩、实际跑全注意力"的静默错误）。

## 验收结果摘要（详见 docs/experiments-log.md）

| 项 | 结果 |
|---|---|
| 旁路一致性（关闭态） | 与原生逐字一致（服务器测试 2/2） |
| KV 有界性 | 基线块数无界增长 86→204 vs 插件平台化 **103**（Qwen2.5）；Qwen3 57→75 平台化 |
| 无损性（逐字一致） | Qwen2.5 **24/24**、Qwen3 **18/18**、prefix caching 开/关 **18/18** |
| 吞吐（容量受限并发） | 347.8 → **446.5 tok/s（+28%）**；batch=1 持平 |
| Prefix caching 兼容 | 语义/有界/复用收益四项均通过 |

## 能力边界（已确认）

- 零 fork 下只能表达**连续区间驱逐**（RSWA/SW 语义）；**SnapKV/H2O 散点驱逐不可行**（需内核级散点掩码）。
- **逐层窗口预算不可行**：vLLM 的 RSWA 掩码是模型级全局窗口，逐层不同窗口会导致掩码与驱逐不一致。
- **Sink+Window 激进模式不可行**：vLLM 的静态 sink 机制是 OpenPanGu 专用（learned sink + 专用后端）。
- 单请求超长生成"自引用被驱逐内容"的**损失量级未量化**（探针 v1–v5 因小模型不执行长格式指令而失败）。

## 目录

```
kvcompress/      插件包（入口 / config / policy / adapter）
bench/           评测（数据生成 / 评测入口 / 报告 / 一致性 / 探针）
tests/           单测（本地）+ 服务器集成测试
docs/            调研报告、设计文档、实现计划、spike 结论、实验总账、环境档案
```

- `docs/survey-kvcache-papers.md` — KV 压缩论文调研
- `docs/survey-vllm-integration.md` — vLLM 0.28 接入面源码级调研
- `docs/superpowers/specs/2026-09-08-kv-eviction-plugin-design.md` — 设计文档
- `docs/superpowers/plans/2026-09-08-kv-eviction-plugin.md` — 实现计划
- `docs/spike-r1-findings.md` — R1 可行性 spike 结论
- `docs/bench-report-2026-09-08.md` — 基准评测报告
- `docs/experiments-log.md` — **全部开发/测试数据总账（§1–§22）**
- `docs/FINAL-REPORT.md` — **项目总报告（全部实验结论与数据汇总）**
- `docs/fp8-kv-report.md` — FP8 KV 量化完整对比
- `docs/survey-query-sparse.md` — Query 块稀疏（Quest/InfLLM/MagicPIG/PQCache）调研
- `docs/attention-heatmap.md` — **注意力分布热力图（1024 token）**
- `docs/environment.md` — 服务器环境与重建档案
