# 设计文档：ai-compress —— vLLM 连续区间 KV 驱逐插件

> 日期：2026-09-08 | 状态：草案待审阅 | 关联：`docs/survey-kvcache-papers.md`、`docs/survey-vllm-integration.md`

## 1. 背景与目标

在 vLLM 0.28.0 上以**独立包 + 运行时注入（零 fork）**形态交付一个可插拔的 KV cache 压缩插件，面向**长上下文 / 长多轮对话**场景压缩 KV 显存。验收标准：插件在**不改动 vLLM 源码**的前提下被加载，能真实降低长对话/长文推理的峰值 KV 显存，吞吐/时延指标可测量，且输出质量可评估（与全注意力基线对比）。

约束回顾（spike 实测）：
- GPU 服务器：2× RTX 5070 (12GB, sm_120)，vLLM 0.28.0 + torch 2.13(cu13) 已跑通 Qwen2.5-1.5B；需 `VLLM_USE_FLASHINFER_SAMPLER=0`。
- 磁盘仅 ~7.9GB 可用 → 评测模型以 1.5B 为主，数据集优先自生成。

## 2. 技术路线（源自 spike 结论）

vLLM 0.28 已内置连续区间驱逐全套机制（`KVCacheSpecRegistry` + 各类 Manager + null-block 位置对齐 + RSWA/Sink 注意力窗口）。零 fork 只能表达**连续区间驱逐**；散点驱逐（SnapKV/H2O）需内核级散点掩码，排除。

本插件 = **策略化的连续区间驱逐**：
1. 通过 `vllm.general_plugins` 入口点在全部进程注入。
2. 注册自定义 `KVCacheSpec` + Manager（policy 化的窗口/sink 配置），复用 vLLM 原生驱逐/对齐逻辑。
3. 模型适配层使标准 MHA/GQA 模型（首目标 Qwen2.5 系）在加载时走受限窗口注意力路径，**复用 vLLM 已有 RSWA/sink 注意力 metadata 与 mask 支持，不写 CUDA kernel**。

## 3. 驱逐语义（v1 → v1.x）

**v1 = RSWA 式（生成段有界）**：prompt(prefill) KV 全部可见；生成 token 仅保留末尾 `window` 个，逐 decode 步驱逐中间 gap。语义与 vLLM 原生 `RSWASpec` 一致 → 多轮对话生成 KV 有界 O(prefix + window)。质量损失最小，先打通端到端。

**v1.x = Sink+Window 式（激进，可选开启）**：整个上下文（含 prompt 中段与旧轮次）只保留 sink + 末尾窗口，中段驱逐；对"仅需最近 + 首部信息"的长任务收益最大。

两者共用同一架构：驱逐 = `EvictionPolicy` 决定"保留哪些连续区间"；Manager 执行释放与对齐；模型注意力声明同区间可见。

## 4. 架构与模块

```
ai-compress/
├── pyproject.toml              # 包元数据 + vllm.general_plugins entry point
├── kvcompress/
│   ├── __init__.py             # 入口：os.environ 设置、注册、模型适配装载
│   ├── config.py               # AI_COMPRESS_* 配置解析（window/sink/开关/目标架构）
│   ├── policy.py               # EvictionPolicy 抽象 + RSWAPolicy + SinkWindowPolicy
│   ├── specs.py                # 注册自定义 spec（继承 FullAttentionSpec/RSWASpec）
│   ├── manager.py              # 自定义 manager（继承原生 RSWAManager/FullAttentionManager，策略化）
│   └── adapters/
│       └── qwen2.py            # Qwen2.5 系模型适配：让该架构产出受限窗口 spec
├── bench/
│   ├── gen_data.py             # 自生成评测数据（needle-in-haystack / 多轮长对话）
│   ├── run_baseline.py         # 原生 vLLM 全注意力基线
│   ├── run_plugin.py           # 加载插件的 vLLM 评测
│   └── report.py               # 汇总质量/显存/吞吐对比
└── docs/                       # 调研报告 + 本设计文档
```

### 4.1 数据流

1. 用户以 `python -m bench.run_plugin`（或任何 vLLM 入口）启动；包被 entry point 自动加载。
2. `__init__` 在进程早期 `os.environ.setdefault` 关键项并注册 spec/manager、装载模型适配。
3. 模型加载时适配层使目标架构产出自定义 spec → 引擎用我们的 manager 管理该 KV group。
4. 推理中 manager 按 policy 逐步驱逐窗口外连续区间（null-block 对齐）；注意力路径只读可见区间。

### 4.2 配置接口（环境变量，v1）

| 变量 | 默认 | 含义 |
|---|---|---|
| `AI_COMPRESS_ENABLE` | `0` | 总开关（0 = 完全旁路，行为与原生一致） |
| `AI_COMPRESS_POLICY` | `rswa` | `rswa` 或 `sink_window` |
| `AI_COMPRESS_RSWA_WINDOW` | `1024` | 生成段保留窗口 token 数 |
| `AI_COMPRESS_SINK_LEN` | `64` | sink token 数（sink_window 策略） |
| `AI_COMPRESS_TARGET_ARCHS` | `Qwen2ForCausalLM,Qwen2MoeForCausalLM` | 启用适配的架构列表 |

窗口长度须为 block_size 整数倍（vLLM 约束），配置解析时向上对齐并告警。

## 5. 与 vLLM 的集成点（对照调研报告 §5）

- 注册：`KVCacheSpecRegistry.register(SpecCls, ManagerCls, uniform_type_base_spec=FullAttentionSpec)`（模块 import 时执行）。
- Manager 复用：v1 直接子类化 `RSWAManager`/`FullAttentionManager`，策略参数化 `get_num_skipped_tokens`；必要时按 RSWA gap 路径处理 `num_prompt_tokens`。
- 注意力：RSWA 路径要求模型配置与后端支持 `rswa_mask_mod`（FA4/FlashAttention）。模型适配层负责把目标架构的注意力产出/元数据切换到受限窗口形态。**若 v0.28.0 对非原生 RSWA 模型的后端可用性在 spike 实验中被证伪**，v1 回退方案为复用 Sink/滑窗路径或禁用 chunked prefill 的简化路径（见风险 R2）。
- 注入：`pyproject.toml` `[project.entry-points."vllm.general_plugins"]`。

## 6. 评测方案

模型：Qwen2.5-1.5B-Instruct（主），条件允许补 Qwen2.5-3B。对比：原生 vLLM（基线）vs 插件启用，同模型同 prompt。

| 任务 | 数据 | 指标 |
|---|---|---|
| 多轮长对话（合成，每轮续写 → 累积 ~4k-32k token） | 自生成 | 逐轮困惑度/延续一致性 + 峰值 KV 显存 + KV 块数 |
| Needle-in-haystack | 自生成（随机位置插入事实） | 召回准确率（按深度分桶） |
| LongBench 子集（可选，ModelScope） | 2-3 类 | 官方 metric |

系统指标：TTFT / TPOT / 吞吐（tokens/s）/ 峰值 GPU 显存 / 每请求 KV 块数（引擎日志 `GPU KV cache size` 与 nvidia-smi 交叉）。控制：同 batch=1 与 batch=4 各一组；temperature=0。

## 7. 错误处理与边界

- 关闭态必须与原生行为**逐位一致**（旁路零副作用）。
- 配置非法（window < block_size、非对齐、未知 policy/arch）→ 启动期显式报错并提示合法值，不静默降级。
- 适配层只对声明支持的架构生效；遇到未知架构且开关开启时**拒绝服务并给出原因**，而不是静默跑全注意力（避免"以为压缩了其实没有"）。
- 与前缀缓存/块复用冲突时：v1 关闭与压缩互斥的前缀缓存特性或验证其正确性，优先保证语义正确。

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| R1: 标准模型走 RSWA 窗口注意力路径在 v0.28 上的后端可用性未知（FA4 rswa_mask_mod 或需特定 attention 实现） | 实现前先做最小 spike 实验：手工把 Qwen2.5 config 注入 rswa_window 后跑通并对比输出；失败则走 Sink/滑窗复用或简化路径 |
| R2: Qwen2.5 模型适配工作量 > 预期（每架构数百行） | v1 只支持 1 个架构族（Qwen2），代码结构按注册式设计便于扩展 |
| R3: 12GB 显存限制长文评测长度 | 1.5B 模型 + 4bit 权重可选 + 上限 ~32k；评测设计与之匹配 |
| R4: 磁盘 7.9GB | 优先自生成数据；模型复用已下载的 1.5B；LongBench 子集按需最小化 |
| R5: 质量损失超预期（窗口语义近似） | 评测报告如实呈现"质量-显存"权衡曲线；策略参数可调 |

## 9. 里程碑与验收

- **M1** 包骨架 + 旁路验证：`AI_COMPRESS_ENABLE=0` 时输出与原生一致。
- **M2** RSWA 式驱逐打通（Qwen2.5-1.5B）：启用后 KV 块数随对话增长受窗约束；输出合理。
- **M3** 评测闭环：基线 vs 插件全套指标报告（质量/显存/吞吐），含 R1 风险的决策记录。
- **M4**（可选 v1.x）Sink+Window 策略实现 + 激进模式评测。
- 验收 = M2/M3 交付可复现脚本与报告，明确标注各语义的适用场景。

## 10. 非目标（明确排除）

- 不做任意 token 散点驱逐（H2O/SnapKV 精确语义）——需 fork/内核改造。
- 不做 KV 量化（FP8 已原生；sub-8bit 需内核）。
- 不做 MLA/结构压缩（架构级）。
- 不改动 vLLM 源码；所有能力通过注册/注入获得。
