# KV Cache 压缩论文调研报告

> 调研日期：2026-09-08 | 引用数来源：Semantic Scholar（实时 API 批量核实）| 项目：ai-compress

## 1. 方法族分类

KV cache 压缩研究可归为四大方法族：

1. **Token 驱逐 / 注意力稀疏（Eviction / Sparsity）**：按 token 对后续生成的重要性决定哪些 KV 不保留或逐步释放。无需重训，直接作用于现有模型。代表：H2O、StreamingLLM、SnapKV、PyramidKV、Quest、InfLLM。
2. **KV 量化（Quantization）**：将 K/V 缓存低位量化存储（2-8 bit），配合 per-channel/per-token 缩放或旋转去离群点。近无损、压缩比中等（2-4×）。代表：KIVI、KVQuant、GEAR、QoQ。
3. **结构 / 低秩压缩（Structural / Low-rank）**：修改注意力结构本身——低秩联合投影（MLA）、跨层共享 KV（CLA/YOCO）等。需要改变架构或训练/继续预训练，通常作为模型侧基线而非可插拔插件。
4. **混合 / Cache 复用与卸载**：PD 分离、prefix caching、KV offload（CPU/disk）属于内存分层而非压缩，通常与其他方法叠加。

## 2. 论文明细表（引用数为 S2 2026-09 数据）

### 2.1 Token 驱逐 / 注意力稀疏

| 标题 | 年份 | Venue | 引用 | arXiv | 方法要点 |
|------|------|-------|------|-------|---------|
| Efficient Streaming Language Models with Attention Sinks | 2023 | ICLR'24 | 2374 | 2309.17453 | 保留首 few sink token + 末尾窗口，中段丢弃；attention sink 机制 |
| H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs | 2023 | NeurIPS'23 | 910 | 2306.14048 | 依据累计 attention score 保留 Heavy-Hitter token，在线驱逐 |
| Scissorhands: Exploiting the Persistence of Importance Hypothesis | 2023 | NeurIPS'23 | 519 | 2305.17118 | "持久重要性"假设：历史高分 token 持续重要 |
| SnapKV: LLM Knows What You Are Looking for Before Generation | 2024 | NeurIPS'24 | 898 | 2404.14469 | prefill 后按最后层 attention 对 prompt 投票，压缩 prompt KV 再进 decode |
| Quest: Query-Aware Sparsity for Long-Context | 2024 | ICML'24 | 493 | 2406.10774 | 按 query 相关性检索重要 KV 块（KV 块级稀疏） |
| PyramidKV: Dynamic KV Cache Compression（按层分配） | 2024 | arXiv | 416 | 2406.02069 | 金字塔信息漏斗，浅层多留深层少留 |
| InfLLM: Training-Free Long-Context Extrapolation | 2024 | NeurIPS'24 | 198 | 2402.04617 | 单位块记忆 + 与当前 query 相关的块参与注意力 |
| TOVA (Transformers are Multi-State RNNs) | 2024 | EMNLP'24 | 124 | 2401.06104 | 输出感知驱逐：用任务 loss 判定驱逐对象 |
| KVStream / D2O / 2025-26 改进 | 2025 | — | 待复核 | ID 待复核 | 基于 attention 模式自适应预算、分层驱逐策略等（见 3.3） |

### 2.2 KV 量化

| 标题 | 年份 | Venue | 引用 | arXiv | 方法要点 |
|------|------|-------|------|-------|---------|
| KVQuant: Towards 10M Context with KV Cache Quantization | 2024 | NeurIPS'24 | 673 | 2401.18079 | per-channel + per-token 量化，旋转/PCA 去离群点，4bit 近无损 |
| KIVI: Tuning-Free Asymmetric 2bit Quantization for KV Cache | 2024 | ICML'24 | 638 | 2402.02750 | K 按 channel、V 按 token 非对称量化，2/4bit，无需微调 |
| GEAR: Near-Lossless KV Cache Compression | 2024 | arXiv | 187 | 2403.05527 | 量化 + 低秩 + 稀疏三组件纠偏 |
| QoQ | 2024 | arXiv | ID 待复核 | 4bit 量化精度对齐研究 |
| FP8 KV cache（vLLM/SGLang 原生） | 2024-25 | 框架内置 | — | — | e4m3/e5m2，工程基线 |
| QuaRot/SpinQuant（旋转法，主要面向 weight/act） | 2024 | — | — | — | 随机旋转消除离群点，KV 量化受益于同机制 |

### 2.3 结构 / 低秩

| 标题 | 年份 | Venue | 引用 | arXiv | 方法要点 |
|------|------|-------|------|-------|---------|
| DeepSeek-V2（MLA） | 2024 | arXiv | 1419 | 2405.04434 | 低秩 KV 联合压缩（潜变量 + 上采样），KV 减 ~93% |
| You Only Cache Once (YOCO) | 2024 | NeurIPS'24 | 151 | 2405.05254 | decoder-decoder，仅缓存一次 KV |

## 3. 结论

### 3.1 高星工程现状
几乎所有方法都以 **fork vLLM/SGLang、或 HF 原型** 形态存在；SnapKV/PyramidKV（nackliu）、KVQuant（SqueezeAILab）、KIVI/GEAR 均为 fork 仓库。无一是"官方插件"。llama.cpp 将 KV 量化作为配置原生支持（q8_0 等）。

### 3.2 工程落地约束（本项目关键）
- **任意位置重要性驱逐**（H2O/SnapKV/PyramidKV）需要注意力对"被驱逐 token 位置"做掩码或空洞语义 → 框架需内核级支持。
- **连续区间驱逐**（StreamingLLM / 滑窗 / RSWA）语义可由框架级缓存管理表达（释放整段 + 位置对齐）。
- 量化族中 **FP8 已被 vLLM 原生支持**；sub-8bit 需自定义 kernel。

### 3.3 对本项目（ai-compress，vLLM 0.28 插件）的启示
vLLM 0.28 已内置连续区间驱逐的全部机制（RSWA/Sink/SlidingWindow spec + manager + null-block 对齐），因此本项目选择**连续区间驱逐路线**：v1 复用 RSWA 式语义（prompt 可见 + 生成窗口有界），后续扩展 Sink+Window 激进语义，策略层支持自适应预算。详细技术依据见 `survey-vllm-integration.md` 与设计文档。

> 注：表中标记"ID 待复核"的条目为本次检索未能核实的次要论文，后续按需补全；不做推测性引用。
