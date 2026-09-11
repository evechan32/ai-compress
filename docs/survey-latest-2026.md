# 最新 KV 压缩研究（2025–2026）与可落地实现

> 检索日期：2026-09-12。来源：联网检索（arXiv API 在本环境被网络挡、S2 限流），以下 arXiv 编号来自检索结果，**正式引用前建议再核验**。

## 0. 两个"立刻可用"的发现

| 实现 | 说明 | 价值 |
|---|---|---|
| **NVIDIA KVpress**（`github.com/NVIDIA/kvpress`） | 统一实现多种 KV 驱逐/压缩方法（KVzip、Fast KVzip、ChunkKV、SnapKV…）+ Leaderboard（HF Space），支持 HF 模型 | 可直接 pip 安装、跑多方法对照；**是研究/复现的标准入口** |
| **R-KV 的 SGLang 移植**（`github.com/Zefan-Cai/R-KV`，2026-07） | 提供 **SGLang v0.5.14 patch**，在 FlashInfer decode 路径做**真·物理驱逐**，含 server batching / DP，Qwen2.5-Math-7B 上 95% GSM8K、吞吐最高 5.2× | 开源框架里"真正回收 KV"的现成路径（可尝试适配 0.5.19） |

## 1. Prefill 阶段压缩 prompt KV（训练无关，主流方向）

| 论文 | arXiv | 机制 | 备注 |
|---|---|---|---|
| **KVzip**（NeurIPS'25 Oral） | 2505.23416 | query-agnostic：用"上下文重建"打分，可跨查询复用压缩缓存；3–4× 内存、2× 延迟 | 已进 NVIDIA KVpress |
| **Fast KVzip** | 2601.17668 | 训练轻量 gate 蒸馏 KVzip 分数，prefill+decode 双阶段，≤70% 驱逐近无损 | 训练 <1 张 H100 小时 |
| **BeaconKV**（ICML'26） | 2609.04971 | 维护"信标 query"簇代表，预测长推理中被回访的 KV；5.8× 内存↓ / 4.3× 吞吐↑ | 面向 reasoning 长 CoT |
| **DistillCache** | 2608.08878 | RL（REINFORCE + 逐步 KL 奖励）学习驱逐策略；25% 预算保 94.2% LongBench | 学习式驱逐 |
| **RestoreKV** | 2608.01247 | 驱逐后加少量"恢复 token"，LoRA 自蒸馏生成补偿缓存；5% 预算 RULER 38→73 | 选择+恢复 |
| **TwinKV** | 2608.27128 | 免注意力信号：成对 key 冗余去重，作为"修复层"叠加在任意驱逐策略上 | 可组合 |
| **Minima-KV** | 2608.23834 | 混合格式分页注意力：Anchor 页 FP8 + 旧页 TQ3，全局在线 softmax 合并；3.5× vs BF16 | 系统向，LongBench v2 掉 0.4-0.8pt |
| **CompressKV** | 2606.24467 | 识别"语义检索头 (SRH)"+ 离线层预算；3% 预算保 97% QA、0.7% 保 90% NIAH | 头级+层预算 |
| **IndexMem** | 2605.25475 | 可学习 indexer + 潜记忆（压缩被驱逐信息，可回读） | 学习式 + 补偿 |
| **CacheCraft / FRC** | 2608.14555 | 用 LLM 程序进化自动搜索驱逐策略（三信号打分） | 自动化策略发现 |
| ChunkKV | 2502.00299 | 语义块级保留（chunk top-k） | 常被作为基线 |
| KeyDiff | 2504.15364 | key 相似度差异作为重要性 | 免注意力权重 |
| ExpectedAttentionPress | 2510.00636 | 期望注意力分数（query 分布） | 免在线统计 |
| RocketKV（ICML'25） | — | 两阶段：粗粒度永久驱逐 + 细粒度混合稀疏注意力 | 报告 400×/3.7× |
| PagedEviction（EACL'26 Findings） | — | 结构化块级剪枝 | 块级 |
| LookaheadKV | 2603.10899 | 无生成长度的"未来 query"预测驱逐 | |
| Lookahead Q-Cache（EMNLP'25） | — | 伪 query 提升驱逐一致性 | |
| R-KV（NeurIPS'25） | 2505.24133 | 冗余感知（重要性−冗余），面向长推理 decode；10–34% 预算 parity | 有 SGLang 移植 |

## 2. 低秩 / 少维度（对应"用更少维度存"）

| 论文 | arXiv | 机制 |
|---|---|---|
| **STAR-KV**（ICML'26 Spotlight） | 2606.08382 | 可微软阈值自适应 rank（head/block 级）+ 混合分解 + 低秩感知混合精度量化；75% 低秩压缩，配合量化 **20×**；Triton kernel 6.9× attn、3.1× 端到端 |
| Palu | 2407.21118 | 低秩投影 K/V（离线分解，无运行时开销） |

## 3. 可训练/原生稀疏注意力（训练期，非即插即用）

| 论文 | arXiv | 机制 |
|---|---|---|
| **NSA**（ACL'25） | 2502.11089 | 三分支：压缩块 + 选择块 + 滑窗，端到端可训练，64k 全生命周期加速 |
| MoBA（Kimi） | 2502.13189 | 把块当 MoE 专家，每 query 路由 top-k 块 |
| DSA（DeepSeek-V3.2） | 2512.02556 | 轻量 indexer 打 token 分，注意力只看 top-k（GLM-5 系在生产用） |
| MSA（MiniMax） | 2606.13392 | DSA 的 learned indexer + max-pool 到块，NSA 骨架 |
| SeerAttention | 2410.13276 | 可学习门控预测块稀疏（自蒸馏训练 gate） |
| XAttention | 2503.16428 | 用"反对角线和"估块重要性，即插即用 |
| DuoAttention | 2410.10819 | 检索头（全 KV）+ 流式头（滑窗） |

## 4. 与本项目的关系

- 我们已实测的事实（LongBench 上丢 prompt 中段掉分、注意力分布 top10% 占 68–77%）与这一波工作的动机一致；但新方法（KVzip/RestoreKV/IndexMem/CompressKV）在**选择信号**（重建/学习/头级）与**补偿机制**（潜记忆/恢复 token）上明显比我们实现的简单版本更好。
- **零 fork 约束下**：这些仍多需自定义注意力/内核；KVzip 走 HF+KVpress，R-KV 有 SGLang patch（真驱逐）。
- **可立即可做的对照**：装 NVIDIA KVpress 在 Qwen2.5 上跑 LongBench，与我们的 RSWA / 散点 backend 数据对照；或尝试把 R-KV 的 SGLang patch 适配到 0.5.19。

## 5. 建议下一步（择一）

1. **装 NVIDIA KVpress** → 多方法（KVzip/ChunkKV/SnapKV…）在 LongBench 上的压缩-质量对照（最快拿到"最新方法"实测）。
2. **适配 R-KV 的 SGLang 0.5.14 patch 到 0.5.19** → 开源框架里真正物理驱逐的实测。
3. **STAR-KV/低秩** → 回答"少维度能压多少"（需 fork kernel，成本较高）。
