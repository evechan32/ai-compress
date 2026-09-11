# Query-aware 块稀疏注意力：现有实现调研

> 调研日期：2026-09-11 ｜ 目标：判断 Quest / InfLLM / MagicPIG / PQCache 是否有可直接复用的实现

## 1. 官方实现一览

| 方法 | 仓库 | Stars | 集成方式 | 内核/系统 | 是否 vLLM/SGLang 原生 |
|---|---|---|---|---|---|
| **Quest**（ICML'24, arXiv 2406.10774） | mit-han-lab/Quest | 396 | HF Transformers patch（`quest/models/QuestAttention.py`） | FlashInfer kernels | ❌ |
| **InfLLM**（arXiv 2402.04617） | thunlp/InfLLM | 405 | 独立 HF fork + YAML 配置 | Triton 多段 flash-attn + faiss 检索 | ❌ |
| **MagicPIG**（ICLR'25 Spotlight, arXiv 2410.16179） | Infini-AI-Lab/MagicPIG | — | 系统级（GPU 投影 + CPU LSH 表/注意力） | FlashInfer（GPU 部分）+ CPU | ❌（LLaMA 专用） |
| **PQCache**（SIGMOD'25, arXiv 2407.12820） | ZhengtongYan/PQCache（镜像 HugoZHL/PQCache） | — | HF patch（`mistral_patch.py` / `llama31_patch.py`） | PQ 索引 + MIPS + CPU offload + GPU block cache | ❌ |

共性：都是**独立 HF/系统实现**，没有上游进入 vLLM/SGLang；均以"减少 decode 时的 KV 读取/内存搬运"为目标（**保留全部 KV，不减内存**）。

## 2. 框架原生支持现状

- **SGLang 0.5.19** 内置稀疏后端：`DeepseekSparseAttnBackend`（DSA，`dsa_backend.py`）、`MiniMaxSparseAttnBackend`（`minimax_sparse_backend.py`）、`nsa`、`dsv4`、`hpc_ops`、`wave`；server_args 有 hierarchical sparse attention 配置（`{"top_k":..., "device_buffer_size":..., "host_to_device_ratio":...}`）与 `--dsa-prefill/decode-backend`。
  - 这些是**架构专用**（DeepSeek DSA / MiniMax / NSA），**不能**直接套标准 dense 模型（Qwen2.5）。
  - 注：代码里大量 "quest" 文本命中实为 `request` 的子串，**不是 Quest 算法实现**。
- **vLLM 0.28**：无原生 query-aware 稀疏后端（fp8 KV 走 FlashInfer，另见环境限制）。

## 3. 结论与可行路径

1. **有官方实现**，但都要独立 HF/系统栈运行，无法直接插入我们的 vLLM/SGLang 插件体系。
2. **最契合我们的路线**：把 Quest 的"页上界估计"移植进已有的 `sglang_kvx` 散点 backend——
   - Quest 算法本身轻量：prefill 时按页记录 K 的 min/max，decode 用当前 query 估每页注意力上界，选 top-k 页；
   - 我们**已具备** decode KV 索引过滤能力，只需把"选择策略"从位置式/重要性换成"页上界估计"，用现有 triton 注意力（少喂索引），**无需新 kernel**；
   - 性质：保留全部 KV（不减内存），**降的是 decode 计算/搬运**（计算型压缩），论文报告近无损。
3. 备选：直接跑官方 HF 实现做论文复现（与插件体系正交）；或用 SGLang 内置 DSA/NSA（需换 DeepSeek 系模型）。

## 4. 与"无损"目标的关系
- Query-aware 稀疏（Quest 等）**不丢 KV**，只近似选择参与注意力的页 → 质量损失理论上很小（论文称 near-lossless），符合"尽量无损"；
- 与显存压缩（量化/offload/驱逐）正交，可叠加。
