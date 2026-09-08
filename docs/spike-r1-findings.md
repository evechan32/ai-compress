# R1 Spike 结论：标准模型走 RSWA 窗口注意力路径的可行性

> 日期：2026-09-08 | 实验服务器：2×RTX 5070 (sm_120), vLLM 0.28.0, Qwen2.5-1.5B-Instruct
> 实验脚本：`/root/ai-compress/spike/r1_inproc.py`（服务器临时，不入库）

## 问题

设计文档风险 R1：标准全注意力模型（Qwen2.5）能否在不 fork vLLM 的前提下，通过运行时注入切换到 RSWA 受限窗口注意力路径，且 KV 显存随生成长度有界、输出正常？

## 方法与注入面

两个进程级 monkey-patch（在 `vllm.general_plugins` 入口或主模块顶层执行，随 spawn 传播/同进程直接生效）：

1. **Attention 层替换**：`vllm.model_executor.models.qwen2.Attention` → 工厂函数，构造 `vllm.model_executor.layers.attention.rswa_attention.RSWAAttention(*args, rswa_window=WINDOW, **kwargs)`。
   - 原理：`RSWAAttention` 重写 `get_kv_cache_spec()` 返回 `RSWASpec` → 缓存管理器实例化 `RSWAManager`（而非 `FullAttentionManager`），逐 decode 步驱逐 prefill 尾与生成窗口之间的 gap 块（null-block 对齐保持位置编号）。
2. **窗口配置**：覆盖 `vllm.config.model.ModelConfig.rswa_window` property，在 HF config 未声明时返回 `WINDOW`。attention backend（FlashAttention）与 metadata 依据它构建窗口语义。

实验用同进程 `LLMEngine`（`VLLM_ENABLE_V1_MULTIPROCESSING=0` → InprocClient）手动 `step()` 循环，每 100 步经 `scheduler.kv_cache_manager.get_blocks()` 读取该请求的**真实（非 null）KV 块数**。prompt=1500 token，生成 2000 token，窗口 256。

## 结果

| 配置 | 每请求真实 KV 块数轨迹 | 行为 |
|---|---|---|
| 基线（全注意力） | 86 → 92 → 98 → … 无界增长，step1900 ≈ 204（≈(1500+1900)/16） | KV 随生成长度线性增长 |
| RSWA 强制 (window=256) | 86 → 103（step~300 起）→ **恒定 103 至 step 2000** | 生成超窗后 KV 有界 |

- 103 块 ≈ prompt 1500 (94 块) + 窗口 256 (16 块) + 块边界余量；与 RSWA 语义 O(prefix + window) 一致。
- 两次运行均正常完成 2000 token 生成（17s），无崩溃、无后端报错；FlashAttention 后端在本次配置下直接承载（未触发 FA4 mask_mod 可用性报错；decode 窗口语义正确性由"块数平台化 + 正常完成"佐证，**最终质量对照留待评测阶段**）。

## 结论

1. **机制可行**：标准 Qwen2.5 模型经两个 monkey-patch 点即可零 fork 启用 RSWA 驱逐；驱逐管理器、块回收、null 对齐、注意力窗口全部复用 vLLM 原生实现。
2. **KV 有界性实证**：生成超出窗口后每请求 KV 块数平台化，压缩收益可测量。
3. **适配层形态 = 形式 M（模型代码注入）**，具体为 Task 4/5 中的 `adapter.install()` 需做两件事：
   - 把目标架构模块命名空间里的 `Attention` 名替换为"可配置窗口的 RSWAAttention 工厂"；
   - 覆盖 `ModelConfig.rswa_window` 使后端/元数据感知窗口。
4. **约束与注意**：
   - 适配必须在模型构建前、且在各进程（含 EngineCore）生效 → 用 `vllm.general_plugins` 入口点（天然多进程加载）。
   - 仅对声明支持的架构（`AI_COMPRESS_TARGET_ARCHS`）注入；其余架构保持原生。
   - decode 正确性（窗口外内容被正确忽略、无信息泄漏到错误位置）需在 Task 6 评测中以"近期窗口内事实可答出 + 语义合理"验证。
   - RSWA 语义下 prompt(prefill) KV 全量保留 → 单轮超长 prompt 场景收益有限（符合 spec §3 语义分期：v1 主打长多轮对话生成段有界；Sink+Window 激进模式留 v1.x）。

## 对后续任务的直接影响

- Task 4 `adapter.install(cfg)`：实现形式 M 的两个 patch + 架构白名单校验 + 启用日志；旁路（ENABLE=0）零副作用。
- Task 5：把 spike 的 `RSWAAttention` 替换与 `rswa_window` 覆盖封装为可配置实现，并在启用态 smoke 验证块数平台化。
- Task 6 评测须覆盖"窗口内事实召回"以确认 decode 语义正确。
