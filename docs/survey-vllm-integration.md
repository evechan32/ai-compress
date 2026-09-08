# vLLM 0.28.0 KV 驱逐接入面调研报告

> 调研日期：2026-09-08 | 对象：vllm-project/vllm v0.28.0（服务器 site-packages 内已安装源码为准，交叉验证 GitHub v0.28.0 tag）
> 方法：源码定位 + 类/方法签名核对，无运行实验（运行实验见 spike 记录）

## 1. vLLM 插件机制

`vllm/plugins/__init__.py`：入口点组（entry point groups）：
- `vllm.general_plugins`：所有进程加载（process0 + EngineCore + worker）
- `vllm.model_plugins` / `vllm.platform_plugins` / `vllm.stat_logger_plugins` / `vllm.endpoint_plugins` / `vllm.io_processor_plugins`：各自作用域

**关键**：`vllm.general_plugins` 会在 EngineCore 子进程中也加载 → 独立包注册该组入口点即可在所有进程注入代码，无需改动 vLLM 源码。`VLLM_PLUGINS` 环境变量可过滤加载。**KV 缓存层本身没有官方 plugin hook**，但模型注册、量化方法等有注册表可扩展。

## 2. KV 缓存层结构（v1）

文件（`vllm/v1/core/`）：`kv_cache_manager.py`（KVCacheManager 门面）、`kv_cache_coordinator.py`（多 group 协调）、`single_type_kv_cache_manager.py`（具体 manager）、`block_pool.py`、`sched/`。

Manager 层次（v0.28.0）：

| 类 | 行号 | 语义 |
|---|---|---|
| `SingleTypeKVCacheManager` (ABC) | 36 | 抽象基类：`get_num_skipped_tokens()` 默认返回 0（全不跳） |
| `FullAttentionManager` | 680 | 全注意力，从不释放 |
| `RSWAManager(FullAttentionManager)` | 834 | R-SWA：prompt 全可见 + 生成段滑动窗口，逐 decode 步驱逐中段 gap |
| `SlidingWindowManager` | 880 | 经典滑窗：头部逐块释放 + null 填充 |
| `ChunkedLocalAttentionManager` | 1110 | CLA/分块局部注意力 |
| `MambaManager` | 1268 | Mamba 状态只留最后 token |
| `CrossAttentionManager` | 1763 | encoder-decoder 交叉注意力 |
| `SinkFullAttentionManager(FullAttentionManager)` | 1826 | sink + 全注意力（sink_len 需块对齐），有静态 sink 注意力层配合 |

**核心驱逐语义**（基类 `remove_skipped_blocks`，v0.28.0）：把"注意力窗口之外的整段连续 token 块"从块表移除并**用 null block 填充**（保持 token 位置编号对齐），释放物理块回池。`get_num_skipped_tokens(num_computed_tokens)` 是每类注意力的"可跳过 token 数"策略钩子。RSWA 变体额外支持**中段 gap 驱逐**（`num_prompt_tokens` 定位 prefill 尾）。

**注册机制**（v0.28.0，约 1860-1953 行）：`KVCacheSpecRegistry` 支持 `@register_kv_cache_spec` / `KVCacheSpecRegistry.register(spec_cls, manager_cls, uniform_type_base_spec=...)` 注册自定义 spec+manager；`register_all_kvcache_specs(vllm_config)` 统一注册内置 spec，并调用 `current_platform.register_custom_kv_cache_specs(vllm_config)` 供平台插件追加。

## 3. 注意力路径（限制窗口语义如何生效）

- `vllm/v1/attention/backends/flash_attn.py`：decode 支持 `rswa_window`/`rswa_window_tensor`，窗口语义通过 FA4 `rswa_mask_mod` 或 metadata 限制实际参与注意力的范围；后端选择见 `vllm/v1/attention/selector.py`。
- `vllm/model_executor/models/config.py`：RSWA 激活需要模型 HF config 携带 `rswa_window`，并走支持 `rswa_mask_mod` 的注意力后端。
- `StaticSinkAttention`（`vllm/model_executor/layers/attention/static_sink_attention.py`）→ 产出 `SinkFullAttentionSpec`，用于"全注意力 + 静态 sink"架构模型。

**推论**：窗口限制不是由"释放了块"自动产生，而是由模型注意力层 + metadata 声明可见范围。驱逐管理器与注意力语义必须**成对**匹配（释放的 = 注意力声明不可见的）。

## 4. 对插件设计的关键结论

1. **零 fork 可注入**：自定义 spec+manager 可通过 registry 注册；代码注入用 `vllm.general_plugins` 入口点（所有进程生效）。
2. **可表达的驱逐语义 = 连续区间**（前缀释放 / 中段 gap 释放 / sink+窗口），这正是 RSWA/Sink/SlidingWindow 已有机制。
3. **不可表达（零 fork）**：任意 token 散点驱逐（H2O/SnapKV 语义）——需要散点掩码的注意力内核，vLLM 无此抽象。
4. **标准模型（Qwen2.5 等）默认不驱逐**：spec 由模型架构代码产出（FullAttentionSpec）。要让标准模型启用驱逐，需要模型适配层——通过 vLLM 模型注册机制提供"受限窗口注意力变体"包装，或 import 期 monkey-patch 该架构的 attention spec 产出点。这是本项目主要工作量所在。
5. **环境注意**：RTX 5070 (sm_120) 上 flashinfer JIT 采样需要 CUDA≥12.9（机器为 12.8）→ 需 `VLLM_USE_FLASHINFER_SAMPLER=0`（插件可在入口统一设置）。

## 5. 复用清单（实现时对照）

| 要复用/参考 | 位置 (site-packages) |
|---|---|
| Registry + register | `vllm/v1/core/single_type_kv_cache_manager.py` (~1860+) |
| RSWA manager gap 驱逐 | 同文件 `RSWAManager` (~834) / `get_num_skipped_tokens` |
| null-block 对齐机制 | 同文件 ~240-280（skip 区间 null 填充） |
| RSWA decode 窗口 | `vllm/v1/attention/backends/flash_attn.py` (~297-303, 509-517) |
| 插件入口点 | `vllm/plugins/__init__.py`（组名 `vllm.general_plugins`） |
