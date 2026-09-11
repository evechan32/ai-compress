# KVzip 实现解析（源码级）

> 来源：`github.com/snu-mllab/KVzip`（MIT）｜论文 arXiv 2505.23416（NeurIPS'25 Oral）
> 关键文件：`model/wrapper.py`、`attention/kvcache.py`、`attention/score.py`

## 1. 总体流程

```
prefill(context) ──► 常规 KV cache
        │
        ├─ scoring()：让模型"复述原上下文"，记录每个 KV 被多少注意力读到（重要性分数）
        │
        ├─ prune(ratio)：按分数取阈值 → 每个 (layer, kv_head, ctx_pos) 的 valid 掩码
        │      （system prompt 与 query 永远保留）
        │
        └─ prepare_init()：物理驱逐未保留 KV → 压平成变长布局 → 变长 FlashAttention
                                   （EvictCache）
```

## 2. 打分 = "上下文重建"（query-agnostic 的关键）

`wrapper.self_task()` 构造"复述任务"输入（分块，每块 2000 token）：

- 第 1 块提示词：`"\n\nRepeat the previous context exactly."`
- 后续块：`"\n\nRepeat the part of the previous context exactly, starting with "` + **上一块末尾 8 token**（postfix，保证衔接）

`wrapper.scoring()`：

1. `kv.init_score()` 清零；
2. 逐块把 `kv.start_idx/end_idx` 限定到该块上下文范围，调用模型做一次前向（`update_cache=False`——只读分数、不写 cache）；
3. 把"复述时各 KV 收到的注意力"累加到 `kv.score`（`attention/score.py` 的 `KVScore/HybridKVScore`；按 (layer, kv_head, ctx_pos) 粒度）。

含义：**一个 KV 越能帮助模型重建原文，就越重要**（自监督代理），而不是"最近 query 看不看它"（SnapKV 的观察窗口，有局部性/query 偏差）。这就是它"query-agnostic、可跨查询复用"的来源。

## 3. 剪枝：阈值 → valid 掩码

`EvictCache.prune(ratio, level)`：

- `_threshold(score, ratio)`（或 `_threshold_uniform`）→ `self.valid`（bool，形状 = layer × kv_head × ctx_len）；
- `level` 可选：`pair`（逐 KV 对）、head 级（整头驱逐，配离线 head score）、uniform；
- 保留规则见 `_get_valid()`：`valid_pad`（**system prompt 永远保留**）+ context 掩码 + 后续 query/生成 token 的 `ones`（**query 也永远保留**）。即**只裁 context 本身**。

## 4. 物理驱逐 + 高效注意力（工程核心）

`EvictCache.prepare_init()`：

```python
self.key_cache[layer] = self.key_cache[layer].contiguous().view(-1, dim)[valid.view(-1)]
```

- 先按掩码把 KV **紧凑化**（每个 KV head 保留长度不同 → 变长）；
- 生成 `len_k / max_len_k / cu_len_k`（每头长度与累积偏移）元数据；
- 配合自定义 CUDA 辅助 `tiny_api_cuda.update_flatten_view`，后续 decode 用**变长 FlashAttention**（flatten 布局）直接算，无需冻结的 dense shadow；
- `RetainCache`：不物理驱逐、只在注意力时按掩码子采样 → 一次 prefill 可评测多个 ratio（用于 benchmark）；
- `slice()`：生成结束后把"query/生成 token"的 KV 摘除，让 context 缓存可被**下一个 query 复用**（多 query 场景）。

## 5. 两种部署模式

| 模式 | 做法 | 开销 |
|---|---|---|
| **context-dependent** | 每个 context prefill 后跑一次复述打分 | 有 per-context 打分开销（分块前向） |
| **context-independent** | 预计算 **head-level score**（`load_score=True`，仓库 `utils/head_score` 提供 Llama3.1-8B / Qwen2.5-7B/14B），整头驱逐，ratio≈0.6 | 零运行时打分开销；替代 DuoAttention 数小时优化（几次前向 <1 分钟） |

## 6. 集成方式

- HF `transformers` 的 `DynamicCache` 子类 + 模型 monkeypatch（`model/monkeypatch.py`），支持 LLaMA3 / Qwen2.5 / Qwen3 / Gemma3；
- 变体：`int4static`（INT4 KV）、`hybrid_static`（HybridCache，静态 KV）也在仓库里；
- **进 NVIDIA KVpress**（Leaderboard 支持），Fast KVzip（2601.17668）进一步用 gate 蒸馏分数，去掉运行时打分开销。

## 7. 与本项目的关系

| 维度 | 我们的 kvx_scatter（SGLang） | KVzip |
|---|---|---|
| 选择信号 | 位置 / 单层注意力 / V 范数 / 页上界（Quest） | **上下文重建**注意力（query-agnostic） |
| 物理驱逐 | 只过滤注意力索引（未回收显存）；KVX_FREE 尝试失败 | **真物理紧凑化** + 变长 FA |
| 系统 | SGLang 自定义 backend | HF + 自定义 CUDA 辅助；框架侧走 KVpress |
| 保留规则 | prompt + 窗口 | system prompt + query 永不裁；只裁 context |

启示：我们前面的失败（散点掉分、槽回收撞不变量）部分是**选择信号弱 + 无紧凑化内核**；KVzip 用"重建打分 + 紧凑化 + 变长 FA"补齐，并明确把 system/query 排除在驱逐之外。要在开源推理框架里复现，现实路径是 **KVpress（HF）** 或参考其 flatten 思路做自定义 kernel。
