# ai-compress 项目总报告

> 日期：2026-09-11 ｜ 项目：kvcompress（vLLM 零 fork KV 驱逐插件）+ sglang_kvx（SGLang 稀疏/散点探索）
> 本报告汇总全部实验与数据；细节与原始命令见 `docs/experiments-log.md`（§1–§22）。

---

## 0. 一句话结论

在"零 fork（不改框架源码）+ 即插即用（不重训）"的约束下：
- **可交付且可用的成果** = vLLM **RSWA 连续区间驱逐插件**：KV 有界、常规任务**逐字无损**、KV 受限并发 **+28%**；
- **可用的量化结论** = **FP8 KV（e4m3）**：容量 **~2×**，但质量平均 **-0.04 F1**（非逐字无损）；
- **散点驱逐（SnapKV/H2O/重要性/Quest）** 与 **KV 槽位回收** 在零 fork 下**无法兑现收益**，需内核/调度器级集成（已用多组实验界定并量化）。

---

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | 2 × RTX 5070（Blackwell sm_120），各 12GB，无 NVLink |
| 驱动 / CUDA | 580.173.02 / CUDA 13.0（驱动）；工具链 nvcc 12.8 |
| CPU / 内存 | 32 核 / 62GB |
| vLLM（系统） | 0.28.0（torch 2.13.0+cu130） |
| SGLang（venv） | 0.5.19（`attention_backend="triton"`，flashinfer 在 sm_120 不可用） |
| 模型 | Qwen2.5-1.5B-Instruct（主）、Qwen3-8B-AWQ（多架构验证） |
| 数据 | `/hy-tmp/longbench/data/`（LongBench v1 全 21 任务） |

---

## 2. vLLM RSWA 插件（kvcompress）——主交付物

机制：`vllm.general_plugins` 入口注入；`qwen2/qwen3` 模块的 `Attention` → `RSWAAttention` 工厂 + 覆盖 `ModelConfig.rswa_window`。语义 = **prompt 全保留 + 生成段滑动窗口**，逐 decode 步释放窗口外 gap 块。

### 2.1 功能正确性

| 测试 | 结果 |
|---|---|
| 本地单测（config/policy） | 8 passed |
| 服务器旁路（关闭态与原生一致） | 2/2 passed |
| 未知架构拒绝（不静默跑全注意力） | 通过 |

### 2.2 KV 有界性（同进程逐 100 步读取每请求真实 KV 块数）

Qwen2.5-1.5B：prompt=1500 token，生成 2000 token，窗口=256

| 配置 | 块数轨迹（step→blocks） | 稳态 |
|---|---|---|
| baseline | 86, 92, 98, …, step1900→**204**，结束→0 | 无界增长 |
| plugin-w256 | 86, 92, 98, 103, 104, 103, … step300–1900 **恒 103-104** | **平台化** |

Qwen3-8B-AWQ：prompt=1000，生成 1200，w256 → `57,63,69,75,74,74,…`（step300 起平台化 ~74-75）。

### 2.3 吞吐

**batch=1（非容量受限，合成数据）**

| Tag | eval_wall | 吞吐 |
|---|---|---|
| baseline | 5.32s | 129.8 tok/s |
| plugin-w256 | 5.33s | 129.7 tok/s |
| plugin-w1024 | 5.34s | 129.5 tok/s |

**容量受限并发（池≈7760 token；6 并发；各 prompt1000+gen800）**

| 配置 | 完成 | wall | 吞吐 |
|---|---|---|---|
| baseline | 6/6 | 13.8s | 347.8 tok/s |
| **plugin-w256** | 6/6 | **10.7s** | **446.5 tok/s（+28%）** |

### 2.4 无损性（逐字一致，`bench/agreement.py`，temp=0）

| 对比 | n | exact_match | mean_sim | 差异 |
|---|---|---|---|---|
| baseline vs plugin-w256（needle+multiturn） | 18 | **1.0** | 1.0 | 0 |
| baseline vs plugin-w1024 | 18 | **1.0** | 1.0 | 0 |
| 加入 3k/8k/12k 长文档（longqa） | 24 | **1.0** | 1.0 | 0 |
| baseline-q3 vs plugin-q3-w256（Qwen3-8B） | 18 | **1.0** | 1.0 | 0 |
| baseline-pc vs plugin-w256-pc（prefix caching 开） | 18 | **1.0** | 1.0 | 0 |
| plugin w256（pc 关）vs（pc 开） | 18 | **1.0** | 1.0 | 0 |

结论：信息位于 prompt 区时，插件与基线**逐字完全相同**。

### 2.5 Prefix caching 兼容性

| 项 | 结果 |
|---|---|
| 语义一致（pc 开） | baseline vs plugin 18/18 逐字一致 |
| KV 有界保持 | 平台化 103（与 pc 关相同） |
| 前缀复用收益 | req1 0.12s → req2/3 0.05s；基线与插件一致 |

### 2.6 精度损失探针（对抗场景尝试，v1–v5 未果）

目标：量化"单请求超长生成后依赖自身被驱逐内容"的损失。v1–v3 设计缺陷（多轮重 prefill，驱逐不发生）；v4/v5 因模型（1.5B/Qwen3-8B）无法稳定执行长格式指令而无效。**该损失量级仍为开放问题**（机制上：被驱逐 token 物理不可见）。

---

## 3. SGLang 探索（sglang_kvx）

### 3.1 环境与扩展点
- SGLang 0.5.19 可用（`attention_backend="triton"`）；**自定义 AttentionBackend 注册验证通过**（`kvx_triton` 注册+加载+生成）。
- 散点机制：decode 的 `kv_indices` 为扁平槽索引，可任意过滤 → 验证 prompt 402 token 过滤为 96（head64+win32）后生成正常。

### 3.2 合成 24 任务：各选择策略一致率（vs 完整注意力）

| 选择策略（保留≈384） | exact_match | mean_sim |
|---|---|---|
| 完整注意力 | 1.0 | 1.0 |
| 位置式 head256+win128 | **0.458** | 0.636 |
| Quest（页16 topk16 win128 sink64） | 0.223/0.256* | — |
| 注意力分数 top256+win128 | 0.125 | 0.335 |
| V 范数 top256+win128 | 0.083 | 0.260 |
| 位置式 head64+win32 | 0.042 | 0.282 |
| 注意力分数 top64+win32 | 0.000 | 0.213 |

（*Quest 为 LongBench F1，见下。）

### 3.3 LongBench 真实长文（F1，n=20）

| 子集 | 完整 | 位置 h256w128 | Quest | 注意力分数 | 多层mean obs64 | 多层max obs64 |
|---|---|---|---|---|---|---|
| qasper | 0.3448 | 0.2166 | 0.2229 | 0.1159 | 0.0949 | 0.0874 |
| 2wikimqa | 0.1036 | 0.1027 | 0.0963 | 0.0635 | 0.0588 | 0.0493 |
| multifieldqa_en | 0.4326 | 0.2657 | 0.2557 | 0.1853 | 0.1504 | 0.1656 |

- 完整注意力最优；**位置式 ≈ Quest > 注意力分数 > V 范数**，均显著低于完整。

### 3.4 计算收益（prompt 4000，生成 256，decode tok/s）

| 配置 | tok/s |
|---|---|
| 完整 | **153.9** |
| 位置散点 | 146.6 |
| Quest | 140.3 |

- **无算力收益**：原型每步 Python 重建索引 + 每步从池重算页 min/max，抵消并超过省下的读取。

### 3.5 KV 槽位释放（真回收尝试）

| 场景 | 结果 |
|---|---|
| 单请求 | 可用槽 **137645→137951（+306）**，生成正常（EXITCODE=0） |
| 多请求 | ❌ `ValueError: pool memory leak detected!`（调度器不变量）→ SIGQUIT |

- 结论：**backend-only 无法安全回收 KV；需调度器级集成（fork）**。开关 `KVX_FREE` 默认关闭。

### 3.6 HiCache（真无损 offload）

| 项 | 结果 |
|---|---|
| 逐字无损 | 开/关输出**完全相同**（加载 12.8s→20.2s） |
| 容量扩展 | ❌ 无：小 GPU 池下 HC 开/关都被 `max_total_num_tokens=11052` 拦住 |

- 结论：HiCache 是**分层前缀缓存**（复用），不提升可寻址上下文。

---

## 4. FP8 KV 量化（SGLang + triton）

| 维度 | 结果 |
|---|---|
| **容量** | **~2×**（`mem_fraction_static=0.30`：bf16 上限 11052 token；fp8_e4m3 接受 17778 token 输入） |
| **质量（F1, n=30）** | qasper 0.3059→0.2253；2wikimqa 0.1521→0.1148；multifieldqa 0.4176→0.3347；hotpotqa 0.1751→0.1613；triviaqa 0.1969→0.2089（平均 Δ≈**-0.04**） |
| **时延** | 4k prompt + gen256：bf16 156.5 → fp8 147.6 tok/s（**-6%**，非容量受限场景） |
| **e5m2** | **崩坏**（n=20：qasper 0.0104 / mfqa 0.0784），不可用 |
| 无损性 | 非逐字；度量级"近无损但有代价" |

---

## 5. 调研结论（方法可行性）

| 方法族 | 代表 | 零 fork 可行性 | 我们的实测/结论 |
|---|---|---|---|
| 连续区间驱逐 | RSWA/滑窗 | ✅ | **已实现**：KV 有界、常规任务逐字无损、+28% |
| 散点驱逐 | H2O/SnapKV/PyramidKV | ❌（需散点掩码内核） | 自研散点 backend 在真实 LongBench 上均显著掉分 |
| Query 块稀疏 | Quest/InfLLM/MagicPIG/PQCache | ❌ 无原生；官方均独立 HF/系统实现 | SGLang 有架构专用稀疏后端（DSA/NSA/MiniMax），非通用 |
| KV 量化 | FP8/INT4 | FP8 ✅ | **FP8 已量化**：容量 2×、质量 -0.04；INT4 需 fork |
| Offload/复用 | HiCache / KV connector | ✅（真无损） | HiCache 逐字无损但**不扩容量**；前缀复用未在本实验展开 |
| 架构级 | MLA/Mamba/YOCO | 换模型 | 收益最真实，但需相应模型权重 |

**零 fork 能力边界（已确认）**：
1. 只能表达**连续区间**驱逐（RSWA/滑窗）；
2. 散点驱逐、Query 块稀疏需**内核级**散点/检索支持；
3. KV 槽回收需**调度器级**集成；
4. 逐层窗口预算不可行（注意力掩码为模型级全局窗口）；
5. Sink+Window 激进模式不可行（vLLM 该机制为 OpenPanGu 架构专用）。

---

## 6. 可复用资产

| 资产 | 位置 | 用途 |
|---|---|---|
| RSWA 插件 | `kvcompress/` | vLLM 0.28 零 fork KV 有界化 |
| 评测工具 | `bench/`（gen_data/run_eval/report/agreement/sgl_run_eval/sgl_longbench/longgen_recall） | 质量/一致性/损失量化 |
| SGLang 实验 backend | `sglang_kvx/` | 散点/重要性/Quest 选择实验 |
| 文档 | `docs/`（survey×3、spec、plan、spike、bench-report、experiments-log §1–22、environment、fp8-kv-report、本报告） | 全量数据与结论 |
| 数据 | 服务器 `/hy-tmp/longbench/data/` | LongBench 全任务 |

### 关键复现命令

```bash
# vLLM 插件（系统环境）
AI_COMPRESS_ENABLE=1 AI_COMPRESS_RSWA_WINDOW=256 VLLM_USE_FLASHINFER_SAMPLER=0 \
  /usr/local/bin/python3 <script>
# SGLang 量化
/root/sglang-venv/bin/python bench/sgl_longbench.py --data /hy-tmp/longbench/data \
  --files qasper.jsonl 2wikimqa.jsonl multifieldqa_en.jsonl hotpotqa.jsonl triviaqa.jsonl \
  --n 30 --tag <tag> --attention-backend triton --kv-cache-dtype fp8_e4m3
# vLLM 构建环境（源码编译）
export LD_LIBRARY_PATH=/hy-tmp/vllm-build/lib:$LD_LIBRARY_PATH
/hy-tmp/vllm-build/bin/python <script>
```

---

## 7. 开放问题与后续方向

1. **精度损失量级**（单请求自引用）未量化——需更强指令遵循模型 + chat template。
2. **INT4/INT8 KV**（近无损 3.5-4×）：需 fork 自定义 kernel。
3. **Query 块稀疏的算力收益**：需 prefill 一次性元数据 + 内核融合（零 fork 做不到）。
4. **真·容量扩展的 offload**：vLLM KV connector / InfiniGen 式预取（HiCache 不扩容量）。
5. **KV 高效架构**（MLA/Mamba/YOCO）：换模型而非压模型，收益最真实。

---

## 8. 重要性说明（诚实披露）

- 质量评测为确定性采样（temp=0）；除标注外 n=20-30，单次运行存在噪声。
- "逐字无损"以固定输入输出完全一致为准；"近无损"指指标波动在可接受范围，不等于无损。
- 散点/重要性/Quest 为我们自研近似实现（非官方代码），性能不代表原论文最优实现。
- 本环境网络受限（HuggingFace 不可达、部分镜像不稳定），LongBench 经 ModelScope 镜像获取。
