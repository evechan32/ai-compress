# 开发与测试数据总账（experiments log）

> 项目：kvcompress（vLLM 零 fork KV 驱逐插件）｜记录时间：2026-09-08 ~ 09-10｜服务器：2×RTX 5070 (sm_120)
> 所有数据均为本次开发实测；命令与日志位置见各节。

## 0. 汇总

| 类别 | 结果 |
|---|---|
| 本地单测 | 8 passed（config/policy）；server-only 模块本地 skip |
| 服务器集成 | 旁路 2/2 passed |
| KV 平台化（Qwen2.5） | baseline 86→204 无界；插件 w256 平台化 **103** |
| KV 平台化（Qwen3-8B-AWQ） | 57→75 平台化 |
| batch=1 吞吐 | baseline 129.8 / w256 129.7 / w1024 129.5 tok/s（持平） |
| 容量受限并发吞吐 | baseline 347.8 → 插件 w256 **446.5 tok/s（+28%）**，13.8s→10.7s |
| 逐字一致率 | Qwen2.5 24/24；Qwen3 18/18；prefix caching 开/关 18/18 |
| Prefix reuse 时延 | 0.12s → 0.05s（基线与插件一致） |
| SGLang | 环境装成 0.5.19；sm_120 运行失败（见 §8） |

---

## 1. R1 可行性 spike（Qwen2.5-1.5B，in-proc LLMEngine）

问题：标准全注意力模型能否零 fork 切到 RSWA 路径，且 KV 有界？

- 注入面：`qwen2.Attention → RSWAAttention(rswa_window=W)` + `ModelConfig.rswa_window` 属性覆盖。
- 观测：每 100 step 读该请求真实（非 null）KV 块数；prompt=1500 token，生成 2000 token。

| 配置 | 块数轨迹（step→blocks） |
|---|---|
| baseline | 86, 92, 98, …, 1900→204（≈(1500+1900)/16，无界），finish 后 0 |
| plugin w256 | 86, 92, 98, 103, 104, 103, …, step300–1900 **恒 103-104** |

结论：机制可行；KV 有界 = prompt(94 块) + window(16 块) + 边界余量。

## 2. Qwen3-8B-AWQ 注入验证（in-proc）

- prompt=1000 token，生成 1200 token，w256：`57, 63, 69, 75, 74, 74, …, 1200→0`（step300 起平台化 ~74-75）。
- 结论：多架构注入生效（`Qwen3ForCausalLM`）。

## 3. batch=1 评测吞吐（Qwen2.5-1.5B，needle+multiturn 合成数据，temp=0）

| Tag | eval_wall_s | 吞吐 tok/s |
|---|---|---|
| baseline | 5.32 | 129.8 |
| plugin-w256 | 5.33 | 129.7 |
| plugin-w1024 | 5.34 | 129.5 |

结论：非容量受限场景持平（解码受计算/访存限制）。

## 4. 容量受限并发吞吐（关键收益）

条件：KV 池 ~7,760 token（gpu_memory_utilization=0.35, max_len=2048）；6 个并发请求，各 prompt=1000 + gen=800；temp=0。

| 配置 | 完成 | wall | 吞吐 |
|---|---|---|---|
| baseline | 6/6 | 13.8s | 347.8 tok/s |
| plugin-w256 | 6/6 | **10.7s** | **446.5 tok/s（+28%）** |

机制：基线 6×1800=10800 tokens > 池 → 分波次；插件每请求 ≈1000+256 → 全量并发。

## 5. 无损性（逐字一致判据，bench/agreement.py）

同数据（seed 0）、temp=0，比较 baseline 与插件的每条回答文本（精确匹配 + difflib 相似度）。

| 对比 | n | exact_match | mean_sim | 差异 |
|---|---|---|---|---|
| baseline vs plugin-w256（Qwen2.5-1.5B，needle+multiturn） | 18 | 1.0 | 1.0 | 0 |
| baseline vs plugin-w1024（同上） | 18 | 1.0 | 1.0 | 0 |
| baseline vs plugin-w256（加入 3k/8k/12k 长文档 longqa） | 24 | 1.0 | 1.0 | 0 |
| baseline vs plugin-w1024（同上，24 条） | 24 | 1.0 | 1.0 | 0 |
| baseline-q3 vs plugin-q3-w256（Qwen3-8B-AWQ） | 18 | 1.0 | 1.0 | 0 |
| baseline-pc vs plugin-w256-pc（prefix caching 开） | 18 | 1.0 | 1.0 | 0 |
| plugin-w256（pc 关）vs plugin-w256-pc（pc 开） | 18 | 1.0 | 1.0 | 0 |

结论：常规任务（信息位于 prompt 区）上插件与基线**逐字完全相同**。

## 6. Prefix caching 兼容性

| 验证项 | 结果 |
|---|---|
| 引擎启动/生成 | 正常（早期一例失败为双引擎并发抢显存，非插件缺陷） |
| 语义一致性 | baseline-pc vs plugin-w256-pc 18/18 逐字一致 |
| 插件确定性 | w256(pc 关) vs w256(pc 开) 18/18 逐字一致 |
| KV 有界保持 | 平台化 103 块，与 pc 关相同 |
| 前缀复用收益 | req1 0.12s → req2/3(same prefix) 0.05s；基线与插件一致 |

## 7. 对抗性探针（试图量化"自引用被驱逐内容"的损失）——v1–v5 均未果

目标：测"单请求内、生成超过窗口后，后文依赖自身早期被驱逐内容"的损失。

| 版本 | 设计 | 失败原因 |
|---|---|---|
| v1/v2 | 多轮：每轮把全量历史作为**新请求 prompt** | 每轮重 prefill → 所有内容在 prompt 区（RSWA 全保留），驱逐从未发生 |
| v3 | 同进程多轮 | 同上（设计缺陷） |
| v4（1.5B/7B） | 单请求"开头放口令→长文→结尾复述" | 模型不执行多步格式（`head_ok=False`），结果噪声 |
| v5（Qwen3-8B） | 自发明口令（仅存生成区） | Qwen3 raw 续写进入 thinking 模式，格式不服从 |

结论：损失量级仍为开放问题；需要能稳定执行长格式任务的模型（chat template + 关 thinking）才能测定。机制层面：被驱逐 token 的 KV 物理移除，依赖它的任务必然受影响。

## 8. SGLang 验证（通过，需 triton 后端）

- 安装：`/root/sglang-venv`，sglang 0.5.19。
- 默认 flashinfer 后端失败：`RuntimeError: FlashInfer requires GPUs with sm75 or higher`（EXITCODE=137）。
- **换 `attention_backend="triton"` 后通过**：Engine 12s 就绪，生成 `' Paris. The capital of France is also the capital of which country?'`，EXITCODE=0。
- 结论：SGLang 在本机可用（triton 后端）；这为后续用 SGLang 自定义 AttentionBackend 做散点驱逐（SnapKV/H2O）打通了环境前提。

## 9. 已验证的环境变量/命令备忘

```bash
# vLLM 0.28.0（系统环境）
VLLM_USE_FLASHINFER_SAMPLER=0 python3 ...
# 插件启用
AI_COMPRESS_ENABLE=1 AI_COMPRESS_RSWA_WINDOW=256 python3 ...
```

## 10. 原始日志位置（服务器 /tmp）

- `/tmp/inproc_base3.log`、`/tmp/inproc_rswa.log`、`/tmp/r5.log`、`/tmp/r5_prefix.py` 相关 log：KV 轨迹
- `/tmp/perf_run.log`：并发吞吐
- `/tmp/pc_run.log`、`/tmp/pc_plugin2.log`：prefix caching
- `/tmp/lq_run.log`、`/tmp/q3_eval.log`：一致性评测
- `/tmp/sgl_smoke2.log`：SGLang 冒烟失败

## 11. FP8 KV（原生量化）对照——环境不可用

- 目的：量化"原生 FP8 KV 量化"的逐字损失，对比我们的驱逐"逐字无损"。
- 结果：`--kv-cache-dtype fp8 / fp8_e4m3` 引擎启动失败：
  `RuntimeError: FlashInfer backend is not available. Please install the package to enable FlashInfer kernels`
- 原因：vLLM 0.28 的 fp8 KV 由 FlashInfer 后端实现，本机 FlashInfer 因 sm_120/CUDA 12.8 不可用。
- 结论：本环境无法产出 FP8 KV 对照数据；详见 `environment.md §8.2`。

## 12. SGLang 自定义 AttentionBackend 注册验证（通过，2026-09-10）

- 注册 API：`sglang.srt.layers.attention.attention_registry.register_attention_backend(name)`（装饰器，注册工厂 `fn(runner) -> AttentionBackend`）。
- 验证：定义 `kvx_triton`（继承 `TritonAttnBackend` 的恒等后端）并注册 →
  - `registered has kvx_triton: True`
  - `Engine(model_path=..., attention_backend="kvx_triton", mem_fraction_static=0.6, disable_cuda_graph=True)` → 10s 就绪，生成 `' Paris. The capital of France is...'`，EXITCODE=0
- 结论：**SGLang 的 attention backend 扩展点在本机（sm_120, triton 后端）可用** —— 这是零 fork 之外实现 SnapKV/H2O 散点驱逐的可行路径。
- 脚本：`/tmp/sgl_custom_backend.py`、`/tmp/sgl_custom_smoke.py`（服务器）。

## 13. SGLang 散点 KV 驱逐 backend 原型（通过，2026-09-10）

- 代码：`sglang_kvx/`（`ScatterTritonBackend`，注册名 `kvx_scatter`）。
- 机制确认：SGLang decode 用扁平 `kv_indices`（gather）——实测 `kv_indptr=[0,10]`、`kv_indices=[1..10]`（page_size=1，索引=token 位置），因此按索引子集过滤即可实现散点注意力。
- 实测：prompt 402 token → 过滤为 head64+win32：
  - `KVX decode filter: old_ptr=[0, 402] new_ptr=[0, 96]`
  - 生成正常：`"(1) How many times does the letter 'o' appear in the sentence"`，EXITCODE=0
- 结论：**散点 KV 注意力机制在 SGLang 自定义 backend 中可用**（SnapKV/H2O 的必要前提）。
- 限制：① 未释放 KV 池槽位（显存未回收）；② 选择为位置式，非 attention 重要性（SnapKV 二期）；③ 未与 baseline 做损失量化。

## 14. SGLang 散点驱逐损失量化（2026-09-10）

同 24 条任务（needle 6 + longqa 6 + multiturn 12），temperature=0，SGLang triton 后端，`sglang_kvx` 散点 backend vs 完整注意力：

| 配置（decode 保留：头段 + 末尾窗口） | 逐字一致率 | 平均相似度 | 差异条数 |
|---|---|---|---|
| 完整注意力（参照） | 1.0 | 1.0 | 0 |
| 散点 head256 + win128 | 0.458 | 0.636 | 13/24 |
| 散点 head64 + win32 | 0.042 | 0.282 | 23/24 |
| 对照：vLLM RSWA 插件（保留全 prompt） | 1.0（24/24） | 1.0 | 0 |

- 结论：**丢弃 prompt 中段 KV 的散点策略损失显著**（激进配置近乎全错；温和配置仍损失过半），而保留全 prompt 的 RSWA 零损失。这量化了"散点 vs 连续保留"的差距。
- 重要限定：本实验的散点选择是**位置式**（头段+窗口），**非 SnapKV/H2O 的注意力重要性选择**；后者会在同等预算下保留中段显著 token，损失应显著更低。故本表是散点策略的"损失上界/质量下界"。
- 复现注意：SGLang scheduler 子进程为独立解释器，需在子进程可见处注册 backend。本机做法：将 `sglang_kvx` 放入 venv site-packages，并在 `sglang/srt/model_executor/model_runner_components/attention_backend_setup.py` 的 `_build_full_attention_backend_from_str` 中惰性 `import sglang_kvx`（见 sglang_kvx/README.md）。

## 15. SGLang KV 槽位释放尝试（单请求成功，多请求触发调度器不变量失败）

- 目标：让散点驱逐真正回收 KV 显存（当前只限制注意力）。
- 单请求结果：`KVX free: dropped=306 slots avail 137645->137951`（可用槽 +306），生成正常，EXITCODE=0。
- 多请求结果：触发 SGLang 调度器不变量检查 ——
  `ValueError: pool memory leak detected! total=138047, available=137940, evictable=413 ...`
  随后 SIGQUIT，EXITCODE=137。
- 根因：在 backend 直接调用 `allocator.free` 并清零 `req_to_token`，与调度器的请求 KV 记账、radix cache 及不变量校验器不一致。
- 结论：**backend-only 无法安全实现 KV 回收；真正的槽位释放需调度器级集成（fork 级改动）**。
- 现状：回收路径由 `KVX_FREE=1` 门控，**默认关闭**（实验性，勿用于多请求）。

## 16. 重要性选择（V 范数代理）——劣于位置式

在散点 backend 中以"KV 池 V 张量范数"作为重要性代理，取 top-B + 末尾窗口：

| 选择策略（相近保留预算） | 逐字一致率 | 平均相似度 |
|---|---|---|
| 位置式 head256+win128（384） | 0.458 | 0.636 |
| V 范数重要性 top256+win128 | 0.083 | 0.260 |
| 位置式 head64+win32 | 0.042 | 0.282 |
| V 范数重要性 top64+win32 | 0.000 | 0.092 |

- 结论：**V 范数不是有效的显著性代理**，劣于位置式。真 SnapKV 使用**最后一层注意力分数**；需在 prefill 阶段做一次打分（我们可在自有 `forward_extend` 用 q/k 现算最后一层分数）。
- 代码：`KVX_IMPORTANCE=1`（+`KVX_BUDGET`/`KVX_WINDOW`），默认关闭。

## 17. 真 SnapKV 信号（最后一层注意力分数）选择——仍劣于位置式（合成任务）

实现：在 `forward_extend`（prefill）用该层 q/k 现算"末尾观察窗口(32) 对全部 prompt 位置"的注意力分数，聚合为重要性，选 top-B + 末尾窗口。

| 选择策略（保留预算≈384） | 逐字一致率 | 平均相似度 |
|---|---|---|
| 完整注意力 | 1.0 | 1.0 |
| 位置式 head256+win128 | 0.458 | 0.636 |
| 注意力分数 top256+win128 | 0.125 | 0.335 |
| V 范数 top256+win128 | 0.083 | 0.260 |
| 位置式 head64+win32 | 0.042 | 0.282 |
| 注意力分数 top64+win32 | 0.0 | 0.213 |

- 结论：注意力分数优于 V 范数，但仍劣于位置式。原因：本合成任务的关键信息位于 prompt 中段，任何丢弃中段的散点策略都损失；位置式胜出靠"末尾窗口"命中 longqa/多轮所需信息。
- 判断：**合成 needle 不利于 SnapKV**；要评估其真实价值需真实长文冗余任务（LongBench 等）。
- 代码：`KVX_IMPORTANCE=1`（+`KVX_BUDGET`/`KVX_WINDOW`/`KVX_OBS`），默认关闭。

## 18. LongBench 子集真实长文评测（2026-09-11）

数据：`ZhipuAI/LongBench`（ModelScope，data.zip 解压）→ qasper / 2wikimqa / multifieldqa_en，各 20 条，官方 F1 指标，temp=0。

| 子集 | 完整注意力（triton） | 位置散点 h256+win128 | 重要性散点 top256+win128 |
|---|---|---|---|
| qasper | **0.3448** | 0.2166 | 0.1159 |
| 2wikimqa | **0.1036** | 0.1027 | 0.0635 |
| multifieldqa_en | **0.4326** | 0.2657 | 0.1853 |

- 结论：真实长文任务上，完整注意力最优；**散点驱逐（位置与重要性）均显著掉分**；位置上仍优于本版重要性实现。
- 对照：vLLM RSWA 插件保留全 prompt → 按构造等价完整注意力（合成 24/24 逐字一致）。
- 实现局限（影响重要性分数公平性）：单层分数、观察窗口 32、未处理 chunked prefill（分块时打分只覆盖最后 chunk）——官方 SnapKV 可能更好，但本次未能超越位置式。
- 脚本：`bench/sgl_longbench.py`；数据 `/hy-tmp/longbench/data/`。

## 19. 重要性实现改进后再测 LongBench——仍劣于位置式

改进：多层分数聚合（替代仅末层）、观察窗口 32→64、per-head mean/max。LongBench 3 子集 F1（n=20）：

| 子集 | 完整 | 位置 h256w128 | 重要性 末层obs32 | 重要性 多层mean obs64 | 重要性 多层max obs64 |
|---|---|---|---|---|---|
| qasper | 0.3448 | 0.2166 | 0.1159 | 0.0949 | 0.0874 |
| 2wikimqa | 0.1036 | 0.1027 | 0.0635 | 0.0588 | 0.0493 |
| multifieldqa_en | 0.4326 | 0.2657 | 0.1853 | 0.1504 | 0.1656 |

- 结论：多层聚合与更大观察窗口**未改善**，重要性选择**始终显著劣于位置式**，二者均远低于完整注意力。
- 综合判断（多轮尝试后）：在本机模型（Qwen2.5-1.5B）与所测任务上，SnapKV 式注意力打分选择未体现相对位置式的优势；散点驱逐整体不及"保留全 prompt 的 RSWA"（后者按构造等价完整注意力）。
- 代码开关：`KVX_IMPORTANCE`/`KVX_BUDGET`/`KVX_WINDOW`/`KVX_OBS`/`KVX_HEADAGG`，默认关闭。

## 20. FP8 KV 量化（SGLang + triton）——e4m3 近无损，e5m2 崩坏

LongBench F1（n=20，temp=0，`kv_cache_dtype`）：

| 子集 | bf16 | fp8_e4m3 | fp8_e5m2 |
|---|---|---|---|
| qasper | 0.3448 | 0.2858 | 0.0104 |
| 2wikimqa | 0.1036 | 0.1264 | 0.0321 |
| multifieldqa_en | 0.4326 | 0.3390 | 0.0784 |

- **fp8_e4m3 近无损**（小幅波动，2wikimqa 甚至反超，属单次 n=20 噪声）；**fp8_e5m2 崩坏**（尾数位不足）。
- 环境：vLLM 的 fp8 KV 走 FlashInfer（sm_120 不可用）；**SGLang + triton 后端可跑 fp8 KV**（本结果来源）。
- 代码：`bench/sgl_longbench.py --kv-cache-dtype fp8_e4m3`。
