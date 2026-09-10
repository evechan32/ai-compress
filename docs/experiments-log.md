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

## 8. SGLang 验证（未通过）

- 安装：`/root/sglang-venv`（virtualenv + system site packages），**sglang 0.5.19 安装成功**。
- 冒烟：`Engine(model_path=..., mem_fraction_static=0.6, disable_cuda_graph=True)` 启动失败：
  - `RuntimeError: FlashInfer requires GPUs with sm75 or higher`
  - `SIGQUIT/SIGKILL`，`EXITCODE=137`
- 判断：SGLang 的 flashinfer 路径在 sm_120 + 本机 CUDA 12.8 运行时下能力检测失败（与 vLLM 早期 "SM 12.x requires CUDA >= 12.9" 同类问题）。**SGLang 尚未验证通过**。

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
