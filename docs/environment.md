# 测试服务器环境档案

> 记录时间：2026-09-10 ｜ 主机：i-2.gpushare.com:29196（SSH）

## 1. 硬件

| 项 | 规格 |
|---|---|
| GPU | 2 × NVIDIA RTX 5070（Blackwell sm_120），各 12,227 MiB 显存，无 NVLink |
| 驱动 | 580.173.02，CUDA Version 13.0（驱动侧） |
| CPU / 内存 | 32 核 / 62 GB |
| OS | Ubuntu 22.04.5 |

## 2. 存储布局（重点，避免误删）

| 挂载 | 大小 | 用途 | 备注 |
|---|---|---|---|
| `/`（overlay） | 30G | 系统 + 系统 Python 包（vLLM/torch/CUDA） | 余 ~9GB |
| `/hy-tmp`（xfs, `/dev/mapper/vgdata-lvdata`） | 50G | **既有资源区** | 余 ~11GB |
| 额外块设备 | `nvme1n1` 1.8T（未格式化/未挂载）；`nvme0n1p2` 233G 以文件 bind 挂到 `/usr/bin/nvidia-smi`（异常镜像配置） | 勿动 |

**`/hy-tmp` 内含（不属于本项目，勿删）：**
- `/hy-tmp/models/Qwen3-Coder-30B-A3B-Instruct-FP8`（**30GB**，FP8，4 shards）
- `/hy-tmp/models/OLMoE-1B-7B-0125-Q3_K_L.gguf`（3.6GB，llama.cpp 用）
- `/hy-tmp/vllm`（5.9GB，vLLM 源码树）/ `/hy-tmp/llama.cpp` / `/hy-tmp/sglang-venv`

**模型（本项目使用）**：`/models/qwen2.5-1.5b-instruct`（2.9GB，完好）。

## 3. 运行时环境

| 项 | 值 |
|---|---|
| 系统 Python | `/usr/local/bin/python3` 3.11.12 |
| vLLM（系统） | **0.28.0**，装于 `/usr/local/lib/python3.11/dist-packages/vllm` |
| torch | 2.13.0+cu130 |
| CUDA toolkit | `/usr/local/cuda-12.8`（nvcc 12.8） |
| 必需环境变量 | `VLLM_USE_FLASHINFER_SAMPLER=0`（sm_120 上 flashinfer JIT 采样需 CUDA≥12.9） |
| 网络 | PyPI 可达；HuggingFace 不可达（用 `hf-mirror.com` / ModelScope）；GitHub SSH 可用 |

## 4. vLLM 构建环境（/vllm-build，已重建）

原 `/vllm-build`（11GB conda 环境）在一次腾空间的误操作中被删除。已重建等价环境：

| 项 | 值 |
|---|---|
| 位置 | `/hy-tmp/vllm-build`（Python 3.11.16）；软链 **`/vllm-build` → `/hy-tmp/vllm-build`** |
| 创建方式 | `/usr/local/miniconda3/bin/conda create -p /hy-tmp/vllm-build -c conda-forge python=3.11`（`CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes` 规避 Anaconda ToS） |
| torch | 复用系统（`site-packages/_system_site.pth` 指向 `/usr/local/lib/python3.11/dist-packages`）→ torch 2.13.0+cu130，免重复占盘 |
| 构建依赖 | `requirements/common.txt` + `requirements/build/cuda.txt` + `requirements/build/rust.txt` |
| Rust | `/root/.cargo`（cargo/rustc 1.98.1，rustup minimal） |
| 源码构建 | `cd /hy-tmp/vllm/vllm && VLLM_USE_PRECOMPILED=1 pip install -e . --no-build-isolation --no-deps` → **`vllm-0.28.1rc1.dev453+ga1541f574...precompiled`**，编译成功 |
| 运行验证 | 用该构建加载 Qwen2.5-1.5B 生成成功（`' Paris. The capital of France is...'`，EXITCODE=0） |

**运行注意事项**（conda 与系统 libstdc++ ABI 差异）：
```bash
export LD_LIBRARY_PATH=/hy-tmp/vllm-build/lib:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
/hy-tmp/vllm-build/bin/python your_script.py
```
若不设 `LD_LIBRARY_PATH`：`ImportError: ... libicui18n.so.78 ... CXXABI_1.3.15 not found`。

## 5. SGLang 环境（已验证通过，需换后端）

- `/root/sglang-venv`（virtualenv + system site packages）：**sglang 0.5.19 安装成功**。
- 默认后端（flashinfer）失败：`RuntimeError: FlashInfer requires GPUs with sm75 or higher`（本机 FlashInfer 在 sm_120 + CUDA 12.8 下不可用）。
- **可用配置 = 换 triton 后端**：
  ```python
  from sglang import Engine
  e = Engine(model_path="/models/qwen2.5-1.5b-instruct", dtype="bfloat16",
             attention_backend="triton", mem_fraction_static=0.6, disable_cuda_graph=True)
  ```
  实测：12s 就绪，生成正常（`' Paris. The capital of France is...'`），EXITCODE=0。
- 可用后端清单（`sglang.srt.layers.attention.attention_registry.ATTENTION_BACKENDS`）含 `triton / torch_native / flex_attention / fa3 / fa4` 等，sm_120 下优先试 `triton`。
- 注意：`Engine` 真实参数名为 `model_path`（`sglang.Engine` 是懒加载代理）。

## 6. 运维记录（重要）

- **已删除**：`/vllm-build`（旧 conda 构建环境，11GB，非权重；已按上文重建）；`/models/qwen3-8b-awq`（本项目会话中从 ModelScope 下载的模型，可重下）。
- **未删除**：`/hy-tmp` 下 30B FP8 权重、OLMoE gguf、vLLM 源码、`/models/qwen2.5-1.5b-instruct` 全部完好。
- **磁盘腾挪来源**：删除 `/vllm-build`（11GB，构建残留 conda 环境）后根盘由 1.2GB → 12GB 可用。
- **教训**：操作前应检查全部挂载点（含 `/hy-tmp`）与既有资源，不只看 `/models` 和根盘。

## 7. 模型获取方式（复现用）

HuggingFace 不可达；用 ModelScope：
```python
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/models/qwen2.5-1.5b-instruct")
snapshot_download("Qwen/Qwen3-8B-AWQ", local_dir="/models/qwen3-8b-awq")
```

## 8. 运维事件记录（2026-09-10）

### 8.1 系统 vLLM 0.28.0 曾被源码构建误删（已恢复）
- 现象：`/usr/local/lib/python3.11/dist-packages/vllm` 消失，系统 `python3` 报 `No module named 'vllm'`。
- 根因：源码构建使用 `--system-site-packages` 的 venv，`pip install -e .` 通过系统路径看到同名 `vllm 0.28.0`，安装 dev 版本时卸载了系统 vllm。
- 恢复：`/usr/local/bin/python3 -m pip install "vllm==0.28.0" --no-cache-dir` → 成功，导入验证 `0.28.0` 正常。
- 教训：`--system-site-packages` 环境里安装与系统同名的包会误删系统安装；构建 vLLM 应用**独立（非 system-site-packages）环境**或显式 `--ignore-installed`。

### 8.2 FP8 KV cache 在本环境不可用
- 尝试：`--kv-cache-dtype fp8 / fp8_e4m3`（run_eval 已加 `--kv-cache-dtype` 参数）。
- 结果：引擎启动失败 → `RuntimeError: FlashInfer backend is not available`。
- 原因：vLLM 0.28 的 fp8 KV 路径实现在 **FlashInfer 后端**（`flashinfer.py` 有 fp8 KV 代码；`flash_attn.py` 无）；而本机 FlashInfer 在 sm_120 + CUDA 12.8 下不可用（需 CUDA ≥ 12.9）。
- 结论：**在本机栈上，原生 FP8 KV 量化无法启用**；如需启用须安装/编译匹配 CUDA 13 的 FlashInfer（高风险环境改动，暂不做）。
