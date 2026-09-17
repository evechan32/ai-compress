# 在旧驱动（CUDA 12.1）+ sm_86 上跑 vLLM —— 环境配方（2026-09-17 实测）

## 结论速查

| 项 | 值 |
|---|---|
| 机器 | RTX 3090 24G，**sm_86**，驱动 **530.30.02（CUDA 12.1 上限）** |
| 能跑的最新 vLLM | **0.11.2**（torch 2.9.0+cu128）✅ 实测加载模型并生成成功 |
| 跑不了的 | **vLLM 0.28 / 0.29** —— 其依赖全为 `cu13`（CUDA 13.4），驱动 530 无解 |
| 关键原理 | **CUDA 12.x 小版本兼容**：cu128 运行时可在 12.1 驱动上工作（实测 GPU 计算正确） |
| 编译 | ❌ 不是出路：vLLM 0.28 源码本身硬依赖 cu13；且容器无 nvcc（但 `/usr/local/cuda-12.1/bin/nvcc` 12.1 其实存在） |

## 隐藏的磁盘配额（重要陷阱）

`df` 报告 `/hy-tmp` 有 26G 空闲，但 **`dd` 实测只能写 ~4.4GB 就 ENOSPC**。
容器有隐藏配额（总预算约 30G，被模型 18G + 环境吃掉后只剩 ~4G）。
→ **装大包前先 `dd` 实测可写空间**，不要信 `df`。

## 安装配方（隔离 target + PYTHONPATH，不动系统 torch）

```bash
# 1) torch（cu128，能跑在驱动 530 上）
export TMPDIR=/hy-tmp/tmp        # 关键：pip 临时目录指到大盘
python3 -m pip install --target=/hy-tmp/t29 --no-cache-dir \
  -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.9.0

# 2) vLLM 本体（先 --no-deps 省空间，再补缺）
python3 -m pip install --target=/hy-tmp/t29 --no-cache-dir --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple vllm==0.11.2

# 3) 钉住版本敏感的包（用 --no-deps 装会版本错乱）
#    transformers 必须 <5（vLLM 0.11.2 声明 transformers<5,>=4.56）
rm -rf /hy-tmp/t29/transformers* /hy-tmp/t29/tokenizers*
python3 -m pip install --target=/hy-tmp/t29 --no-cache-dir -q \
  -i https://pypi.tuna.tsinghua.edu.cn/simple "transformers>=4.56,<5"
#    torchvision 必须与 torch 2.9 配套，否则 `operator torchvision::nms does not exist`
rm -rf /hy-tmp/t29/torchvision*
python3 -m pip install --target=/hy-tmp/t29 --no-cache-dir --no-deps -q \
  -i https://pypi.tuna.tsinghua.edu.cn/simple "torchvision==0.24.0"
#    pydantic / pydantic-core 必须配套（否则 SystemError），
#    且用 --target 升级不会清掉旧 dist-info，需先 rm -rf 再装
rm -rf /hy-tmp/t29/pydantic* 
python3 -m pip install --target=/hy-tmp/t29 --no-cache-dir -q \
  -i https://pypi.tuna.tsinghua.edu.cn/simple pydantic

# 4) 循环补缺模块（每次报 No module named X 就 --no-deps 装 X）
#    实测需要：cloudpickle msgspec regex safetensors aiohttp multidict yarl
#    frozenlist cachetools openai httpx2 openai_harmony starlette annotated_doc
#    diskcache cbor2 numba llvmlite
```

## 运行

```bash
PYTHONPATH=/hy-tmp/t29 VLLM_USE_FLASHINFER_SAMPLER=0 python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='/hy-tmp/models/Qwen2.5-1.5B-Instruct', max_model_len=2048,
          gpu_memory_utilization=0.6, enforce_eager=True, disable_log_stats=True)
print(llm.generate(['The capital of France is'], SamplingParams(max_tokens=16, temperature=0))[0].outputs[0].text)
"
```
实测输出：`' Paris. The capital of France is also the capital of which country?'`

## 对我们插件（0.28 目标）的移植可行性

| 插件依赖的 API | 在 0.11.2 |
|---|---|
| `KVCacheSpec` | ✅ 36 文件 |
| `FullAttentionManager` | ✅ 4 文件 |
| `remove_skipped_blocks`（核心钩子） | ✅ 6 文件 |
| `TritonAttentionBackend` | ✅ 4 文件 |
| `ModelRunnerOutput`（TP 回传） | ✅ 32 文件 |
| **`rswa_prefix_lens`**（我们复用的 R-SWA 机制） | ❌ **0 文件** → 需改用 0.11.2 自己的滑动窗口/skip-block 机制重写 |

⇒ **移植主体可行，主要工作是把"复用 R-SWA 掩码"换成 0.11.2 原生的 skip-block 语义。**

## 教训

1. **不要相信 `df`** —— 先 `dd` 实测。
2. **vLLM 版本的真正门槛是它链接的 CUDA 运行时**，不是"新旧"：0.28 要 CUDA 13 就无解，0.11.2 的 cu128 反而能跑。
3. **`--no-deps` + 循环补缺**是在受限磁盘上装 vLLM 的可行路径，但**版本敏感包（transformers/torchvision/pydantic-core）必须显式钉版**，且 `--target` 升级要先 `rm -rf` 旧文件。
