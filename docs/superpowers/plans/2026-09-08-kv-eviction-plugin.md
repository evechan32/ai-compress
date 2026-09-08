# ai-compress v1（连续区间 KV 驱逐插件）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个零 fork 的 vLLM 0.28 插件包：标准 MHA/GQA 模型（首目标 Qwen2.5）在长多轮对话中启用 RSWA 式 KV 驱逐，KV 显存有界且质量可评估。

**Architecture:** 插件通过 `vllm.general_plugins` 入口点在全部进程注入；模型适配层让目标架构产出 RSWA 式受限窗口注意力（复用 vLLM 原生 RSWA spec/manager/注意力 mask，不写 kernel）；配置走 `AI_COMPRESS_*` 环境变量。Task 1 spike 决定适配层是"config 注入"还是"模型包装"。

**Tech Stack:** Python 3.11、vLLM 0.28.0（服务器已装）、torch 2.13 cu13、pytest、rsync（本地→服务器同步）。

## Global Constraints

- vLLM 版本钉死 0.28.0；不改动 `/usr/local/lib/python3.11/dist-packages/vllm/` 任何文件。
- 服务器环境：Ubuntu 22.04、2×RTX 5070(sm_120)、模型 `/models/qwen2.5-1.5b-instruct`。
- 所有 vLLM 运行必须设 `VLLM_USE_FLASHINFER_SAMPLER=0`。
- 服务器磁盘 <8GB：不新增重依赖，评测数据自生成。
- 代码在本地仓库 `/home/qyw/projects/ai-compress` 编写与提交；`rsync` 同步到服务器 `/root/ai-compress` 后测试/运行。
- 纯 Python 模块（config/policy/bench 数据生成）不得在 import 链上依赖 vllm/torch，保证可本地单测。
- `AI_COMPRESS_ENABLE=0`（默认）时插件行为与原生 vLLM 完全一致（旁路零副作用）。
- 提交信息遵循仓库现有风格（`feat:`/`docs:`/`test:`/`fix:`）。

## File Structure

```
ai-compress/
├── pyproject.toml            # 包 + entry point
├── kvcompress/
│   ├── __init__.py           # 插件入口（env 设置 + 装载）
│   ├── config.py             # AI_COMPRESS_* 解析 + 校验（纯 python）
│   ├── policy.py             # 驱逐区间策略（纯 python）
│   └── adapter.py            # 目标架构 → RSWA 受限注意力（形态由 Task 1 决定）
├── bench/
│   ├── gen_data.py           # 自生成评测数据（纯 python）
│   ├── run_eval.py           # 基线/插件统一评测入口
│   └── report.py             # 汇总输出
├── tests/
│   ├── test_config.py        # 本地跑
│   ├── test_policy.py        # 本地跑
│   └── test_bypass.py        # 服务器跑（需 vllm）
└── docs/
    ├── survey-kvcache-papers.md
    ├── survey-vllm-integration.md
    ├── spike-r1-findings.md          # Task 1 产出
    └── superpowers/specs/2026-09-08-kv-eviction-plugin-design.md
```

---

### Task 1: R1 Spike——标准模型能否走 RSWA 注意力路径（服务器）

**Files:**
- Create: `docs/spike-r1-findings.md`（结论记录，后续任务读取）
- Create: `/root/ai-compress/spike/r1_force_rswa.py`（服务器侧临时脚本，不入库）

**Interfaces:**
- Produces: `docs/spike-r1-findings.md`，必须明确记录三件事：
  1. 让 Qwen2.5-1.5B 以 RSWA 语义（rswa_window=512，prompt 全可见、生成段滑窗）加载所需的最小改动点（config 注入 or 模型代码路径）。
  2. 上述模式下 decode 输出与原生基线是否一致（窗口内）或偏差可解释（窗口外被正确忽略）。
  3. KV 有界性验证：多轮生成长度超过窗口后，每请求 KV 块数是否停止增长。
  4. 结论：适配层走 **config 注入（推荐，改动最小）** 还是 **模型包装**；以及 v0.28 中 RSWA 需要的后端可用性（FA4 rswa_mask_mod / 回退路径）。

- [ ] **Step 1: 同步仓库到服务器并准备实验**

Run (local):
```bash
rsync -az --exclude .git /home/qyw/projects/ai-compress/ root@i-2.gpushare.com:/root/ai-compress/
```
（ssh 端口 29196、密码由执行者提供；可用 `sshpass -p "$PWD" rsync -e "ssh -p 29196 -o StrictHostKeyChecking=no" ...`）

- [ ] **Step 2: 阅读 RSWA 原生实现，确定注入面**

Run (server):
```bash
V=/usr/local/lib/python3.11/dist-packages/vllm
sed -n '1,120p' $V/model_executor/models/qwen3_next.py    # 原生 RSWA 模型如何声明 spec/窗口
grep -n "RSWASpec\|rswa" $V/model_executor/models/qwen3_next.py | head -20
sed -n '1460,1470p' $V/config/model.py                    # model_config.rswa_window 来源
```
记录：RSWASpec 由哪个模型方法产出、窗口如何从 config 传播到 attention backend。

- [ ] **Step 3: 实验 A——config 注入（改副本 config.json）**

Run (server):
```bash
mkdir -p /models/qwen2.5-1.5b-rswa && cp -r /models/qwen2.5-1.5b-instruct/* /models/qwen2.5-1.5b-rswa/
python3 - <<'PY'
import json
p="/models/qwen2.5-1.5b-rswa/config.json"
c=json.load(open(p)); c["rswa_window"]=512
json.dump(c, open(p,"w"), indent=2); print("patched")
PY
```
写 `/root/ai-compress/spike/r1_force_rswa.py`：加载副本模型，打印引擎日志中 KV cache spec 类型与 `GPU KV cache size`，并用多轮对话脚本验证（3 轮各 256 token，观察第 3 轮起 KV 块数是否≈有界）。
Expected: 若 config 注入生效，日志出现 RSWA 相关 spec/manager 且第 3 轮 KV 不再线性增长；若 Qwen2 架构不读 rswa_window，输出仍 FullAttentionSpec——记录此结论。

- [ ] **Step 4: 实验 B——若 A 失败，走架构级强制**

在实验脚本中用 `kvcompress.adapter` 的雏形（monkey-patch 该架构 attention spec 产出函数返回 RSWASpec）重试，并检查后端选择（`rswa_mask_mod` 可用性、是否自动回退 TRITON_ATTN）。
Run: 同 Step 3 的验证脚本。
Expected: 明确记录"模型包装可行/不可行 + 需要的最小改动点"。

- [ ] **Step 5: 正确性对照**

同一长 prompt + 多轮续写：基线（原生模型，全注意力）vs 实验 A/B 成功路径。比较前 `rswa_window` 步内的逐 token 输出是否一致；窗口外老 token 被正确忽略时输出应有语义差异但无崩溃/乱码。
Expected: 在 findings 中记录对照结论（一致/可解释偏差/异常）。

- [ ] **Step 6: 写 findings + 提交**

把四件事的结论写入 `docs/spike-r1-findings.md`，明确给出 Task 5 的适配层形态选择与后端约束。本地 commit：
```bash
git add docs/spike-r1-findings.md
git commit -m "docs: R1 spike 结论——标准模型 RSWA 路径可行性"
```

---

### Task 2: 包骨架 + 插件入口 + config 模块（本地可测）

**Files:**
- Create: `pyproject.toml`
- Create: `kvcompress/__init__.py`
- Create: `kvcompress/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `kvcompress.config.CompressConfig` dataclass + `kvcompress.config.load_config() -> CompressConfig`；`kvcompress/__init__.py` 暴露 `entrypoint()`。
- Later tasks consume: `cfg.enabled`, `cfg.policy` (`"rswa"|"sink_window"`), `cfg.rswa_window`, `cfg.sink_len`, `cfg.target_archs`, `cfg.policy_obj`。

- [ ] **Step 1: 写失败测试** `tests/test_config.py`

```python
import os
from kvcompress.config import load_config

def _clear():
    for k in list(os.environ):
        if k.startswith("AI_COMPRESS_"):
            del os.environ[k]

def test_disabled_by_default():
    _clear()
    cfg = load_config()
    assert cfg.enabled is False

def test_enable_and_window():
    _clear()
    os.environ["AI_COMPRESS_ENABLE"] = "1"
    os.environ["AI_COMPRESS_RSWA_WINDOW"] = "513"   # 非 16 对齐 → 向上对齐
    cfg = load_config()
    assert cfg.enabled is True
    assert cfg.rswa_window == 528   # ceil(513/16)*16

def test_unknown_policy_raises():
    _clear()
    os.environ["AI_COMPRESS_ENABLE"] = "1"
    os.environ["AI_COMPRESS_POLICY"] = "bogus"
    try:
        load_config()
        assert False, "should raise"
    except ValueError as e:
        assert "policy" in str(e)
```

- [ ] **Step 2: 运行确认失败**
Run (local): `python -m pytest tests/test_config.py -v`
Expected: FAIL（ImportError: No module named 'kvcompress'）

- [ ] **Step 3: 写实现**

`pyproject.toml`：
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "kvcompress"
version = "0.1.0"
description = "Continuous-range KV cache eviction plugin for vLLM (zero-fork)"
requires-python = ">=3.11"
dependencies = []  # vllm 由宿主环境提供

[project.entry-points."vllm.general_plugins"]
kvcompress = "kvcompress:entrypoint"

[tool.setuptools]
packages = ["kvcompress"]
```

`kvcompress/config.py`：
```python
"""AI_COMPRESS_* 配置解析。纯 python，禁止 import vllm/torch。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

BLOCK_SIZE = 16  # vLLM 默认 block_size（向上对齐用）

_POLICIES = ("rswa", "sink_window")


def _aligned(n: int, block: int = BLOCK_SIZE) -> int:
    return ((n + block - 1) // block) * block


@dataclass(frozen=True)
class CompressConfig:
    enabled: bool = False
    policy: str = "rswa"
    rswa_window: int = 1024
    sink_len: int = 64
    target_archs: tuple[str, ...] = ("Qwen2ForCausalLM",)

    @property
    def policy_obj(self):
        from kvcompress.policy import RSWAPolicy, SinkWindowPolicy  # noqa: 延迟避免循环
        if self.policy == "rswa":
            return RSWAPolicy(window=self.rswa_window)
        return SinkWindowPolicy(window=self.rswa_window, sink_len=self.sink_len)


def load_config() -> CompressConfig:
    if os.environ.get("AI_COMPRESS_ENABLE", "0") not in ("1", "true", "True"):
        return CompressConfig(enabled=False)

    policy = os.environ.get("AI_COMPRESS_POLICY", "rswa")
    if policy not in _POLICIES:
        raise ValueError(f"AI_COMPRESS_POLICY must be one of {_POLICIES}, got {policy!r}")

    rswa_window = int(os.environ.get("AI_COMPRESS_RSWA_WINDOW", "1024"))
    if rswa_window < BLOCK_SIZE:
        raise ValueError(f"AI_COMPRESS_RSWA_WINDOW must be >= {BLOCK_SIZE}")
    rswa_window = _aligned(rswa_window)

    sink_len = int(os.environ.get("AI_COMPRESS_SINK_LEN", "64"))
    if policy == "sink_window" and sink_len < BLOCK_SIZE:
        raise ValueError(f"AI_COMPRESS_SINK_LEN must be >= {BLOCK_SIZE}")
    sink_len = _aligned(sink_len)

    archs = tuple(
        a.strip()
        for a in os.environ.get(
            "AI_COMPRESS_TARGET_ARCHS", "Qwen2ForCausalLM"
        ).split(",")
        if a.strip()
    )
    if not archs:
        raise ValueError("AI_COMPRESS_TARGET_ARCHS is empty")
    return CompressConfig(
        enabled=True, policy=policy, rswa_window=rswa_window,
        sink_len=sink_len, target_archs=archs,
    )
```

`kvcompress/__init__.py`：
```python
"""kvcompress: vLLM 连续区间 KV 驱逐插件入口。"""
from __future__ import annotations

_ENV_SETS = {
    # sm_120 (RTX 50xx) 上 flashinfer JIT 采样需 CUDA>=12.9；统一走 vLLM 采样器
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


def _apply_env_defaults() -> None:
    import os
    for k, v in _ENV_SETS.items():
        os.environ.setdefault(k, v)


def entrypoint():
    """vllm.general_plugins 入口：所有进程加载时执行。"""
    _apply_env_defaults()
    from kvcompress import config
    cfg = config.load_config()
    if not cfg.enabled:
        return  # 旁路：零副作用
    from kvcompress import adapter  # noqa: 仅在启用时导入，避免 vllm 耦合提前加载
    adapter.install(cfg)
```

- [ ] **Step 4: 运行确认通过**
Run (local): `python -m pytest tests/test_config.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml kvcompress/ tests/test_config.py
git commit -m "feat: kvcompress 包骨架、插件入口与配置模块"
```

---

### Task 3: policy 模块（本地可测）

**Files:**
- Create: `kvcompress/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: 无（纯逻辑）。
- Produces: `EvictionPolicy` ABC（`retain_ranges(num_prefix: int, num_generated: int) -> list[tuple[int,int]]`，返回 [start,end) token 区间，0=首个 token）；`RSWAPolicy(window:int)`——`[(0, num_prefix), (num_prefix + max(0, num_generated-window), num_prefix+num_generated)]` 若 num_generated>window 则两段间存在 gap；`SinkWindowPolicy(window:int, sink_len:int)`——`[(0,sink_len), (total-window, total)]`（total=num_prefix+num_generated，仅当 total>sink_len+window 时有 gap）。

- [ ] **Step 1: 写失败测试** `tests/test_policy.py`

```python
from kvcompress.policy import RSWAPolicy, SinkWindowPolicy

def test_rswa_prefix_always_kept():
    p = RSWAPolicy(window=200)
    # prefix=1000, generated=100 -> 生成全部在窗口内 -> 无 gap
    assert p.retain_ranges(1000, 100) == [(0, 1100)]

def test_rswa_gap_appears_when_generation_exceeds_window():
    p = RSWAPolicy(window=64)
    # prefix=100, generated=200 -> 保留 [0,100) + 末尾 64 个生成 token [236,300)
    ranges = p.retain_ranges(100, 200)
    assert ranges == [(0, 100), (236, 300)]  # 生成段前 136 token 被驱逐
    kept = sum(e - s for s, e in ranges)
    assert kept == 100 + 64

def test_sink_window_evicts_middle_of_total():
    p = SinkWindowPolicy(window=64, sink_len=16)
    ranges = p.retain_ranges(1000, 50)       # total=1050 > 16+64
    assert ranges == [(0, 16), (1050 - 64, 1050)]
```

- [ ] **Step 2: 运行确认失败**
Run (local): `python -m pytest tests/test_policy.py -v`
Expected: FAIL（ImportError）

- [ ] **Step 3: 写实现** `kvcompress/policy.py`

```python
"""驱逐策略：给定已计算 token 计数，返回应保留的连续 [start,end) token 区间。"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EvictionPolicy(ABC):
    @abstractmethod
    def retain_ranges(self, num_prefix: int, num_generated: int) -> list[tuple[int, int]]:
        """num_prefix=prompt(prefill) token 数；num_generated=已生成 token 数。"""


class RSWAPolicy(EvictionPolicy):
    """prompt 全保留；生成段只保留末尾 window。语义对齐 vLLM RSWASpec。"""

    def __init__(self, window: int):
        self.window = window

    def retain_ranges(self, num_prefix: int, num_generated: int) -> list[tuple[int, int]]:
        if num_generated <= self.window:
            return [(0, num_prefix + num_generated)]
        gen_start = num_prefix + (num_generated - self.window)
        return [(0, num_prefix), (gen_start, num_prefix + num_generated)]


class SinkWindowPolicy(EvictionPolicy):
    """整段上下文只保留开头 sink 与末尾 window（StreamingLLM 式，v1.x 默认关闭）。"""

    def __init__(self, window: int, sink_len: int):
        self.window = window
        self.sink_len = sink_len

    def retain_ranges(self, num_prefix: int, num_generated: int) -> list[tuple[int, int]]:
        total = num_prefix + num_generated
        if total <= self.sink_len + self.window:
            return [(0, total)]
        return [(0, self.sink_len), (total - self.window, total)]
```

- [ ] **Step 4: 运行确认通过**
Run (local): `python -m pytest tests/test_policy.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: Commit**
```bash
git add kvcompress/policy.py tests/test_policy.py
git commit -m "feat: 驱逐策略模块（RSWA / SinkWindow）"
```

---

### Task 4: adapter 安装机制（骨架，形态由 Task 1 决定）

**Files:**
- Create: `kvcompress/adapter.py`
- Test: `tests/test_bypass.py`（服务器，需 vllm）

**Interfaces:**
- Consumes: `cfg`（Task 2）、findings（Task 1）。
- Produces: `adapter.install(cfg)`——在启用时把 `cfg.target_archs` 中架构的注意力路径切到 RSWA 受限语义；并实现 `adapter._patch_config_or_model(...)`。

- [ ] **Step 1: 依据 findings 写 adapter 骨架**

先读 `docs/spike-r1-findings.md`，按其结论实现两种形态之一：
- 形态 C（config 注入可用）：`install()` 校验模型副本 config 携带 rswa_window（加载侧在 Task 5 处理），此处仅占位日志 + 断言 cfg 合法。
- 形态 M（需模型包装）：`install()` 对 `cfg.target_archs` 注册一个模型适配（monkey-patch 其 `get_kv_cache_spec`/attention 构建路径，具体改点以 findings 记录为准）。

本 Task 落一个**可运行、可测试**的 `install()`：无论形态，先实现"启用后对未知架构拒绝 + 已知架构打印注入日志"，把真实注入逻辑留在 Task 5（依赖 Task 1 结论文件的具体 seam 描述）。代码以 findings 结论为准，不得凭空假设。

- [ ] **Step 2: 旁路一致性测试** `tests/test_bypass.py`（服务器运行）

```python
"""服务器集成测试：AI_COMPRESS_ENABLE=0 时 kvcompress 不影响 vLLM 输出。"""
import os, subprocess, sys, textwrap

REPO = "/root/ai-compress"
MODEL = "/models/qwen2.5-1.5b-instruct"


def _run_once(extra_env: dict) -> str:
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {REPO!r})
        from vllm import LLM, SamplingParams
        llm = LLM(model={MODEL!r}, dtype="bfloat16", max_model_len=4096,
                  gpu_memory_utilization=0.85, enforce_eager=True)
        out = llm.generate(["Repeat exactly: KV-COMPRESS-BYPASS-42"],
                           SamplingParams(max_tokens=24, temperature=0.0))
        print(out[0].outputs[0].text)
    """)
    env = dict(os.environ, **extra_env)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=600, cwd=REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


def test_bypass_identical_to_native():
    # 插件已装（pip install -e .）时：关闭态输出 == 原生输出
    native = _run_once({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    with_plugin = _run_once({"VLLM_USE_FLASHINFER_SAMPLER": "0",
                             "AI_COMPRESS_ENABLE": "0"})
    assert native == with_plugin
```

- [ ] **Step 3: 服务器安装插件并跑测试**
Run (server):
```bash
cd /root/ai-compress && pip install -e . --no-deps --no-cache-dir
VLLM_USE_FLASHINFER_SAMPLER=0 python -m pytest tests/test_bypass.py -v
```
Expected: PASS（旁路输出与原生一致）。若 entry point 未被 vLLM 识别，检查 `pip show kvcompress` 的 entry_points 与 vLLM `load_plugins_by_group("vllm.general_plugins")` 输出。

- [ ] **Step 4: Commit**
```bash
git add kvcompress/adapter.py tests/test_bypass.py
git commit -m "feat: adapter 安装机制与旁路一致性测试"
```

---

### Task 5: Qwen2 适配层（真实注入，依据 Task 1 findings）

**Files:**
- Modify: `kvcompress/adapter.py`

**Interfaces:**
- Consumes: Task 1 findings（注入 seam 与后端约束）、Task 4 骨架。
- Produces: 启用后 Qwen2 模型以 RSWA 语义运行。

- [ ] **Step 1: 实现 findings 选定的形态**

- 形态 C：适配层在模型加载早期读取/校验副本 config（rswa_window 已由用户用配置脚本注入），adapter 只负责：
  - 校验该架构 config 是否有 `rswa_window` 且 ≥ cfg 值；
  - 打印注入确认日志（含 window/架构）；
  - 若 config 缺失则抛错并给出注入命令（不静默跑全注意力）。
- 形态 M：按 findings 的具体 seam（函数路径 + 改法）monkey-patch；含后端回退检查（FA4 rswa_mask_mod 不可用时改选允许的 backend 或显式报错）。
- 无论形态：`cfg.policy == "sink_window"` 时 v1 直接报"not implemented in v1"（按 spec，sink 模式为 v1.x），保证不静默错误。

- [ ] **Step 2: 启用态 smoke（服务器）**

Run (server)：多轮对话脚本——首轮 prompt 800 token + 3 轮 × 300 token 生成。
Expected: 引擎日志出现 RSWA 相关 spec；第 3 轮起该请求 KV 块数不再随轮次线性增长（用日志 `GPU KV cache size`/块数或 nvidia-smi 佐证）；输出无崩溃/乱码。

- [ ] **Step 3: 写启用态回归检查（质量 sanity）**

把启用态对"窗口内近期事实"的回答与基线对比（needle 在最近窗口内应仍能答出）。记录到 `docs/spike-r1-findings.md` 附录或 bench 报告。

- [ ] **Step 4: Commit**
```bash
git add kvcompress/adapter.py docs/spike-r1-findings.md
git commit -m "feat: Qwen2 RSWA 适配层（注入 + 校验 + 日志）"
```

---

### Task 6: 评测闭环（M2/M3 验收）

**Files:**
- Create: `bench/gen_data.py`
- Create: `bench/run_eval.py`
- Create: `bench/report.py`

**Interfaces:**
- Consumes: 插件启用态（Task 5）。
- Produces: `bench/out/{baseline,plugin}/metrics.json` + 汇总报告。

- [ ] **Step 1: 数据生成** `bench/gen_data.py`

纯 python，输出三份：
1. 多轮长对话合成（确定性 seed）：N 轮，每轮注入 1-2 条"事实"，总长可控（~8k-32k token）。
2. needle-in-haystack：随机文档 + 在指定深度插入事实句 + 提问。
3. 单轮超长 prompt 一致性（可选，受 12GB 限制）。
提供 `--out-dir`、`--max-tokens-per-scene`。

- [ ] **Step 2: 评测入口** `bench/run_eval.py`

```bash
# 基线（原生 vLLM）
python bench/run_eval.py --model /models/qwen2.5-1.5b-instruct \
    --data bench/out --out bench/out/baseline --tag baseline
# 插件（启用 RSWA）
AI_COMPRESS_ENABLE=1 AI_COMPRESS_RSWA_WINDOW=512 \
python bench/run_eval.py --model /models/qwen2.5-1.5b-instruct \
    --data bench/out --out bench/out/plugin-w512 --tag plugin-w512
```
对每个 scene：跑多轮/needle 提问，收集——答案文本、逐 token 时间戳、`/proc` 或引擎日志 KV 块数。脚本须 `VLLM_USE_FLASHINFER_SAMPLER=0`，temperature=0。

- [ ] **Step 3: 汇总报告** `bench/report.py`

输出：质量（needle 召回率按深度桶 / 对话轮次一致性）、峰值 GPU 显存、TTFT/TPOT、吞吐、估算每请求 KV 块数（基线 vs 各窗口）。表格输出 + JSON。

- [ ] **Step 4: 跑通 M2/M3 并提交报告**
Run (server)：基线 + 至少 w512/w1024 两组。
Expected: 报告显示启用后 KV 显存/块数随对话有界，质量随窗口增大趋近基线；把 `bench/out/` 摘要写 `docs/bench-report-2026-09-08.md`。
```bash
git add bench/ docs/bench-report-2026-09-08.md
git commit -m "feat: 评测闭环与 v1 基准报告"
```

---

## 收尾

- M1 旁路验证 = Task 4 Step 2 测试通过。
- M2 RSWA 驱逐打通 = Task 5 Step 2/3 通过。
- M3 评测闭环 = Task 6 产出对比报告。
- 非目标复查：未做散点驱逐、KV 量化、Sink+Window（v1.x）——均按 spec 第 10 节标注。
