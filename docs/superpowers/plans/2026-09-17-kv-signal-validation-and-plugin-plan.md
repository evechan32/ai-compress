# KV 压缩：信号验证 + 插件落地 计划（3090 24G / sm_86）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 sm_86（3090 24G）上用 **≤35 GPU-小时** 判定"哪种重要性信号能把 KV 压到 ≥2× 且质量损失 ≤1%"，并把赢家移植进现有零 fork vLLM 插件（`kvcompress/pevict.py`），形成一个可复现、可发表的结论（正结果或**有证据的负结果**）。

**Architecture:** 分三层验证，**成本从低到高，每层设 kill gate**：
1. **信号层（HF + kvpress，便宜）** —— 信号好坏在这里判定。已有 `bench/kvpress_eval.py` / `bench/kvpress_pipeline_eval.py`，无需碰 vLLM。
2. **插件层（vLLM）** —— 只移植信号层的赢家；释放机制（块回收 + `block_table` 压实）**已冻结不动**。
3. **端到端层** —— 质量（F1 + 配对 bootstrap CI）+ 系统（容量/吞吐）+ 正确性（NaN 毒化证明）。

**Tech Stack:** vLLM 0.28（零 fork 插件 `kvcompress`）、NVIDIA kvpress 0.5.4（隔离在 `/root/kvpress-libs`）、transformers 5.2.0、Qwen2.5-1.5B/7B、LongBench 5 任务、自研 NIAH harness。

## Global Constraints

- **平台**：测试机为 **3090 24G，sm_86（Ampere）**。禁止与 sm_120（RTX 5070）的历史数字混表；所有新结论必须标注 sm_86。
- **零 fork 约束保留**：`kvcompress` 不得修改 vLLM 源码。
- **不可删除**：既有模型权重、`/hy-tmp` 内容。
- **git**：每条 git 命令前缀 `GIT_MASTER=1`；原子提交；不提交机密（密码不得落盘到被追踪文件）。
- **服务器运行前置**：`export LD_LIBRARY_PATH=<cu13 lib>:$LD_LIBRARY_PATH`（torch 导入需要）；`VLLM_USE_FLASHINFER_SAMPLER=0`（sm_120 遗留；**sm_86 上需重新验证该设置是否仍然必要**）。
- **Qwen2.5 模型常量**：1.5B = 28 层 / 2 KV 头 / hidden 1536 / vocab 151936；7B = 28 层 / 4 KV 头 / hidden 3584 / vocab 151936。
- **统计口径**：任何质量结论必须带 **配对 bootstrap 95% CI**（同 prompt、deterministic 解码）；只报均值差视为未完成。
- **成本口径**：一切以 **GPU-小时** 计，每阶段结束报实际消耗。

---

## 0. 目标与判据（先定判据，再花钱）

### 成功判据（按优先级）

| 代号 | 判据 | 度量 |
|---|---|---|
| **T1（主）** | ≥**2×** KV 压缩（相对不驱逐的 KV 容量）且 **≤1% 相对质量损失** | LongBench F1 配对 CI 下界 ≥ −0.01；7B |
| **T2** | 该信号额外 prefill 开销 **≤1.2×** | prefill 墙钟比 |
| **T3** | 容量受限 decode 负载下吞吐 **≥2×** | 同 SLA 下 tok/s |

**为什么是 2×/1%**：现成对照组 —— FP8 KV 是 2× 但 F1 掉 **−0.04**（我们实测）。**2× 且 ≤1% 就能明确胜过量化**，这是能被认可的门槛。

### Kill 判据（省钱的关键）

| Gate | 位置 | 条件 | 不通过则 |
|---|---|---|---|
| **G0** | Stage 0 末 | sm_86 上 FlashInfer 采样 / FA2 / marlin(AWQ) / Triton 至少 3/4 可用 | 修环境或换 vLLM 版本；**不进入后续阶段** |
| **G1** | Stage 1 末 | 至少一个信号达 **needle rank ≤10** 且 **prefill ≤1.2×** | **放弃驱逐方向**，转 Stage 4-B（量化/卸载）；只写负结果 |
| **G2** | Stage 2 末 | 插件与 HF parity **≤±0.01 F1**，无 pool leak | 不写端到端结论，先修 parity |
| **G3** | Stage 3 末 | T1 达成 | 写负结果 + 转 Stage 4-B |

### 成本预算

| Stage | 内容 | 预估 GPU-h | 最坏（触发 kill） |
|---|---|---|---|
| 0 | 环境 + kernel 冒烟 | 1 | 1 |
| 1 | 信号验证（HF） | 6–10 | 10 |
| 2 | 插件移植 + parity | 8–14 | 14 |
| 3 | 端到端 + 正确性 | 6–10 | 10 |
| **合计** | | **21–35** | **早停 ≈11** |

**关键省钱点**：G1 在 ~11 GPU-h 处就能杀掉整条路线。**Stage 1 不允许碰 vLLM 插件代码。**

---

## 1. 文件结构（锁定边界）

| 文件 | 职责 | 动作 |
|---|---|---|
| `bench/kvpress_eval.py` | kvpress 多方法扫描（HF 路径） | 改：新增 `--model`、`--per-sample` |
| `kvcompress/signals.py` | **新建**：纯函数信号库（无 vLLM 依赖）——**HF 与插件共用的唯一真源** | 建 |
| `bench/surprisal_press.py` | **新建**：kvpress press 包装（import `kvcompress.signals`） | 建 |
| `bench/head_restricted_press.py` | **新建**：检索头限定的第二遍打分 | 建 |
| `bench/p3_longbench_f1.py` | 质量主 harness（已有 `PE_MODEL/PE_TP` 旋钮） | 改：dump 逐样本 + bootstrap |
| `bench/bootstrap_ci.py` | **新建**：配对 bootstrap CI（复用 `p3_pair_eval.py` 逻辑） | 建 |
| `bench/p3_needle_rank.py` | **新建**：needle rank + 自查率探针（整合 `probe_ablation.py`） | 建 |
| `kvcompress/pevict.py` | 插件主文件（信号层 + 释放层） | 改：**只加 `PE_SCORE_MODE`，不动释放路径** |
| `bench/p3_poison.py` | **新建**：NaN 毒化正确性实验 | 建 |
| `tests/test_pevict_parity.py` | **新建**：HF 信号 vs 插件信号的 parity 断言 | 建 |
| `docs/experiments-log.md` | 数据总账 | 改：逐阶段追加 |

---

## Stage 0 — 环境与 kernel 冒烟（1 GPU-h，Gate G0）

**为什么先做**：sm_120 上我们因为 FlashInfer/marlin 缺 kernel 浪费了数天。sm_86 理论上全支持，但**必须实测**，否则后面所有失败都无法归因。

### Task 0.1: 确认 3090 环境与 kernel 可用性

**Files:**
- Create: `bench/smoke_kernels.py`
- Modify: `docs/environment.md`

**Interfaces:**
- Produces: 一份 kernel 可用性矩阵（FlashInfer / FA2 / marlin / Triton），供 G0 判定。

- [ ] **Step 1: 写冒烟脚本**

```python
"""sm_86 kernel 可用性冒烟：逐项 try，失败不抛，最后打印矩阵。"""
import os, sys, traceback
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

results = {}

def check(name, fn):
    try:
        fn(); results[name] = "OK"
    except Exception as e:
        results[name] = f"FAIL: {type(e).__name__}: {str(e)[:120]}"

def _triton_attn():
    import torch
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.environ["PE_MODEL"], max_model_len=1024,
              enforce_eager=True, attention_backend="TRITON_ATTN",
              gpu_memory_utilization=0.85, disable_log_stats=True)
    o = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
    assert len(o[0].outputs[0].text) > 0

def _fa2():
    import torch
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.environ["PE_MODEL"], max_model_len=1024,
              enforce_eager=True, attention_backend="FLASH_ATTN",
              gpu_memory_utilization=0.85, disable_log_stats=True)
    llm.generate(["hi"], SamplingParams(max_tokens=4, temperature=0))

def _marlin():
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.environ["PE_AWQ"], max_model_len=1024,
              enforce_eager=True, gpu_memory_utilization=0.85, disable_log_stats=True)
    llm.generate(["hi"], SamplingParams(max_tokens=4, temperature=0))

def _flashinfer_sampler():
    os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
    from vllm import LLM, SamplingParams
    llm = LLM(model=os.environ["PE_MODEL"], max_model_len=1024,
              enforce_eager=True, gpu_memory_utilization=0.85, disable_log_stats=True)
    llm.generate(["hi"], SamplingParams(max_tokens=4, temperature=0))

if __name__ == "__main__":
    check("triton_attn", _triton_attn)
    check("fa2", _fa2)
    check("marlin_awq", _marlin)
    check("flashinfer_sampler", _flashinfer_sampler)
    for k, v in results.items():
        print(f"[SMOKE] {k}: {v}", flush=True)
    ok = sum(v == "OK" for v in results.values())
    print(f"[SMOKE] {ok}/4 OK", flush=True)
    sys.exit(0 if ok >= 3 else 1)
```

- [ ] **Step 2: 运行并记录**

```bash
export LD_LIBRARY_PATH=/hy-tmp/vllm-build/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
cd /root/ai-compress
PE_MODEL=/models/qwen2.5-1.5b-instruct \
PE_AWQ=/root/models/Qwen2.5-7B-Instruct-AWQ \
  python3 bench/smoke_kernels.py
```
Expected: `[SMOKE] 3/4 OK` 或更好；若 `flashinfer_sampler` OK，则**记录并在后续去掉 `VLLM_USE_FLASHINFER_SAMPLER=0`**。

- [ ] **Step 3: 判定 Gate G0**

- ≥3/4 OK → 进入 Stage 1。
- <3/4 → 先修环境（换 vLLM 版本 / 装 flash-attn 轮子），**不得进入 Stage 1**。

- [ ] **Step 4: 更新 `docs/environment.md` 并提交**

```bash
GIT_MASTER=1 git add bench/smoke_kernels.py docs/environment.md
GIT_MASTER=1 git commit -m "test(smoke): sm_86 kernel 可用性冒烟（G0 门）"
```

### Task 0.2: 准备 7B bf16（24G 的关键红利）

**Files:** Create: `docs/environment.md`（追加）

- [ ] **Step 1: 下载 7B bf16**

```bash
cd /root/models
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download Qwen/Qwen2.5-7B-Instruct --local-dir /root/models/Qwen2.5-7B-Instruct
du -sh /root/models/Qwen2.5-7B-Instruct   # 期望 ~15G
```

- [ ] **Step 2: 冒烟（确认 24G 装得下 7B bf16 + KV）**

```bash
PE_MODEL=/root/models/Qwen2.5-7B-Instruct python3 bench/smoke_kernels.py 2>&1 | tail -6
```
Expected: 至少 `triton_attn: OK`（7B bf16 ≈ 15G 权重，`gpu_memory_utilization=0.85` ≈ 20.4G，KV 约 5G）。

- [ ] **Step 3: 提交**

```bash
GIT_MASTER=1 git add docs/environment.md
GIT_MASTER=1 git commit -m "docs(env): 3090 sm_86 环境档案 + 7B bf16 就位"
```

---

## Stage 1 — 信号层验证（HF，6–10 GPU-h，Gate G1）★ 本计划的成本核心

**铁律：本阶段不允许修改 `kvcompress/pevict.py`。** 全部在 HF + kvpress 里做。

**为什么可行**：`docs/kvpress-longbench.md` 已证明这条路径可用（18 方法横扫跑通），且 7B 只需要 24G 里的一半显存，单配置约 15 分钟。

### Task 1.1: 把 kvpress 扫描参数化到 7B bf16 + 逐样本落盘

**Files:**
- Modify: `bench/kvpress_eval.py`
- Test: `bench/out/`（数据产物）

**Interfaces:**
- Produces: `bench/out/kvpress-<model>-<ratio>.json`，含逐样本 F1，供 Task 1.4 做配对 CI。

- [ ] **Step 1: 加 `--model` 与逐样本落盘**（读现状后做最小改动）

```bash
grep -n "argparse\|add_argument\|json.dump\|--files\|--n\|--ratio" bench/kvpress_eval.py | head -30
```
确认现有参数后，新增：
- `--model`（默认 `/models/qwen2.5-1.5b-instruct`）
- 落盘时每个 (task, method) **保留逐样本 F1 列表**（现在只存 mean）

- [ ] **Step 2: 跑 7B bf16 基线 + 关键对照（先小 n 试跑）**

```bash
PYTHONPATH=/root/kvpress-libs HF_ENDPOINT=https://hf-mirror.com \
python3 bench/kvpress_eval.py \
  --model /root/models/Qwen2.5-7B-Instruct \
  --data /hy-tmp/longbench/data \
  --files qasper.jsonl multifieldqa_en.jsonl \
  --n 10 --ratio 0.5 --tag smoke7b --out /root/kvpress-out \
  --methods none chunkkv compactor
```
Expected: 三个方法都出数、无 OOM、总耗时 < 25 分钟。
**若 OOM**：把 `--model` 换成 4-bit（`Qwen2.5-7B-Instruct-AWQ`）并在结论里标注量化。

- [ ] **Step 3: 提交**

```bash
GIT_MASTER=1 git add bench/kvpress_eval.py
GIT_MASTER=1 git commit -m "test(kvpress): 扫描支持 --model 与逐样本落盘（7B bf16 基线）"
```

### Task 1.2: 惊讶度 press（训练无关，零额外前向）★ 最高性价比候选

**Files:**
- Create: `bench/surprisal_press.py`
- Test: `bench/out/kvpress-*-surprisal.json`

**Interfaces:**
- Consumes: kvpress `ScorerPress` 基类（**先读源码确认签名**）。
- Produces: `SurprisalPress.score(module, hidden_states, keys, values, attentions, kwargs) -> Tensor[b, kv_heads, seq]`，把**逐 token 标量**广播到所有 head。

**为什么值得先做**：我们自己的机制结论（`docs/attention-signal.md`）说第二遍之所以有用，是因为**信息 token 会去核对"不可预测的自己"**（needle 自查率 = filler 的 5–13 倍）。惊讶度 `−log p` 正是"不可预测性"的**零前向**代理 —— 我们论证过它为什么应该成立，却**从没测过**。

- [ ] **Step 1: 读 kvpress press API（不可跳过；不要凭记忆写签名）**

```bash
sed -n '1,120p' /root/kvpress-libs/kvpress/press.py
grep -n "class .*Press\|def score\|def compress" /root/kvpress-libs/kvpress/press.py | head -30
```
记录：`ScorerPress.score` 的确切签名、返回张量形状、`compression_ratio` 如何被消费。

- [ ] **Step 2: 写失败测试（先证明它没被接上）**

```python
# tests/test_surprisal_press.py
def test_surprisal_press_produces_uint_mask():
    """先跑通 import 与 score 形状契约（信号值正确性由 Task 1.3 的 rank 指标判定）。"""
    import torch
    from bench.surprisal_press import SurprisalPress
    p = SurprisalPress(compression_ratio=0.5)
    b, h, s, d = 1, 4, 64, 8
    keys = torch.randn(b, h, s, d); values = torch.randn(b, h, s, d)
    scores = p.score(module=None, hidden_states=torch.randn(b, s, d),
                     keys=keys, values=values, attentions=None, kwargs={"logits": torch.randn(b, s, 100)})
    assert scores.shape == (b, h, s), scores.shape
```

- [ ] **Step 3: 运行确认失败**

Run: `python3 -m pytest tests/test_surprisal_press.py -v`
Expected: FAIL — `ModuleNotFoundError: bench.surprisal_press`

- [ ] **Step 4: 实现（按 Step 1 确认的真实签名调整）**

```python
"""惊讶度 press：importance = -log softmax(logits)[next_token]。

训练无关、零额外前向，只需第一遍 prefill 的 logits。
逐 token 标量广播到所有 (kv_head, layer) —— 与 KVzip 的 per-(layer,head) 粒度不同，
这正是本任务要验证的问题：token 级信号够不够。
"""
import torch


def surprisal_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """logits: (b, seq, vocab) → (b, seq) 的 -log p(token_t)，长度 seq（末位为 0）。"""
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = torch.arange(logits.shape[1], device=logits.device)
    sur = -logp[:, :-1, :].gather(-1, tgt[1:].view(1, -1, 1)).squeeze(-1)
    return torch.cat([torch.zeros(logits.shape[0], 1, device=logits.device), sur], dim=1)


class SurprisalPress:  # 继承 kvpress 的 ScorerPress —— 按 Step 1 实际基类名替换
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        sur = kwargs["logits"]  # 由外层 hook 注入
        return surprisal_from_logits(sur).unsqueeze(1).expand(-1, keys.shape[1], -1)
```

- [ ] **Step 5: 运行确认通过**

Run: `python3 -m pytest tests/test_surprisal_press.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
GIT_MASTER=1 git add bench/surprisal_press.py tests/test_surprisal_press.py
GIT_MASTER=1 git commit -m "feat(kvpress): 惊讶度 press（训练无关，零额外前向）"
```

### Task 1.3: needle rank 探针（廉价、高区分度的主指标）

**Files:**
- Create: `bench/p3_needle_rank.py`
- Test: 报告 rank 数值

**Interfaces:**
- Consumes: 任意 press 的 score；一份 needle-in-haystack context。
- Produces: `rank`（needle 在分数降序中的位次，越小越好）+ `self_check_rate`。

- [ ] **Step 1: 整合现有探针**

```bash
sed -n '1,60p' bench/probe_ablation.py
sed -n '1,50p' bench/firstpass_noncausal.py
```
把其中的 context 构造 + 打分回调抽成 `build_probe_ctx()` / `rank_of(ctx, score_fn)`。

- [ ] **Step 2: 跑基准复现（必须复现历史数字，否则探针本身有问题）**

Run（1.5B，三种已知信号）：
Expected 复现（`docs/attention-signal.md`，给正确 position_ids 后的版本）：

| 信号 | 期望 rank |
|---|---|
| 第一遍 LM + sum | ~98 |
| 第一遍 LM + max | ~16 |
| 真第二遍（A/B/C 任一） | 2–7 |

**若复现不出这张表 → 停，先修探针**（这比任何新信号实验都重要）。

- [ ] **Step 3: 测惊讶度**

Run: 用 Task 1.2 的 press 打分，报 rank。
Expected（待验证）：**若 rank ≤10 → 惊讶度是有效信号**；若 rank ≈ 60–120 → 无效（与末位 query 同类）。

- [ ] **Step 4: 测 7B 上是否一致**（rank 是否随规模改变）
- [ ] **Step 5: 提交**

```bash
GIT_MASTER=1 git add bench/p3_needle_rank.py
GIT_MASTER=1 git commit -m "test(probe): needle rank + 自查率探针（统一入口）"
```

### Task 1.4: 配对 bootstrap CI（补上我们一直缺的统计）

**Files:**
- Create: `bench/bootstrap_ci.py`
- Modify: `bench/p3_longbench_f1.py`（dump 逐样本）

- [ ] **Step 1: 实现 paired bootstrap**

```python
"""配对 bootstrap：同一 prompt 的 (ref, cand) 逐样本 F1 差，重采样 10000 次给 CI。"""
import numpy as np


def paired_bootstrap_ci(ref, cand, n_boot=10000, seed=0, alpha=0.05):
    assert len(ref) == len(cand)
    rng = np.random.default_rng(seed)
    d = np.asarray(cand, dtype=float) - np.asarray(ref, dtype=float)
    n = len(d)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)
```

- [ ] **Step 2: 单测**

```python
def test_paired_ci_zero_when_identical():
    from bench.bootstrap_ci import paired_bootstrap_ci
    m, lo, hi = paired_bootstrap_ci([1.0, 0.0, 0.5], [1.0, 0.0, 0.5])
    assert m == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_ci_excludes_zero_on_clear_gap():
    from bench.bootstrap_ci import paired_bootstrap_ci
    ref = [1.0] * 30
    cand = [0.0] * 30
    m, lo, hi = paired_bootstrap_ci(ref, cand)
    assert hi < 0.0
```

- [ ] **Step 3: 运行 → 通过 → 提交**

```bash
python3 -m pytest tests/test_bootstrap_ci.py -v
GIT_MASTER=1 git add bench/bootstrap_ci.py tests/test_bootstrap_ci.py
GIT_MASTER=1 git commit -m "feat(stats): 配对 bootstrap CI"
```

### Task 1.5: 7B 全量横扫（拿到"可比基线表"）

- [ ] **Step 1: 跑关键方法 × ratio 0.5/0.2/0.1（7B bf16）**

```bash
PYTHONPATH=/root/kvpress-libs python3 bench/kvpress_eval.py \
  --model /root/models/Qwen2.5-7B-Instruct \
  --files qasper.jsonl 2wikimqa.jsonl narrativeqa.jsonl hotpotqa.jsonl multifieldqa_en.jsonl \
  --n 60 --ratio 0.5 --tag 7b-r0.5 --out /root/kvpress-out \
  --methods none compactor chunkkv tova cur leverage snapkv pyramidkv ada surprisal
```
Expected: 出 10 方法表；**对比 1.5B 的排序是否变化**（排序稳定 = 信号结论可跨规模迁移）。

- [ ] **Step 2: 判定 Gate G1**

| 观察 | 判定 |
|---|---|
| 有信号 rank ≤10 且 prefill ≤1.2× | **GO**（带该信号进 Stage 2） |
| 只有 attention-max（rank ~16） | 记录；**评估是否接受 3–5% 保留**（免费方案）；若不可接受 → **转 Stage 4-B** |
| 惊讶度 rank ≤10 但 F1 不升 | 说明 rank 是弱指标 → 以 F1 为准判定 |
| 全部信号 rank ≥50 | **STOP 驱逐方向** → Stage 4-B |

- [ ] **Step 3: 写结论到 `docs/experiments-log.md` 并提交**

---

## Stage 2 — 插件移植（8–14 GPU-h，Gate G2）

**前提：只移植 Stage 1 的赢家。释放路径（`remove_skipped_blocks` / `_patch_build` 的 block_table 压实 / 内存回收）已冻结。**

### Task 2.1: 新增 `PE_SCORE_MODE`（不改释放路径）

**Files:**
- Modify: `kvcompress/pevict.py`（信号分支）
- Test: `tests/test_pevict_parity.py`

**Interfaces:**
- Consumes: 现有 `_PE_RETAINED` / `PE_SCORE_MODE` 机制。
- Produces: `PE_SCORE_MODE=<winner>`，与既有 `window|context|expected` 并列。

- [ ] **Step 1: 找出现有信号分支**

```bash
grep -n "_PE_SCORE_MODE\|PE_SCORE_MODE\|def _score" kvcompress/pevict.py | head -20
```

- [ ] **Step 2: 加分支（以惊讶度为例；若赢家是别的信号则替换 `_score_surprisal` 实现）**

```python
def _score_surprisal(self, logits, block_size):
    """PE_SCORE_MODE=surprisal：用第一遍 logits 的 -log p 逐 token 打分，块内取 max。"""
    import torch
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = torch.arange(logits.shape[-2], device=logits.device)
    sur = -logp[..., :-1, :].gather(-1, tgt[1:].view(*([1] * (logits.dim() - 2)), -1, 1)).squeeze(-1)
    sur = torch.cat([sur[..., :1] * 0 + sur[..., :1], sur], dim=-1)
    nb = sur.shape[-1] // block_size
    return sur[..., :nb * block_size].view(*sur.shape[:-1], nb, block_size).amax(-1)
```

- [ ] **Step 3: 加 parity 测试（防"HF 好、插件坏"——我们被这个坑过：scatter qasper 0.3448→0.2166）**

```python
# tests/test_pevict_parity.py
def test_surprisal_block_scores_match_hf_reference():
    """同样的 logits，插件的块级打分必须等于 HF press 的块级聚合（±1e-4）。"""
    import torch
    from bench.surprisal_press import surprisal_from_logits
    torch.manual_seed(0)
    logits = torch.randn(1, 64, 128)
    bs = 16
    hf = surprisal_from_logits(logits).view(1, 4, bs).amax(-1)
    from kvcompress.pevict import _score_surprisal_ref
    pl = _score_surprisal_ref(logits, bs)
    assert torch.allclose(hf, pl, atol=1e-4)
```

- [ ] **Step 4: 跑测试 → 通过 → 提交**

```bash
python3 -m pytest tests/test_pevict_parity.py -v
GIT_MASTER=1 git add kvcompress/pevict.py tests/test_pevict_parity.py
GIT_MASTER=1 git commit -m "feat(pevict): PE_SCORE_MODE=surprisal（训练无关信号）"
```

### Task 2.2: 插件端 7B parity 实测（Gate G2）

- [ ] **Step 1: 同一批 prompt，HF press vs 插件，各跑一次**

```bash
# HF
PYTHONPATH=/root/kvpress-libs python3 bench/kvpress_eval.py --model /root/models/Qwen2.5-7B-Instruct \
  --files qasper.jsonl multifieldqa_en.jsonl --n 20 --ratio 0.5 --tag hf-parity \
  --methods surprisal none
# 插件
PE_MODEL=/root/models/Qwen2.5-7B-Instruct PE_GMEM=0.85 PE_BACKEND=TRITON_ATTN \
  PE_MODE=chunkkv PE_SCORE_MODE=surprisal PE_RATIO=0.5 PE_N=20 \
  python3 bench/p3_longbench_f1.py
```
- [ ] **Step 2: 判定**：两者 Δvs none 之差 ≤0.01 → G2 通过；否则先修 parity。
- [ ] **Step 3: 检查无 pool leak**（`grep "freed=" 日志`，pool 不得单调增长）
- [ ] **Step 4: 提交**

---

## Stage 3 — 端到端验证（6–10 GPU-h，Gate G3）

### Task 3.1: 质量（带配对 CI）

- [ ] 7B、5 任务 × 60、ratio {0.5, 0.2, 0.1, 0.05}，dump 逐样本 → Task 1.4 的配对 CI
- [ ] 判定：**T1 是否达成**（CI 下界 ≥ −0.01 且压缩 ≥2×）

### Task 3.2: 正确性 —— NaN 毒化（把"物理释放"从推断变证明）

**Files:** Create: `bench/p3_poison.py`

**为什么必须做**：FA2 的发现说明这条路径能**静默读已释放显存**。我们至今只证明了"块被归还"，没证明"被释放的块不会被读到"。

- [ ] **Step 1: 在 `pevict` 释放块后，把该块填 NaN**（仅测试模式 `PE_POISON=1`）
- [ ] **Step 2: 单请求跑到结束**（期间不会复用该块），比对输出与不毒化时一致
- [ ] **Step 3: 若输出出现 NaN/变化 → 说明仍在读已释放块 → 严重 bug，回 Stage 2**
- [ ] **Step 4: 提交**

### Task 3.3: 系统（吞吐 / 容量）

- [ ] 容量受限 decode 负载（M=128、~3k prompt、T=512），对比 passthrough vs 赢家配置
- [ ] 报 KV 池 token 容量 + tok/s（T3）

### Task 3.4: 诚实对照（回答"为什么不量化"）

- [ ] **注意 sm_86 无 FP8**：vLLM 原生 FP8 KV 需 sm_89+ → 在 3090 上量化对照组是 **INT8/INT4（需自定义 kernel）**。
- [ ] 记录：这正是我们的**平台优势论证**（在 sm_86 上驱逐比量化更可落地），但要明确写清这是平台特性而非普适结论。

### Task 3.5: 结论落盘

- [ ] 达成 T1 → 更新 `docs/paper/chunkkv-eviction-vllm.md` 的 §5/§6，标注 sm_86，给配对 CI
- [ ] 未达成 → 写**负结果**（含 kill 证据），进入 Stage 4-B

---

## Stage 4 — 条件分支（仅在 G1/G3 失败时）

### 4-A: 头级预算（ada 式）—— 若 G1 显示"激进预算下才需要"

- [ ] 用 kvpress 的 `ada` 复现头级预算（80% 下 −0.021，优于 50% 的 −0.013）
- [ ] 评估"零 fork 不可行"的旧结论是否可推翻（这是当初放弃的理由）

### 4-B: 转族（若驱逐信号确认无效）

- [ ] **KV 量化**：3090 上 KIVI/INT8（sub-8bit 需自定义 kernel；FP8 不可用）
- [ ] **KV 卸载**：LMCache / CPU offload（24G + 主机内存是天然的 2 层）
- [ ] **免训练稀疏注意力**：sm_86 解锁 MInference/XAttention（sm_120 上不可用的那些）——**这是新机器带来的、之前完全不存在的选项**

---

## 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| kvpress 与 transformers 5.2.0 在 sm_86 上不兼容 | 中 | 阻塞 Stage 1 | G0 先冒烟；退化到 `--model` = AWQ |
| 惊讶度是"局部信号"，F1 不涨 | **高** | Stage 1 失败 | 这正是 G1 要尽早判定的；F1 为准而非 rank |
| 信号排序随规模变化（1.5B→7B） | 中 | 结论不迁移 | Task 1.5 Step 2 专门检查 |
| 插件 parity 不一致（HF 好插件坏） | 中 | 白跑 | Task 2.1 的 parity 单测 + Task 2.2 实测 |
| 把 sm_86 与 sm_120 数字混用 | 中 | 结论失效 | Global Constraints 明令禁止 |
| 子代理基础设施不可用（本 session 9/9 超时） | **确定** | 无法用 subagent 并行 | 本计划全部为**单机串行可执行**；不依赖 subagent |

---

## 成本汇总（按 Gate 早停）

| 路径 | GPU-h | 产出 |
|---|---|---|
| G0 失败 | ~1 | 环境结论 |
| G1 失败 | ~11 | **负结果**（信号无用）+ 转族决策 —— **最坏情况只花 11 小时** |
| G2 失败 | ~25 | parity 问题定位 |
| 全通过 | ~35 | **≥2×/≤1% 的可发表 + 可落地结论** |

---

## 立即执行清单

- [ ] Task 0.1：kernel 冒烟 → **G0**
- [ ] Task 0.2：7B bf16 就位
- [ ] Task 1.1：kvpress 参数化 + 小 n 试跑（**先验证 HF 路径在 3090 上活着**）
- [ ] Task 1.3 Step 2：**复现历史 rank 表**（探针可信性前提）
- [ ] Task 1.2：惊讶度 press → Task 1.3 Step 3 测 rank ← **最高性价比的一步**
- [ ] Task 1.5：7B 横扫 → **G1 判定**
