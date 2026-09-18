# vLLM 0.11.2 端口状态（S1–S3 + 性能）2026-09-18

## 一、功能：S1–S3 全部跑通

| 步骤 | 内容 | 实测 |
|---|---|---|
| **S1** | 管路（自定义 spec→manager、真释放、压实 block_table） | `freed=1`，输出正常 |
| **S2** | 注意力打分（chunkkv） | `score keep=31/69` → `evict chunkkv freed=38` |
| **S3** | **生成段窗口**（每 decode 步释放 `[prompt_nblk, (L-W)/bs)`） | `evict genwin blk[69,70) freed=1` 逐步释放 |

**正确性**：NIAH（5 深度 × 3）`off` 15/15、`chunkkv@0.5` 15/15、`chunkkv@0.05` **15/15**、
`context` 模式 15/15 → 与 0.28 历史行为一致，**移植未引入质量回归**。

## 二、性能：先修掉一个移植回退，但生成段窗口仍为净损失

### 修掉的回退（已提交 `fe1cfb8`）

| 配置（M=128, prompt≈1152, gen=512, gmem=0.30） | 优化前 | 优化后 |
|---|---|---|
| `off` | 1.00× | 1.00× |
| `chunkkv@1.0`（只打分） | 0.76× | 0.89× |
| `chunkkv@0.3` | **0.64×** | **1.09×** |

两处修复：打分路径"全批已打分则早退"（消除每步 2 次 CPU-GPU 同步）；
Builder 加保留集版本缓存（`gen_window=0` 时不再每层每步重复压实）。

### 生成段窗口：容量收益大，但实现开销更大

| 配置（M=64, prompt≈1152, **gen=1500**, gmem=0.30） | tok/s | vs off |
|---|---|---|
| `off` | 2848 | 1.00× |
| `chunkkv@0.3, gw=0` | 2526 | 0.89× |
| `chunkkv@0.3, **gw=256**` | 1688 | **0.59×** |

每请求 KV 应从 1846 → 602 token（≈3× 容量），**但吞吐反而再掉 30 个百分点**。

**已排除**：不是"压实被每层重复执行"（加了"每步只压一次"的缓存，0.58× → 0.59×，无改善）。

**已排除的两个假设**（各自单独实测，均无改善）：
1. ❌ "压实被每个 attention 层重复执行" → 加"每步只压一次"缓存（按 common_attn_metadata
   对象做键 + 持强引用防 id 复用）：0.58× → 0.59×
2. ❌ "manager 每步重扫 `[pnb, tail)` 释放区间" → 改为记录 `_GEN_FROM` 只扫新增块：
   0.59× → 0.56×

**剩余怀疑（未验证，需 profiler）**：
1. `seq_lens` / `block_table` 每步变化 → 注意力 kernel 走慢路径或重新 autotune
2. 释放的块被新 token 立即复用 → 块池回收/再分配串行化
3. 压实时把 block_table 尾部写 0，可能被 kernel 当作合法块 0 而多做工作

**方法学教训**：这一步已经连续两次"猜成因 → 改 → 无改善"。继续盲猜是浪费 GPU 预算，
应当改用 torch profiler / nsys 直接看 time breakdown 再动手。

## 三、结论与建议

- **端口本身可用**：功能对齐、质量无回归、prompt 驱逐在 gen=512 形状下 **+9%**。
- **生成段窗口目前不应默认开启**（`PE_GEN_WINDOW=0`）：容量账面上赢 3×，实现上输 41%。
  需要先做**性能剖析**定位那 41%，再决定是否保留这条路。
- 真正能体现生成段窗口价值的形状尚未测到（需要**同 SLA 下 max 并发**或更长的生成），
  但当前实现的开销已经压倒收益，应先优化再测。

## 四、复现

```bash
cd /root/ai-compress
export TMPDIR=/dev/shm/tmp PYTHONPATH=/hy-tmp/t29:/root/ai-compress VLLM_USE_FLASHINFER_SAMPLER=0
export PE_SINK=0 PE_OBS=32 PE_WIN_AGG=max
PE_MODE=chunkkv PE_RATIO=0.3 PE_GEN_WINDOW=0 \
  python3 bench/perf_v11.py --m 64 --prompt-words 576 --gen 1500 --gmem 0.30 --tag x
```
