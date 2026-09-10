# sglang_kvx — SGLang 散点 KV 驱逐 backend 原型

在 SGLang（0.5.19）上以自定义 `AttentionBackend` 实现**散点 KV 注意力**（SnapKV/H2O 的必要机制）。

## 原理

SGLang decode 通过扁平 `kv_indices` + `kv_indptr` gather KV（page_size=1 时索引即 token 位置）。
本 backend 在 decode 元数据构建后，把每个请求的 KV 索引过滤为 `[头 head 段 + 末尾 window 段]`，
丢弃中段 → decode 只对被保留的分散位置做注意力。

## 运行（服务器）

```bash
cd /root/ai-compress   # 或 PYTHONPATH 指向本目录
KVX_HEAD=64 KVX_WINDOW=32 \
/root/sglang-venv/bin/python your_script.py
```

```python
import sglang_kvx  # 触发注册
from sglang import Engine
e = Engine(model_path="/models/qwen2.5-1.5b-instruct", dtype="bfloat16",
           attention_backend="kvx_scatter", mem_fraction_static=0.6, disable_cuda_graph=True)
e.generate("...", {"max_new_tokens": 16, "temperature": 0})
```

注意：SGLang 需 `attention_backend="triton"` 系列（本机 flashinfer 在 sm_120 不可用）。

## 实测（2026-09-10）

- prompt 402 token → `old_ptr=[0,402]` 过滤为 `new_ptr=[0,96]`（head64+win32），丢弃 306 个 KV。
- 生成正常：`"(1) How many times does the letter 'o' appear in the sentence"`，EXITCODE=0。

## 当前限制 / 二期

1. 仅限制注意力可见范围，**未释放 KV 池槽位**（显存未回收）。
2. 选择策略为**位置式**（头段+窗口），非 SnapKV 的 attention 重要性打分。
3. 未做 baseline 对照的损失量化。


## 子进程注册（必需）

SGLang 的 scheduler 是独立解释器，客户端里的 `import sglang_kvx` 不会传播过去，会报
`ValueError: Invalid attention backend: kvx_scatter`。本机可用做法：

1. 将本包放入 venv site-packages：`cp -r sglang_kvx /root/sglang-venv/lib/python3.11/site-packages/`
2. 在 `sglang/srt/model_executor/model_runner_components/attention_backend_setup.py` 的
   `_build_full_attention_backend_from_str` 中、成员检查之前加惰性导入：
   ```python
   try:
       import sglang_kvx  # 注册自定义 backend
   except Exception:
       pass
   ```
   （此处环境已完全就绪，避免 .pth 启动期导入过早的依赖问题。）
