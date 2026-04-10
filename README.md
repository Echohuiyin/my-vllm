# my-vllm

参考 vLLM 实现的精简 GPU 推理引擎，支持 Qwen2 系列模型，核心特性：**PagedAttention** + **Continuous Batching**。

## 环境要求

- Python >= 3.10
- CUDA GPU (推理时需要)
- 磁盘空间：取决于所用模型大小

## 安装部署

### 1. 创建并激活虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如需 CUDA 版本的 PyTorch，请参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 选择对应的安装命令。

## 快速开始

```python
from my_vllm import LLM, SamplingParams

# 加载模型 (首次运行会自动从 HuggingFace 下载)
llm = LLM(model="Qwen/Qwen2-0.5B")

# 生成文本
outputs = llm.generate(
    ["Hello, world!", "What is artificial intelligence?"],
    SamplingParams(temperature=0.8, max_tokens=50),
)

for output in outputs:
    print(f"Prompt:  {output.prompt}")
    print(f"Output:  {output.outputs[0].text}")
    print()
```

### 使用本地模型

```python
llm = LLM(model="/path/to/local/Qwen2-0.5B")
```

### 参数说明

**LLM 构造参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | (必填) | HuggingFace 模型名或本地路径 |
| `dtype` | `"auto"` | 数据类型: `auto` / `float16` / `bfloat16` / `float32` |
| `max_model_len` | `None` | 最大序列长度，默认从模型配置读取 |
| `gpu_memory_utilization` | `0.9` | GPU 显存使用比例 (0, 1] |
| `max_num_seqs` | `128` | 最大并发序列数 |
| `max_num_batched_tokens` | `2048` | 每步最大 token 数 |
| `block_size` | `16` | KV cache 块大小 |
| `trust_remote_code` | `False` | 是否信任远程代码 |
| `seed` | `0` | 随机种子 |

**SamplingParams 采样参数:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | `1.0` | 温度，0 表示贪心解码 |
| `top_p` | `1.0` | nucleus 采样概率阈值 |
| `top_k` | `0` | top-k 采样，0 表示不限制 |
| `max_tokens` | `16` | 最大生成 token 数 |
| `stop_token_ids` | `None` | 停止 token ID 列表 |
| `seed` | `None` | 采样随机种子 |

## 运行测试

```bash
# 运行全部单元测试 (不含需要网络下载模型的 tokenizer 测试)
python -m pytest tests/ --ignore=tests/test_tokenizer.py -v

# 运行全部测试 (需要网络连接下载 Qwen2-0.5B tokenizer)
python -m pytest tests/ -v

# 运行指定模块测试
python -m pytest tests/test_scheduler.py -v
```

## 项目结构

```
my_vllm/                  # 源码 (1768 行)
  llm.py                  # 用户 API: LLM.generate()
  engine.py               # LLMEngine: 请求管理 + step 循环
  scheduler.py            # FCFS 调度 + continuous batching + 抢占
  block_manager.py        # PagedAttention 块分配与管理
  worker.py               # GPU Worker + ModelRunner
  sampler.py              # 采样策略
  attention/
    paged_attention.py    # PagedAttention 实现
  model/
    qwen2.py              # Qwen2ForCausalLM 模型
    layers.py             # RMSNorm, RoPE, Linear 层
    model_loader.py       # 权重加载
  config.py               # 配置类
  sampling_params.py      # 采样参数
  sequence.py             # 请求状态
  outputs.py              # 输出数据结构
  tokenizer.py            # 分词器封装
  utils.py                # 工具函数
tests/                    # 测试 (177 个测试用例)
spec.md                   # 项目规格说明
project.md                # 开发过程记录
```

## 支持的模型

目前支持 Qwen2 系列模型，包括但不限于：

- `Qwen/Qwen2-0.5B`
- `Qwen/Qwen2-1.5B`
- `Qwen/Qwen2-7B`

模型需要使用 HuggingFace 格式 (safetensors)。
