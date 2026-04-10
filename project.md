# my-vllm 开发记录

## 项目概述

参考 vLLM V1 架构实现的精简 GPU 推理引擎，核心保留 PagedAttention 和 Continuous Batching，支持 Qwen2 系列模型的离线推理。

## 架构设计

```
用户 API (LLM) → 引擎 (LLMEngine) → 调度器 (Scheduler) + 块管理器 (BlockManager)
                                   → 工作器 (GPUWorker/ModelRunner) → 模型 (Qwen2ForCausalLM)
                                                                    → 采样器 (Sampler)
```

核心数据流：`add_request → schedule → execute_model → update_from_output → step 循环`

## 模块清单

| 模块 | 文件 | 职责 |
|------|------|------|
| 配置 | `config.py` | ModelConfig, CacheConfig, SchedulerConfig |
| 采样参数 | `sampling_params.py` | SamplingParams (temperature, top_k, top_p 等) |
| 序列状态 | `sequence.py` | Request, SequenceStatus |
| 输出 | `outputs.py` | CompletionOutput, RequestOutput |
| 分词 | `tokenizer.py` | HuggingFace Tokenizer 封装 |
| 工具 | `utils.py` | Counter, cdiv, set_random_seed 等 |
| 块管理 | `block_manager.py` | PhysicalBlock, BlockAllocator, BlockManager |
| 调度器 | `scheduler.py` | Scheduler (FCFS + continuous batching + 抢占) |
| PagedAttention | `attention/paged_attention.py` | 分页 KV cache 读写与注意力计算 |
| 模型层 | `model/layers.py` | RMSNorm, RotaryEmbedding, Linear 封装 |
| Qwen2 模型 | `model/qwen2.py` | Qwen2Attention, MLP, DecoderLayer, Model, ForCausalLM |
| 模型加载 | `model/model_loader.py` | safetensors 权重加载与 HF→my-vllm 名称映射 |
| 采样 | `sampler.py` | greedy/temperature/top-k/top-p 采样 |
| Worker | `worker.py` | ModelRunner (输入准备+前向), GPUWorker (设备管理+KV cache) |
| 引擎 | `engine.py` | LLMEngine (请求管理+step循环+detokenize) |
| 用户接口 | `llm.py` | LLM 类 (generate API) |

## 迭代开发过程

### 迭代 1：基础设施
- 实现 config.py, sampling_params.py, sequence.py, outputs.py, tokenizer.py, utils.py
- 73 个单元测试全部通过
- 关键决策：ModelConfig 通过 HuggingFace AutoConfig 自动读取模型参数

### 迭代 2：KV Cache 块管理
- 实现 block_manager.py：PhysicalBlock, BlockAllocator, BlockManager
- 21 个单元测试全部通过
- 关键设计：块分配器使用 FIFO 队列管理空闲块，BlockManager 维护 request_id→block_table 映射

### 迭代 3：调度器
- 实现 scheduler.py：FCFS 调度 + continuous batching + 抢占机制
- 17 个单元测试全部通过
- 调度流程：先调度 running 队列 (decode)，OOM 时抢占，再从 waiting 队列调度 (prefill)
- 抢占策略：释放块、清空输出、重置计算进度，放回 preempted 队列

### 迭代 4：模型层
- 实现 layers.py (RMSNorm, RotaryEmbedding, QKVParallelLinear 等)
- 实现 paged_attention.py (PagedAttention：prefill 用 F.scaled_dot_product_attention，decode 用分页 gather)
- 实现 qwen2.py (完整 Qwen2ForCausalLM，支持 GQA)
- 实现 model_loader.py (safetensors 加载 + Q/K/V→QKV 合并映射)
- 实现 sampler.py (greedy/temperature/top-k/top-p)
- 42 个单元测试全部通过
- Bug 修复：RoPE _apply_rotary 中 cos/sin 已是 half-dim，不需要二次切分；decode attention 中 qi 应 unsqueeze(1) 而非 unsqueeze(0)

### 迭代 5：执行与引擎
- 实现 worker.py (ModelRunner 负责输入准备和模型执行，GPUWorker 负责设备和 KV cache 管理)
- 实现 engine.py (LLMEngine 协调 tokenizer, scheduler, worker 的 step 循环)
- 14 个单元测试全部通过
- 清理：删除了与 .py 文件冲突的空 package 目录 (engine/, worker/, core/, sample/)

### 迭代 6：用户接口与集成
- 实现 llm.py (LLM 类：generate() 接口)
- 更新 __init__.py 导出 LLM
- 端到端集成测试：使用随机初始化的小型 Qwen2 模型在 CPU 上完整运行 pipeline
- 10 个测试全部通过（含 E2E）

## 测试汇总

| 测试文件 | 测试数 | 覆盖 |
|---------|--------|------|
| test_config.py | 17 | 配置类创建与校验 |
| test_sampling_params.py | 14 | 采样参数校验 |
| test_outputs.py | 6 | 输出数据结构 |
| test_sequence.py | 24 | 请求状态管理 |
| test_utils.py | 12 | 工具函数 |
| test_block_manager.py | 21 | 块分配与管理 |
| test_scheduler.py | 17 | 调度逻辑与抢占 |
| test_layers.py | 12 | RMSNorm, RoPE, Linear 层 |
| test_paged_attention.py | 9 | PagedAttention 读写与注意力 |
| test_qwen2.py | 10 | Qwen2 模型各组件 |
| test_model_loader.py | 3 | 权重加载 |
| test_sampler.py | 8 | 采样逻辑 |
| test_worker.py | 6 | ModelRunner 输入准备与执行 |
| test_engine.py | 8 | 引擎请求管理与 step |
| test_llm.py | 5 | LLM 接口 |
| test_e2e.py | 5 | 端到端集成 |
| **合计** | **177** | |

## 使用示例

```python
from my_vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2-0.5B")
outputs = llm.generate(
    ["Hello, world!", "What is AI?"],
    SamplingParams(temperature=0.8, max_tokens=50),
)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}")
```

## 与 vLLM 的简化对比

| 特性 | vLLM | my-vllm |
|------|------|---------|
| 分布式 | TP/PP/DP | 仅单 GPU |
| 异步引擎 | AsyncLLM | 仅同步 |
| API Server | OpenAI-compatible | 无 |
| 模型支持 | 100+ 架构 | 仅 Qwen2 |
| 注意力后端 | FlashAttention 等 | PyTorch 原生 |
| 投机解码 | 支持 | 无 |
| LoRA / 多模态 / 量化 | 支持 | 无 |
| CUDA 自定义 Kernel | 大量 | 无 |
