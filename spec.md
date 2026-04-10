# my-vllm 项目规格说明

## 项目目标

参考 vLLM V1 源码（仓库路径 `/d/develop/vllm`），实现一个基于 GPU 的精简推理引擎，核心保留 **PagedAttention** 和 **Continuous Batching** 两大创新，支持 Qwen2 系列模型的离线推理。

## 规格约束

- 项目代码不超过 5000 行（不包含测试用例），当前实际 **1768 行**
- 每个函数和模块都需要单元测试用例，当前共 **177 个测试**
- 每次迭代开发后都需要所有测试用例通过

## 软件架构

### 整体分层

```
┌─────────────────────────────────────────────────┐
│  用户层    LLM.generate()                        │
├─────────────────────────────────────────────────┤
│  引擎层    LLMEngine (add_request / step 循环)    │
│            ├─ Tokenizer (编码/解码)               │
│            ├─ Scheduler (FCFS 调度 + 抢占)        │
│            │   └─ BlockManager (分页块分配/释放)   │
│            └─ GPUWorker                          │
│                └─ ModelRunner (输入准备 + 前向)    │
│                    ├─ Qwen2ForCausalLM           │
│                    │   ├─ Qwen2Attention + RoPE  │
│                    │   ├─ PagedAttention (KV缓存) │
│                    │   └─ Qwen2MLP (SwiGLU)      │
│                    └─ Sampler (采样)              │
├─────────────────────────────────────────────────┤
│  数据结构   Request, SamplingParams,              │
│            SchedulerOutput, RequestOutput         │
└─────────────────────────────────────────────────┘
```

### 请求生命周期

```
LLM.generate(prompts)
  └─ LLMEngine.add_request(prompt, params)
       ├─ Tokenizer.encode(prompt) → token_ids
       └─ Scheduler.add_request(request) → waiting 队列

  └─ LLMEngine.step() 循环:
       ├─ Scheduler.schedule()
       │   ├─ running 队列: 为 decode 请求分配 slot
       │   ├─ 内存不足时: 抢占 → preempted 队列
       │   └─ waiting 队列: 为 prefill 请求分配块
       ├─ GPUWorker.execute_model(scheduler_output)
       │   ├─ ModelRunner.prepare_inputs() → tensors
       │   ├─ Qwen2ForCausalLM.forward() → logits
       │   └─ Sampler(logits) → token_ids
       └─ Scheduler.update_from_output()
            ├─ 追加生成 token, 检查停止条件
            └─ 完成的请求: 释放块, 返回 RequestOutput
```

### 模块清单

```
my_vllm/
  __init__.py                    # 包入口: 导出 LLM, SamplingParams
  config.py                      # ModelConfig, CacheConfig, SchedulerConfig
  sampling_params.py             # SamplingParams (temperature, top_k, top_p)
  sequence.py                    # Request, SequenceStatus
  outputs.py                     # CompletionOutput, RequestOutput
  tokenizer.py                   # HuggingFace Tokenizer 封装
  utils.py                       # Counter, cdiv, set_random_seed
  block_manager.py               # PhysicalBlock, BlockAllocator, BlockManager
  scheduler.py                   # Scheduler (FCFS + continuous batching + 抢占)
  sampler.py                     # greedy / temperature / top-k / top-p 采样
  worker.py                      # ModelRunner + GPUWorker
  engine.py                      # LLMEngine (step 循环 + 请求管理)
  llm.py                         # LLM 类 (用户 API)
  attention/
    paged_attention.py           # PagedAttention (分页 KV cache 读写 + 注意力)
  model/
    layers.py                    # RMSNorm, RotaryEmbedding, QKV/Linear 封装
    qwen2.py                     # Qwen2ForCausalLM (Attention, MLP, Decoder)
    model_loader.py              # safetensors 权重加载 + HF 名称映射
tests/
  test_config.py                 # 17 tests
  test_sampling_params.py        # 14 tests
  test_outputs.py                # 6 tests
  test_sequence.py               # 24 tests
  test_utils.py                  # 12 tests
  test_block_manager.py          # 21 tests
  test_scheduler.py              # 17 tests
  test_layers.py                 # 12 tests
  test_paged_attention.py        # 9 tests
  test_qwen2.py                  # 10 tests (含 GQA, weight mapping)
  test_model_loader.py           # 4 tests
  test_sampler.py                # 8 tests
  test_worker.py                 # 6 tests
  test_engine.py                 # 8 tests
  test_llm.py                    # 5 tests
  test_e2e.py                    # 5 tests (端到端集成)
```

### 与 vLLM 的简化对比

| 特性 | vLLM | my-vllm |
|------|------|---------|
| 分布式 (TP/PP/DP) | 支持 | 仅单 GPU |
| 异步引擎 | AsyncLLM | 仅同步 |
| API Server | OpenAI-compatible | 无 |
| 模型支持 | 100+ 架构 | 仅 Qwen2 |
| 注意力后端 | FlashAttention / 多种 | PyTorch 原生 |
| 投机解码 | 支持 | 无 |
| LoRA / 多模态 / 量化 | 支持 | 无 |
| Prefix Caching | 支持 | 无 |
| CUDA 自定义 Kernel | 大量 | 无 |

### 关键技术点

- **PagedAttention**: 纯 PyTorch 实现分页 KV cache；prefill 阶段使用 `F.scaled_dot_product_attention`，decode 阶段从 block_table 索引 gather 历史 KV
- **Continuous Batching**: 每个 step 动态决定 batch 组成，混合 prefill 和 decode 请求
- **KV Cache 内存管理**: 启动时 profile GPU 可用内存，预分配固定数量 block，运行时按需分配/释放
- **GQA**: 支持 num_kv_heads != num_heads 的分组查询注意力
- **权重加载**: 将 HF checkpoint 中分离的 Q/K/V 权重合并为 QKV 投影矩阵

## 开发环境

- Python 虚拟环境位于项目目录下 `venv/`
- pip 镜像源: `https://pypi.tuna.tsinghua.edu.cn/simple`
- 详细开发过程记录在 `project.md`
