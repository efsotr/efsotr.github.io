---
layout: post
title: "[PyTorch API Guide] torch.compile"
date: 2025-06-21
---

**Keywords**: torch.compile, PyTorch 2.0, pt2, max-autotune, activation checkpointing, gradient checkpointing, memory budget

**Chinese Version:** [https://zhuanlan.zhihu.com/p/30568383519](https://zhuanlan.zhihu.com/p/30568383519)

### Introduction

PyTorch is evolving rapidly. As of June 21, 2025, the latest version is torch 2.7.1. This article series aims to thoroughly explore the existing PyTorch APIs and provide clear, concise usage guides to help users write cleaner and more efficient code. At the same time, it seeks to encourage broader community adoption, thereby driving the continued refinement and enhancement of these APIs.



### Brief Overview

`torch.compile`, introduced in PyTorch 2.0, is a powerful feature that optimizes PyTorch code with just a single line. Its optimization workflow resembles that of a traditional compiler and uses Triton as the backend for fused kernel generation.



### Execution Flow

When considering usage only, a single execution process can be simplified into the following three steps:

1. The code is run and analyzed via symbolic execution, transforming the computation into a complete computation graph.
2. The graph is then optimized by fusing multiple small operators into single kernels, significantly reducing memory reads/writes, kernel launch overhead, and CPU-GPU synchronization. These fused operators are implemented using [Triton](https://triton-lang.org/main/getting-started/tutorials/index.html).
3. If the `triton.cudagraphs` option is enabled, [CUDA graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs) are used to capture the entire computation graph.



### Graph Break

A **graph break** occurs when some logic cannot be parsed, potentially causing a compilation failure. A graph break interrupts the computation graph, and the unparseable part falls back to eager mode. 
This prevents the compiler from accessing the full context, which can negatively impact optimization. 
Additionally, control flow structures that depend on static information may fail due to incomplete tracking.

#### Common Causes of Graph Breaks

* Using unsupported PyTorch or custom operators
* Runtime-dependent control flows that can't be statically analyzed (e.g., data-dependent `if` branches, dynamic loop exit conditions)
* Dynamic indexing or structural changes to containers like `list` or `dict`
* Calling unsupported Python built-ins or standard library methods (e.g., `copy.deepcopy`, `print`)
* Accessing variables from static classes (as observed)

#### Solutions

* For unsupported operators, consider using [`torch.library`](https://docs.pytorch.org/docs/stable/library.html#creating-new-custom-ops-in-python).
* For runtime-dependent control flows or operators, move them outside of `torch.compile` and preprocess the data beforehand.
* For other issues, seek alternative implementations or remove problematic code (e.g., `print`).



### Recompilation

By default, `torch.compile` optimizes based on the specific execution instance, considering the execution path, non-Tensor objects, and Tensor shapes. If any of these factors differ from the compiled instance, recompilation is triggered.

To reduce recompilation due to minor shape differences, the compiler tries to generate a general instance that supports dynamic dimensions when subsequent inputs don't exactly match the original shape. You can enforce this general strategy from the start by setting `dynamic=True`.

**Note**:

* Each object has a small default cache limit for compiled instances.
* Since CUDA Graphs can only replay identical computation graphs. Any new shape requires recompilation and separate graph recording. The memory used by CUDA Graphs scales with the number of recorded graphs, so it's best to keep shapes limited—ideally to just one.



### Debugging

Set the `TORCH_LOGS` environment variable with the appropriate debugging flags to analyze `torch.compile` behavior:

* Add `graph_breaks` to view where and why graph breaks occur.
* Add `recompiles_verbose` to check the reasons behind recompilation.
* Add `output_code` to output the optimized code (for GPU code, Triton is used by default).

For more options, refer to [**torch.compile Troubleshooting**](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html#summary-of-torch-logs-options).


### Usage

#### Apply directly to a model instance

If `model` is an instance of `nn.Module`, you can compile it directly:

```python
compiled_model = torch.compile(model, mode="default")
```

**Note**: When using `torch.compile` on an `nn.Module`, the resulting `compiled_model` will wrap the original model inside an `_orig_mod` layer. Be mindful of this when calling methods like `state_dict()`.

#### Apply directly to a function

You can also apply `torch.compile` as a decorator to a function:

```python
@torch.compile(mode="default")
def model_forward(kwargs):
    return model(**kwargs)
```



### \[Advanced] Non-intensive Ops Selective Activation Checkpointing (nosac)

[**Activation Checkpointing** (also known as Gradient Checkpointing)](https://pytorch.org/docs/stable/checkpoint.html) is a widely used technique to reduce memory usage during training by trading compute for memory.

Traditionally, it works by storing only the inputs of a Transformer block and re-computing the full block during the backward pass to recover intermediate activations. However, this recomputation strategy doesn't have to apply to entire blocks. For example, it's often more efficient to recompute only lightweight operations such as `RMSNorm`, `SwiGLU(x) * y`, etc. This enables memory savings with minimal recomputation overhead.

Starting in **Torch 2.4**, a new [**Memory Budget API**](https://pytorch.org/blog/activation-checkpointing-techniques/#compile-only-memory-budget-api-new) allows users to implement such selective recomputation with just a few lines of code:

```python
import torch._functorch.config
torch._functorch.config.activation_memory_budget = 0.99
```

Due to certain internal issues in PyTorch's implementation, this optimization may unintentionally increase runtime. To mitigate this, you can manually modify the operator list to exclude specific ops like `torch.ops.aten.add`:

```python
import torch._functorch.partitioners as partitioners

def remove_add(fn):
    def wrapped_fn():
        optypes = fn()
        optypes.recomputable_ops.remove(torch.ops.aten.add)
        return optypes
    return wrapped_fn

partitioners.get_default_op_list = remove_add(partitioners.get_default_op_list)
```


In practice, for **meta-llama/Llama-3.2-1B**, with `batch_size = 4` and `seq_len = 1024`, running on an A40 GPU, applying this technique results in notable improvements in memory usage with negligible overhead in runtime — enabling larger batches or longer sequences within the same hardware constraints.

| Mode                         | Description                                                                                              | Activation Memory Usage (GB) | Time for 16 Forward Passes (s) |
|-----------------------------|----------------------------------------------------------------------------------------------------------|------------------------------|-------------------------------|
| eager                       |                                                                                                          | 8.65                         | 6.9358                        |
| default                     |                                                                                                          | 5.64                         | 5.3049                        |
| + nonsac                    |                                                                                                          | 4.13                         | 5.2746                        |
| reduce-overhead             | Reduces Python overhead by leveraging CUDA graphs                                                       | 5.64                         | 4.3035                        |
| max-autotune-no-cudagraphs | Deep matrix multiplication optimization using Triton or template-based methods on supported devices. Note: PyTorch's default may not be optimal. | 5.64                         | 4.9355                        |
| max-autotune                | Enables CUDA Graphs by default on GPU                                                                   | 5.64                         | 3.9695                        |
| + nonsac                    |                                                                                                          | 4.13                         | 4.0174                        |



### \[Advanced / Work-in-Progress] Surpassing Traditional Gradient Checkpointing with the Memory Budget API

Traditional Gradient Checkpointing suffers from three main issues:

1. **Graph Breaks with `torch.compile`**:
   When compiling each Transformer block manually, it works under the `default` mode. However, in `max-autotune` mode, it fails because the same tensor cannot be captured by multiple CUDA Graphs.

2. **Redundant Recomputation**:
   The final `down_proj` in each Transformer block does not need to be recomputed, since only the intermediate activations are required.

3. **Lack of Fine-Grained Control**:
   Current checkpointing mechanisms do not support more granular recomputation strategies.

All of these limitations can be effectively addressed by setting the `torch._functorch.config.activation_memory_budget` parameter. This parameter controls the target activation memory usage **relative to the default mode**, enabling more intelligent and flexible recomputation strategies.


However, this approach introduces three new issues:

1. **Excessive Computation Overhead**:
   The current implementation for selecting recomputable parts is suboptimal and may significantly increase compute cost. See [related issues](https://github.com/pytorch/pytorch/issues/149258) for more details.

2. **Ignoring Recomputation Memory Overhead**:
   The memory budget estimation only considers the memory used by activations, neglecting additional memory used during recomputation, which can lead to higher-than-expected peak memory usage.

3. **Inaccurate Cost Estimation**:
   The default cost model relies on `flop_counter`, which is not implemented for custom operators (e.g., Flash Attention 2). This can be addressed by manually implementing the missing `flop_counter` functions.
