---
layout: post
title: "[BUG + Fix] NCCL Crashes with PyTorch 2.8–2.10 on Older NVIDIA Drivers"
date: 2026-07-31
---

**Keywords**: PyTorch 2.8, PyTorch 2.9, PyTorch 2.10, NCCL, CUDA Driver API, symmetric memory, CUDA_MODULE_LOADING

## Cause

NCCL 2.27 introduced symmetric-memory kernels that require CUDA Driver API 12.7 or later.

User code and frameworks such as Transformers and Accelerate may initialize CUDA before the NCCL communicator is created. This can cause CUDA to use eager module loading, which loads all kernels in a module immediately.

Therefore, when using PyTorch 2.8–2.10 on a system whose NVIDIA driver supports only an older CUDA Driver API version, NCCL communicator initialization may attempt to load symmetric-memory kernels unsupported by the driver, causing the process to crash.

NCCL 2.28.7 fixed this issue by detecting the driver version at runtime and skipping incompatible kernels.

The bundled NCCL version can be checked from the `nvidia-nccl` package version.

## Solutions

1. Enable lazy CUDA module loading so that NCCL does not load the symmetric-memory kernels during communicator initialization:

   ```
   export CUDA_MODULE_LOADING=LAZY
   ```

2. Upgrade to PyTorch 2.11 or later, or NCCL 2.28.7 or later.

3. Downgrade to a PyTorch release bundled with an earlier NCCL version.

## Fix References

- [NCCL v2.28.7-1 Release](https://github.com/NVIDIA/nccl/releases/tag/v2.28.7-1)
- [NCCL 2.28.7 Release Notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-7.html#rel-2-28-7)  
  Search for `older CUDA drivers`.
- [Relevant source-code change in `src/enqueue.cc`](https://github.com/NVIDIA/nccl/blame/v2.28.7-1/src/enqueue.cc#L43-L47)
