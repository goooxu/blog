---
layout: default
title: Gems并行计算开发者日志
---

## CUDA编程模型
 
 * [引言](cuda_programming/introduction.md)

### 理论

 * [块和线程数量对内核运行时间的影响](cuda_programming/blocks_and_threads.md) 
 * [共享内存和寄存器数量对内核运行时间的影响](cuda_programming/shared_memory_and_register.md)
 * [延迟隐藏](cuda_programming/latency_hiding.md)

### 实战

 * [分割文件问题](cuda_programming/split_problem.md)
 * [精度问题](cuda_programming/precision_problem.md)
 * [全连接层的实现（上）](cuda_programming/fullyconnected.md)