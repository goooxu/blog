# CUDA优化实战 - 分割文件问题

> 问题描述：在一个很大的文件中有若干以`\r\n`分开的行，找出所有行尾的偏移坐标

## 朴素算法

```c++
for (size_t i = 0; i < buffer_size - 1; i++) {
    if (buffer[i] == '\r' && buffer[i + 1] == '\n') {
        *breaks++ = i + 2;
    }
}
```

对于朴素算法，测试结果如下 

算法|用时(ms)
---|---
朴素算法|1759ms

我使用的测试环境是DGX-1服务器，拥有40个`Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz`核（80线程），512G内存，8块V100显卡

编译器版本为`gcc 7.4.0`和`nvcc 10.1`

那我们来看看，如此简单的一个程序，会有什么样的优化思路

## 思路一，算法上的优化

观察这个简单的程序，需要对每个字符都访问一下，能不能从算法上减少访问的次数？当然是可行的，就引出算法上优化的方法

因为寻找的子串`\r\n`长度是2，所以并不需要访问母串中的每一个字符，可以只检查偶数位的字符

方法如下
```c++
if (buffer_size > 1 && buffer[0] == '\r' && buffer[1] == '\n') {
    *breaks++ = 2;
}

for (size_t i = 2; i < buffer_size - 1; i += 2) {
    if (buffer[i] == '\r') {
        if (buffer[i + 1] == '\n') {
            *breaks++ = i + 2;
        }
    } else if (buffer[i] == '\n') {
        if (buffer[i - 1] == '\r') {
            *breaks++ = i + 1;
        }
    }
}

if (buffer_size > 1 && buffer_size % 2 != 0) {
    if (buffer[buffer_size - 2] == '\r' && buffer[buffer_size - 1] == '\n') {
        *breaks++ = buffer_size;
    }
}
```

对于思路一，测试结果如下 

算法|用时(ms)
---|---
检查所有字符|1759ms
只检查偶数字符|1312ms


## 思路二，并行化

除了算法方面的优化，同样重要的是实现上的优化，并行化就是一个重要的方向

因为比较每个字符的过程是独立的，完全可以并行，我们使用多线程来实现并行化

我们使用OpenMP来实现CPU上的并行化，代码如下
```c++
size_t token_index = 0;

if (buffer_size > 1 && buffer[0] == '\r' && buffer[1] == '\n')
    breaks[token_index++] = 2;

#pragma omp parallel for
for (size_t i = 2; i < buffer_size; i += 2) {
    if (buffer[i] == '\r') {
        if (buffer[i + 1] == '\n') {
            size_t index;
#pragma omp atomic capture
            index = token_index++;
            breaks[index] = i + 2;
        }
    } else if (buffer[i] == '\n') {
        if (buffer[i - 1] == '\r') {
            size_t index;
#pragma omp atomic capture
            index = token_index++;
            breaks[index] = i + 1;
        }
    }
}

if (buffer_size > 1 && buffer_size % 2 != 0) {
    if (buffer[buffer_size - 2] == '\r' && buffer[buffer_size - 1] == '\n')
        breaks[token_index++] = buffer_size;
}
```

对于思路二，测试结果如下 

方法|用时(ms)
---|---
串行|1312ms
CPU上并行（80线程）|130ms

在80个线程的环境下，用时是原来的`1/10`，而不是线性地减少，这是由内存访问的限制导致的


## 思路三，更高的并行度

为了实现更高的并行度，我们可以使用CUDA的编程模型，借助于通用计算的显卡来完成

最简单的实现，让每一个线程去检查一个字符

内核函数代码如下
```c++
__global__ void work(const char *buffer, size_t buffer_size, size_t *tokens,
                     int *token_index) {

  size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

  if (i == 0) {
    if (buffer[0] == '\r' && buffer[1] == '\n') {
      tokens[atomicAdd(token_index, 1)] = i + 2;
    }
  } else if (i == buffer_size - 1) {
    if (buffer[i - 1] == '\r' && buffer[i] == '\n') {
      tokens[atomicAdd(token_index, 1)] = i + 1;
    }
  } else if (i < buffer_size - 1) {
    if (buffer[i] == '\r') {
      if (buffer[i + 1] == '\n') {
        tokens[atomicAdd(token_index, 1)] = i + 2;
      }
    } else if (buffer[i] == '\n') {
      if (buffer[i - 1] == '\r') {
        tokens[atomicAdd(token_index, 1)] = i + 1;
      }
    }
  }
}
```

启动内核函数的代码如下
```c++
size_t threads = (buffer_size + 1) / 2;
const size_t block_dim = 1024;
size_t blocks = threads / block_dim + ((threads % block_dim) != 0 ? 1 : 0);

work<<<blocks, block_dim>>>(d_buffer, buffer_size, d_breaks, d_break_index);
```

每个线程块使用了`1024`个线程，对于`4GB`大小的文件，线程块的数量为`2097152`，总计约21亿个线程，平均每2个字符就分配1个线程

对于思路三，测试结果如下 

步骤|CPU上并行用时(ms)|GPU上并行用时(ms)
---|---|---
内存拷贝H2D|N/A|487.5
内核函数运行|130.0|10.0
内存拷贝D2H|N/A|4.5

可见，迁移到GPU后，单从运算时间上是大大减少了

## 思路四 - 使用PINNED内存

观察实验数据，发现其中内存拷贝占据了很大的部分时间，由CUDA的编程模型可知，GPU上使用的数据需要先从Host拷贝到Device上，如果在Host上申请内存的时候，申请的为`PINNED`内存，则可以节省拷贝的时间

方法很简单，只需要把`malloc`/`free`改成`cudaMallocHost`/`cudaFreeHost`即可

对于四路四，测试结果如下

步骤|普通内存用时(ms)|PINNED内存用时(ms)
---|---|---
内存拷贝H2D|487.5|388.6
内核函数运行|10.0|10.0
内存拷贝D2H|4.5|4.4

在内存拷贝的用时上，大约节省了20%

## 思路五 - 简化内核函数

我们观察方法三的内核函数，使用了`if-else`的分支语句，但是前2个判断条件在21亿个线程里，只各有1个线程符合，这其实是大量浪费的计算，
我们可以通过精细地调整线程的数量和每个线程的访问位置，来把这个分支语句去掉。

有一个方法是，把需要访问的位置，从偶数位变成奇数位，然后控制线程数量，去除冗余的线程

内核代码可以精简为
```c++
size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;

if (buffer[i] == '\r') {
    if (buffer[i + 1] == '\n') {
        tokens[atomicAdd(token_index, 1)] = i + 2;
    }
} else if (buffer[i] == '\n') {
    if (buffer[i - 1] == '\r') {
        tokens[atomicAdd(token_index, 1)] = i + 1;
    }
}
```

申请内存和拷贝的代码如下
```c++
size_t d_buffer_padding_size = (buffer_size % 2048 != 0) ? (2048 - buffer_size % 2048) : 0;
checkCudaErrors(cudaMalloc((void **)&d_buffer, buffer_size + d_buffer_padding_size + 1));
...
checkCudaErrors(cudaMemcpy((void *)d_buffer, buffer, buffer_size, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemset((void *)(d_buffer + buffer_size), 0, d_buffer_padding_size + 1));
```

调用核函数的代码如下
```c++
size_t threads = (buffer_size + d_buffer_padding_size) / 2;
const size_t block_dim = 1024;
size_t blocks = threads / block_dim;

work<<<blocks, block_dim>>>(d_buffer, d_breaks, d_break_index);
```

对于思路五，测试结果如下

步骤|冗余的内核函数|精简的内核函数
---|---|---
内存拷贝H2D|388.6|388.6
内核函数运行|10.0|9.0
内存拷贝D2H|4.4|4.4

在内核函数的用时上，大约减少了10%

## 思路六 - 通过多GPU分发数据

虽然内核函数的时间有所减少，但相对于内存拷贝的时间来说还是杯水车薪，我们如何继续减少内存拷贝的时间？

由于笔者使用的测试环境是DGX-1服务器，有8块V100 GPU，GPU之间通过NVLink连接，数据传输速度很快，CPU与GPU之间的带宽是独立的，多个GPU可以同时从CPU拷贝数据

思路如下，我们可以使用4块GPU（1主3从）同时从CPU拷贝各1/4的数据，则Host到Device的拷贝时间可以减少为原来的1/4，然后再从3个从GPU把数据拷回主GPU，最后在主GPU上执行内核

内存拷贝分发的代码如下
```c++
int total_gpu_count;
checkCudaErrors(cudaGetDeviceCount(&total_gpu_count));

int master_device = 0;
const size_t use_gpu_count = 4;

for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaFree(0));
    if (i > 0) {
        checkCudaErrors(cudaDeviceEnablePeerAccess(0, 0));
    }
}

const char *d_buffer[use_gpu_count];
size_t *d_breaks;
int *d_break_index;
size_t d_buffer_padding_size = (buffer_size % 2048 != 0) ? (2048 - buffer_size % 2048) : 0;

for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMalloc((void **)&d_buffer[i],
                                i == 0 ? buffer_size + d_buffer_padding_size + 1
                                        : (buffer_size + d_buffer_padding_size) /
                                            use_gpu_count));
}

checkCudaErrors(cudaSetDevice(master_device));
checkCudaErrors(cudaMalloc((void **)&d_breaks, max_breaks_count * sizeof(size_t)));
checkCudaErrors(cudaMalloc((void **)&d_break_index, sizeof(int)));

for (size_t i = 0; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMemcpyAsync(
        (void *)d_buffer[i],
        buffer + i * (buffer_size + d_buffer_padding_size) / use_gpu_count,
        (buffer_size + d_buffer_padding_size) / use_gpu_count,
        cudaMemcpyHostToDevice));
}

checkCudaErrors(cudaSetDevice(master_device));
checkCudaErrors(cudaMemsetAsync((void *)(d_buffer[0] + buffer_size), 0, d_buffer_padding_size + 1));
checkCudaErrors(cudaMemsetAsync(d_break_index, 0, sizeof(int)));

for (size_t i = 1; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaMemcpyPeerAsync(
        (void *)(d_buffer[0] + i * (buffer_size + d_buffer_padding_size) / use_gpu_count),
        0, d_buffer[i], i, (buffer_size + d_buffer_padding_size) / use_gpu_count));
}
for (size_t i = 1; i < use_gpu_count; i++) {
    checkCudaErrors(cudaSetDevice(i));
    checkCudaErrors(cudaDeviceSynchronize());
}
```

对于思路六，测试结果如下

步骤|单GPU拷贝数据|多GPU拷贝数据
---|---|---
内存拷贝H2D|388.6|305.5
内核函数运行|9.0|9.0
内存拷贝D2H|4.4|9.2

在H2D内存拷贝的用时上，减少了20%左右


### 思路七 - 简化多GPU编程模型

我们发现思路六的内存拷贝效率虽然有所提高，但编程上稍显复杂，需要反复切换当前设备

可以使用多线程多GPU的编程模型，每一个线程维护一个GPU的上下文，避免反复切换，代码如下 

```c++
#pragma omp parallel num_threads(use_gpu_count)
  {

    int id = omp_get_thread_num();
    checkCudaErrors(cudaSetDevice(id));
    checkCudaErrors(cudaFree(0));
    if (id != 0) {
      checkCudaErrors(cudaDeviceEnablePeerAccess(0, 0));
    }

    if (id == 0) {
      checkCudaErrors(cudaMalloc((void **)&d_buffer[0],
                                 buffer_size + d_buffer_padding_size + 1));

      checkCudaErrors(
          cudaMalloc((void **)&d_breaks, max_breaks_count * sizeof(size_t)));
      checkCudaErrors(cudaMalloc((void **)&d_break_index, sizeof(int)));

      checkCudaErrors(cudaMemset((void *)(d_buffer[0] + buffer_size), 0,
                                      d_buffer_padding_size + 1));
      checkCudaErrors(cudaMemset(d_break_index, 0, sizeof(int)));
    } else {
      checkCudaErrors(
          cudaMalloc((void **)&d_buffer[id],
                     (buffer_size + d_buffer_padding_size) / use_gpu_count));
    }

#pragma omp barrier

    checkCudaErrors(cudaMemcpy(
        (void *)d_buffer[id],
        buffer + id * (buffer_size + d_buffer_padding_size) / use_gpu_count,
        (buffer_size + d_buffer_padding_size) / use_gpu_count,
        cudaMemcpyHostToDevice));

    if (id != 0) {
      checkCudaErrors(cudaMemcpyPeer(
          (void *)(d_buffer[0] +
                   id * (buffer_size + d_buffer_padding_size) / use_gpu_count),
          0, d_buffer[id], id,
          (buffer_size + d_buffer_padding_size) / use_gpu_count));
    }
    checkCudaErrors(cudaDeviceSynchronize());

#pragma omp barrier

    if (id == 0) {
      size_t threads = (buffer_size + d_buffer_padding_size) / 2;
      const size_t block_dim = 1024;
      size_t blocks = threads / block_dim;
      
      work<<<blocks, block_dim>>>(d_buffer[0], d_breaks, d_break_index);
      checkCudaErrors(cudaDeviceSynchronize());
    }
```

对于思路七，测试结果如下

步骤|单线程多GPU编程模型|多线程多GPU编程模型
---|---|---
内存拷贝H2D|305.5|300.8
内核函数运行|9.0|9.1
内存拷贝D2H|9.2|9.5

可见这两种编程模型下的用时是基本上一样的

## 总结

以上多种方法对同一个问题进行递进式的优化

虽然由于问题本身的特点（多内存少计算），并不特别合适在GPU上执行，但是希望这个教程能够抛砖引玉，帮助大家开拓思路

具体的细节可以参考完整的[原代码](https://github.com/goooxu/parallel_programming/tree/master/cuda_study/tokenizer)