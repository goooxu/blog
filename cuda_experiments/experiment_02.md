# 实验二 共享内存和寄存器数量对内核运行时间的影响 #

我们还是使用前面提到的计时程序

先来看看使用不同的寄存器数量对程序运行时间的影响

有如下在设备上运行的代码
```C++
const long long int SECOND_CLOCK_CYCLES = 1582 * 1000 * 1000;
const size_t REGISTER_COUNT = 36;

__global__ void foo(int *minCycles, int *maxCycles) {
  int cycles[REGISTER_COUNT];

  for (size_t i = 0; i < REGISTER_COUNT; i++)
    cycles[i] = 0;

  long long int start = clock64();
  for (;;) {

    long long int duration = clock64() - start;
    if (duration >= SECOND_CLOCK_CYCLES)
      break;

    for (size_t i = 0; i < REGISTER_COUNT; i++)
      if (duration % REGISTER_COUNT == i)
        cycles[i]++;
  }

  if (blockIdx.x == 0) {

    if (threadIdx.x == 0) {
      *minCycles = INT_MAX;
      *maxCycles = INT_MIN;
    }

    __syncthreads();

    int totalCycles = 0;
    for (size_t i = 0; i < REGISTER_COUNT; i++)
      totalCycles += cycles[i];

    atomicMin(minCycles, totalCycles);
    atomicMax(maxCycles, totalCycles);
  }
}
```

由`nvcc -arch=sm_61 -gencode=arch=compute_61, code=sm_61 -Xptxas`命令可知，这个内核使用了48个寄存器

通过改变块和线程的数量，我们得到以下实验结果

blocks|threads|kernel run time (ms)
---|---|---
1|1024|905
28|1024|1011
29|1024|1841
29|641|1829
29|640|996
56|640|1000
57|640|1839
57|417|1834
57|416|996
84|416|1002
85|416|1832

下面来解释为什么会出现这样的结果

情况|解释
---|---
*1 blocks, 1024 threads*|只占用1个多处理器，线程数目没有超过限制，寄存器数目48 * 1024 < 65536也没有超过限制，运行时间为1倍
*28 blocks, 1024 threads*|每个多处理器分配到1个块，线程和寄存器数目都没有超过限制，运行时间为1倍
*29 blocks, 1024 threads*|至少有1个多处理器分配到2个块，线程数目没有超过限制，但寄存器数目48 * 1024 * 2 > 65536超过限制，需要至少分2次运行，所以运行时间为2倍
*29 blocks, 641 threads*|至少有1个多处理器分配到2个块，线程数目没有超过限制，虽然寄存器数目48 * 641 * 2 < 65536也没有超过限制，*但是笔者猜测，线程有最小计算单位，有一定误差*，需要至少分2次运行，所以运行时间为2倍
*57 blocks, 640 threads*|至少有1个多处理器分配到3个块，线程数目没有超过限制，但寄存器数目48 * 640 * 3 > 65536超过了限制，需要至少分2次运行，所以运行时间为2倍
*57 blocks, 417 threads*|至少有1个多处理器分配到3个块，线程数目没有超过限制，虽然寄存器数目48 * 417 * 3 < 65536也没有超过限制，*但是笔者猜测，线程有最小计算单位，有一定误差*，需要至少分2次运行，所以运行时间为2倍
*85 blocks, 416 threads*|至少有1个多处理器分配到4个块，线程数目没有超过限制，但寄存器数目48 * 416 * 4 > 65536超过了限制，需要至少分2次运行，所以运行时间为2倍

---

现在我们看看分配不同的共享内存的大小对程序运行时间的影响

有如下的在device上运行的代码
```C++
const long long int SECOND_CLOCK_CYCLES = 1582 * 1000 * 1000;
const size_t SHARED_MEMORY_SIZE = 1024 * 36;

__global__ void foo(int *minCycles, int *maxCycles) {

  __shared__ int cycles[SHARED_MEMORY_SIZE / sizeof(int)];

  cycles[threadIdx.x] = 0;

  long long int start = clock64();
  for (;;) {
    cycles[threadIdx.x]++;

    long long int duration = clock64() - start;
    if (duration >= SECOND_CLOCK_CYCLES) {
      break;
    }
  }

  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      *minCycles = INT_MAX;
      *maxCycles = INT_MIN;
    }

    __syncthreads();

    atomicMin(minCycles, cycles[threadIdx.x]);
    atomicMax(maxCycles, cycles[threadIdx.x]);
  }
}
```

注意，每个块分配了`36 KBytes`大小的共享内存

通过改变块和线程的数量，我们得到以下实验结果

blocks|threads|kernel run time (ms)
---|---|---
1|1024|902
56|1024|976
57|1024|1811
57|1|1735

下面来解释为什么会出现这样的结果

情况|解释
---|---
*1 blocks, 1024 threads*|只占用1个多处理器，线程数目没有超过限制，共享内存大小36K < 96K也没有超过限制，运行时间为1倍
*56 blocks, 1024 threads*|每个多处理器分配到2个块，线程数目没有超过限制，共享内存大小36K * 2 < 96K也没有超过限制，运行时间为1倍
*57 blocks, 1024 threads*|至少有1个多处理器分配到3个块，线程数目超过限制，需要至少分2次运行，所以运行时间为2倍
*57 blocks, 1 threads*|至少有1个多处理器分配到3个块，线程数目没有超过限制，但共享内存大小36K * 3 > 96K，需要至少分2次运行，所以运行时间为2倍

---
