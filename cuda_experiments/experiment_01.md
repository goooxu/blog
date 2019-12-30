# 实验一 块和线程数量对内核运行时间的影响 #

> 本实验的结果由**GeForce GTX 1080 Ti**测试得出

我们使用前面提到的计时程序

来看看使用不同的块和线程数量对程序运行时间的影响

有如下的在设备上运行的代码
```C++
const long long int SECOND_CLOCK_CYCLES = 1582 * 1000 * 1000;

__global__ void foo(int *minCycles, int *maxCycles) {

  int cycles = 0;

  long long int start = clock64();
  for (;;) {
    cycles++;

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

    atomicMin(minCycles, cycles);
    atomicMax(maxCycles, cycles);
  }
}
```

为什么设置`SECOND_CLOCK_CYCLES`为`1582 * 1000 * 1000`，因为按照参数表中GPU的运行频率1582 MHz来计算，可以得出大概1,582,000,000‬个时钟周期等于1秒钟

通过改变块和线程的数量，我们得到以下实验结果

blocks|threads|kernel run time (ms)
---|---|---
1|1024|905
56|1024|915
57|1024|1741
57|673|1745
57|672|915
84|672|914
85|672|1747
85|513|1747
85|512|914
112|512|914
113|512|1750

下面来解释为什么会出现这样的结果

首先，引出两条规则

> 规则一：**一个块只会在一个多处理器中运行**

> 规则二：**线程是以wrap为最小单位运行的**

情况|解释
---|---
*1 block, 1024 threads*|只会在1个多处理器中运行，每个线程的运行时间大约是1秒，总计1024个线程，少于单个多处理器最多2048个线程的限制，所以运行时间为1倍
*56 blocks, 1024 threads*|每个多处理器分配到的块数量为56/28=2，2个块总计2048个线程，没有超出多处理器的限制，所以运行时间也为1倍
*57 blocks, 1024 threads*|至少有1个多处理器会分配到3个块，总计3072个线程，超出了多处理器的限制，需要至少分2次运行，所以运行时间为2倍
*57 blocks, 673 threads*|至少有1个多处理器会分配到3个块，总计2019个线程，虽然线程数目没有超出多处理器的限制，但是673个线程至少为22个warps，3个块总计66个warps，超出了多处理器的限制，需要至少分2次运行，所以运行时间为2倍
*85 blocks, 672 threads*|至少有1个多处理器会分配到4个块，总计2688个线程，超出了多处理器的限制，需要至少分2次运行，所以运行时间为2倍
*85 blocks, 513 threads*|至少有1个多处理器会分配到4个块，总计2052个线程，超出了多处理器的限制，需要至少分2次运行，所以运行时间为2倍
*113 blocks, 512 threads*|至少有1个多处理器会分配到5个块，总计2560个线程，超出了多处理器的限制，需要至少分2次运行，所以运行时间为2倍
