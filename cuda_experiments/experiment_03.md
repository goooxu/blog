# 实现三 延迟隐藏 #

我们还是使用前面提到的计时程序

有如下在设备上运行的代码
```C++
const long long int CLOCK_CYCLES = 10 * 1024 * 1024;

__global__ void foo(int *counter1, int *) {
  float f = 0.0f;
  int c = 0;
  for (long long int i = 0; i < CLOCK_CYCLES; i++) {
    f = cosf(f * 2);
    c += static_cast<int>(f + 1.0f);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *counter1 = c;
  }
}
```

通过改变块和线程的数量，我们得到以下实验结果

threads|blocks|kernel run time (ms)|blocks|kernel run time (ms)|blocks|kernel run time (ms)
---|---|---|---|---|---|---
128|1|897|28|899|56|929
256|1|926|28|928|56|1093
384|1|968|28|1005|56|1293
512|1|1022|28|1091|56|1644
640|1|1084|28|1189|56|2056
768|1|1171|28|1295|56|2441
896|1|1337|28|1472|56|2797
1024|1|1494|28|1638|56|3145

现在尝试对实验结果进行一些分析

我们可以观察到几点现象：
1. 在线程数量固定的前提下，内核运行时间随着线程数量的增加而增加，但是是以128为单位的变化的，每增加128个线程，内核的运行时间会增加一个档次
2. 对于1个块和28个块，内核运行时间在大致上是相等的
3. 对于56个块，运行时间和28个块以及一半线程数量的条件下，内核运行时间大致是相等的

对观察到的现象尝试进行解释

由参数表可知 GeForce GTX 1080 Ti 有28个多处理器，每个多处理器有128个CUDA核心

由于我们的内核代码属于密集型运算（反复调用`cos`函数）的状态，可以让GPU的计算单元一直保持忙碌，那128个线程就可以让128个CUDA核心一直处理忙碌。那为什么256个线程的情况下，内核的运行时间不是两倍，而只是增加了一点点呢，这里就要引入“延迟隐藏”的概念了。请看下面一段话，并自行理解

>Each Volta SM includes 4 warp-scheduler units. Each scheduler handles a static set of warps and issues to a dedicated set of arithmetic instruction units. Instructions are performed over two cycles，and the schedulers can issue independent instructions every cycle. Dependent instruction issue latency for core FMA math operations are reduced to four clock cycles，compared to six cycles on Pascal. As a result，execution latencies of core math operations can be hidden by as few as 4 warps per SM，assuming 4-way instruction-level parallelism ILP per warp. Many more warps are，of course，recommended to cover the much greater latency of memory transactions and control-flow operations.

所以可以对**现象一**进行解释，因为计算单元CUDA核心的数目是128，所以线程的数量也是按128为单位来影响运行时间的。同样，由于“延迟隐藏”，所以256个线程相对于128个线程，内核运行时间只是增加了一点，而不是两倍

对于**现象二**，因为当只有1个块的时候，只利用了28个多处理器中的1个，所以同时有28个块和只有1个块的内核运行时间是一样的

对于**现象三**，当有56个块的时候，每个多处理器会分配到2个块，以128个线程为例，此时一共有256个线程需要执行，所以和每个多处理器分配到1个块并且有256个线程需要执行时，所花费的内核时间应该是一致的
