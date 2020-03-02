这里主要探讨一些CUDA的编程模型

我手上有两种GPU可以用来测试，分别是**GeForce GTX 1080 Ti**和**Tesla V100**，下面它们的主要参数

Property|GeForce GTX 1080 Ti|Tesla V100
---|---|---
Computer Capability|6.1|7.0
Max Clock Rate|1582 MHz|1530 MHz
Global Memory Size|10.9 Gbytes|15.8 Gbytes
Multiprocessors|28|80
Total CUDA Cores|3584|5120
Threads / Warp|32|32
L2 Cache Size|2816 KBytes|6144 KBytes
CUDA Cores / Multiprocessor|128|64
Registers / Multiprocessor|65536|65536
Register File Capacity / Multiprocessor|256 Kbytes|256 Kbytes
Constant Memory Size / Multiprocessor|64 Kbytes|64 Kbytes
Shared Memory Size / Multiprocessor|96 Kbytes|up to 96Kbytes
Max Warps / Multiprocessor|64|64
Max Threads / Multiprocessor|2048|2048
Max Thread Blocks / Multiprocessor|32|32
Max Shared Memory Size / Block|48 Kbytes|48 Kbytes
Max Registers / Block|65536|65536
Max Threads / Block|1024|1024
Max Registers / Thread|255|255
Max dimension size of a thread block|1024,1024,64|1024,1024,64
Max dimension size of a grid size|2147483647, 65535, 65535|2147483647, 65535, 65535
Concurrent copy and kernel execution|Yes with 2 copy engines|Yes with 6 copy engines

>另外，使用的CUDA开发工具版本如下：CUDA Driver Version: 10.2, Runtime Version: 10.2

我们使用如下代码来测试我们程序的运行时间
```C++
int *deviceCounters;
cudaMalloc(&deviceCounters, sizeof(int) * 2);
warmup<<<28, 1024>>>();
cudaDeviceSynchronize();

clock_t t1 = clock();
foo<<<blocks, threads>>>(deviceCounters, deviceCounters + 1);
cudaDeviceSynchronize();
clock_t t2 = clock();

int hostCounters[2]{0};
cudaMemcpy(hostCounters, deviceCounters, sizeof(int) * 2, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

printf("counter1=%d, counter2=%d\n", hostCounters[0], hostCounters[1]);
printf("time_elapsed=%.0fms\n", 1000.0f * (t2 - t1) / CLOCKS_PER_SEC);
```

初始化CUDA上下文需要一定的开销，为了更准确的测量时间，通常会使用一个暖场内核在被测内核之前运行
