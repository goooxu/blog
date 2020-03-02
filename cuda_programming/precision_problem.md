# 精度问题

这一讲我们来关注一个在深度学习框架中很普遍的问题，精度问题

通过解决下面这个具体的问题，来对精度问题进行探讨

> 对32位符点数的数组求和，误差越低越好

首先，我们需要一个基准，我们使用了任意精度计算库[MPFR](https://www.mpfr.org/)来得到一个标准答案，代码如下

```c++
float sum(const vector<float> &s) {
  mpfr_t a, t;
  mpfr_init2(a, 1024);
  mpfr_init2(t, 1024);
  mpfr_set_flt(a, 0.0f, MPFR_RNDD);
  for (auto i : s) {
    mpfr_set_flt(t, i, MPFR_RNDD);
    mpfr_add(a, a, t, MPFR_RNDD);
  }

  float result = mpfr_get_flt(a, MPFR_RNDD);
  mpfr_clear(a);
  mpfr_clear(t);
  return result;
}
```

> 累加变量`a`使用了1024位的精度

## 找到最优的方法

有了这个基准，我们便可以对几种常用方法的进行测试，比较优劣

为了编程的方便，首先使用CPU代码来实现算法

### 方法一，32位串行累加

代码如下

```c++
float sum(const vector<float> &s) {
  float a = 0.0f;
  for (const float &e : s)
    a += e;
  return a;
}
```

测试结果如下

数组长度|绝对误差(累计*)|相对误差(累计)
---|---|---
1,024|1.3e-2|1.1e-3
10,000|1.3e-1|4.3e-3
1,000,000|1.2e+1|8.5e-2

> *累计的意思是指1000次测试（每次生成长度相同的随机数组）的误差之和，下同

### 方法二，累加变量升级到64位

把方法一中的32位累加变量升级到64位，可以得到**好的多**的精度，测试结果如下


数组长度|绝对误差(累计)|相对于32位串行累加|相对误差(累计)|相对于32位串行累加
---|---|---|---|---
1,024|1.1e-3|9%|4.3e-5|4%
10,000|3.3e-3|3%|4.4e-5|1%
1,000,000|3.4e-2|0.3%|4.1e-5|0.05%

虽然使用64位累加变量可以得到好的多的精度，但是我还是想回到32位的范畴内继续讨论这个问题

下面的方法会依次与32位串行累加和64位串行累加结果进行比较

### 方法三，先排序再累加

产生误差的原因是由于符点数的二进制结构导致的，当一个比较大的符点数与一个比较小的符点数相加时，由于他们表示小数部分的位数不同，所以会产生误差，当多个符点数累加的时候，这个误差会被放大。

那可以想到一个方法，我们对原始数组按绝对值从小到大排序，开始累加绝对值较小的数，然后逐渐累加绝对值较大的数

代码如下

```c++
float sum(vector<float> &s) {
  sort(s.begin(), s.end(), [](float a, float b) { return abs(a) < abs(b); });
  float a = 0.0f;
  for (float e : s) a += e;
  return a;
}
```

测试结果如下

数组长度|累计绝对误差|相对于32位串行累加|相对于64位串行累加|累计相对误差|相对于32位串行累加|相对于64位串行累加
---|---|---|---|---|---|---
1,024|8.1e-3|63%|735%|7.4e-4|68%|1728%
10,000|7.5e-2|60%|2261%|3.1e-3|73%|7150%
1,000,000|8.0|64%|23557%|1.0e-1|120%|248663%

### 方法四，按平方根分区累加

虽然先排序再串行累加的方法比起直接串行累加的方法精度上好了不少，但还不够好

想到分治的方法，可以把原始数组按长度的平方根分区，先计算每个区的和，再累加，并且递归地完成整个过程

代码如下

```c++
float sum(const vector<float> &s, size_t l, size_t h) {
  size_t n = h - l;
  if (n == 1) {
    return s[l];
  } else if (n == 2) {
    return s[l] + s[l + 1];
  } else {
    size_t sqrtn = ceil(sqrt(n));
    float a = 0.0f;
    while (l < h) {
      a += sum(s, l, min(l + sqrtn, h));
      l += sqrtn;
    }
    return a;
  }
}
```

测试结果如下

数组长度|累计绝对误差|相对于32位串行累加|相对于64位串行累加|累计相对误差|相对于32位串行累加|相对于64位串行累加
---|---|---|---|---|---|---
1,024|3.0e-3|23%|269%|3.1e-4|29%|728%
10,000|1.4e-2|11%|430%|6.2e-4|14%|1405%
1,000,000|4.2e-1|3%|1246%|5.2e-3|6%|12741%

### 方法五，二分累加

继续用分治的方法，可以把数组分成相对均匀的两部分，分别求和，并且递归地完成整个过程

代码如下

```c++
float sum(const vector<float> &s, size_t l, size_t h) {
  if (l + 1 == h)
    return s[l];
  size_t m = l + (h - l) / 2;
  return sum(s, l, m) + sum(s, m, h);
}
```

测试结果如下 

数组长度|累计绝对误差|相对于32位串行累加|相对于64位串行累加|累计相对误差|相对于32位串行累加|相对于64位串行累加
---|---|---|---|---|---|---
1,024|2.4e-3|19%|221%|4.0e-4|36%|926%
10,000|8.2e-3|6%|246%|4.9e-4|11%|1105%
1,000,000|1.0e-1|1%|292%|8.0e-4|1%|1944%

### 方法六，对折累加

方法六和方法五很像，区别在于方法五是逐层累加相邻的元素，方法六是逐层累加相距最远的元素，和把一张纸对折一样

代码如下

```c++
float sum(vector<float> &s) {
  for (int n = static_cast<int>(s.size()); n != 1; n = (n + 1) / 2) {
    for (int i = 0; i < n / 2; i++) {
      s[i] += s[n - 1 - i];
    }
  }
  return s[0];
}
```

测试结果如下

数组长度|累计绝对误差|相对于32位串行累加|相对于64位串行累加|累计相对误差|相对于32位串行累加|相对于64位串行累加
---|---|---|---|---|---|---
1,024|2.4e-3|19%|218%|2.9e-4|26%|674%
10,000|8.0e-3|6%|239%|4.3e-4|10%|980%
1,000,000|1.0e-1|1%|294%|7.5e-4|1%|1817%

可见方法五和方法六的得到的精度相当

### 方法七，先排序再对折累加

可以把方法三和方法六结合，先对数组进行排序，再用对折的方法进行累加

代码如下

```c++
float sum(vector<float> &s) {
  sort(s.begin(), s.end());
  for (int n = static_cast<int>(s.size()); n != 1; n = (n + 1) / 2) {
    for (int i = 0; i < n / 2; i++) {
      s[i] += s[n - 1 - i];
    }
  }

  return s[0];
}
```

测试结果如下

数组长度|累计绝对误差|相对于32位串行累加|相对于64位串行累加|累计相对误差|相对于32位串行累加|相对于64位串行累加
---|---|---|---|---|---|---
1,024|1.1e-3|8%|99%|4.5e-5|4%|105%
10,000|3.7e-3|3%|112%|4.7e-5|1%|107%
1,000,000|3.6e-2|0.3%|107%|4.4e-5|0.05%|107%

可见，方法七的精度和方法二（使用64位累加变量）的精度相当，也明显好于其它方法，但是方法七只使用了32位浮点数的变量

## 把算法并行化，并用GPU代码重新实现

分析比较了各种方法的优劣，那如何把它们并行化并用GPU代码实现呢

### 方法一的并行实现

方法一虽然是串行的算法，但是也有对应的GPU代码实现，可以用`atomicAdd`原语来进行累加

代码如下

```c++
__global__ void sum_kernel(const float *in, float *out, int size) {
  if (threadIdx.x == 0) {
    *out = 0.0f;
  }
  __syncthreads();
  for (int tid = threadIdx.x; tid < size; tid += blockDim.x) {
    atomicAdd(out, in[tid]);
  }
}
```

### 方法五的并行实现

方法五的并行算法，在GPU上属于一种典型的归约算法，可以借助`wrap`,`block`原语的并行归约来实现

代码如下

```c++
__inline__ __device__ float warp_reduce_sum(float value) {
  const unsigned int FINAL_MASK = 0xffffffff;
  for (int i = 1; i < 32; i <<= 1)
    value += __shfl_down_sync(FINAL_MASK, value, i);
  return value;
}

__inline__ __device__ float block_reduce_sum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warp_reduce_sum(val);

  if (lane == 0) {
    shared[wid] = val;
  }

  __syncthreads();

  if (threadIdx.x < 32) {
    val = warp_reduce_sum((threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0);
  }

  return val;
}

__global__ void hierarchy_reduce_sum_kernel(const float *in, float *out,
                                            int size) {
  const int BLOCK_SIZE = 1024;
  const int TIERS = 4; // supports up to 1024^4(1TB) array
  static __shared__ float shared[(TIERS - 1) * BLOCK_SIZE + 1];
  static __shared__ int indices[TIERS];

  if (threadIdx.x < TIERS) {
    indices[threadIdx.x] = 0;
  }

  for (int tid = 0; tid < size; tid += blockDim.x) {
    float sum = block_reduce_sum(
        tid + threadIdx.x < size ? in[tid + threadIdx.x] : 0.0f);
    if (threadIdx.x == 0) {
      shared[indices[0]++] = sum;
    }

    __syncthreads();

    for (int i = 0; i < TIERS - 1 && indices[i] == blockDim.x; i++) {
      float sum = block_reduce_sum(shared[BLOCK_SIZE * i + threadIdx.x]);
      if (threadIdx.x == 0) {
        indices[i] = 0;
        shared[BLOCK_SIZE * (i + 1) + indices[i + 1]++] = sum;
      }

      __syncthreads();
    }
  }

  for (int i = 0; i < TIERS - 1; i++) {
    float sum = block_reduce_sum(
        threadIdx.x < indices[i] ? shared[BLOCK_SIZE * i + threadIdx.x] : 0.0f);
    if (threadIdx.x == 0) {
      shared[BLOCK_SIZE * (i + 1) + indices[i + 1]++] = sum;
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out = shared[(TIERS - 1) * BLOCK_SIZE];
  }
}
```

启动kernel的代码
```c++
hierarchy_reduce_sum_kernel<<<1, 1024>>>(d_in, d_out, size);
```

当然，方法五的CPU版本和GPU版本并不完全等价，因为GPU版本的实现需要结合GPU的特点，充分利用已有的原语

### 方法七的并行实现

方法七需要先排序，这里使用了适合在GPU上使用的[双调排序](https://en.wikipedia.org/wiki/Bitonic_sorter)，再结合并行的对折合并算法

代码如下

```c++
__inline__ __device__ void swap(float *arr, int a, int b) {
  if (arr[a] > arr[b]) {
    float t = arr[a];
    arr[a] = arr[b];
    arr[b] = t;
  }
}

__inline__ __device__ void bitonic_sort_merge(float *arr, int stride, int size, bool inverted) {

  for (int tid = threadIdx.x;; tid += blockDim.x) {
    int gid = tid / stride;
    int mid = tid % stride;
    int j = gid * stride * 2;
    int k = j + stride + mid;
    if (k >= size) {
      break;
    }
    swap(arr, inverted ? j + stride - mid - 1 : j + mid, k);
  }
}

__global__ void bitnoic_sort_kernel(float *arr, int size) {
  for (int i = 1; i < size; i *= 2) {
    bitonic_sort_merge(arr, i, size, true);
    __syncthreads();
    for (int j = i / 2; j != 0; j /= 2) {
      bitonic_sort_merge(arr, j, size, false);
      __syncthreads();
    }
  }
}

__global__ void fold_reduce_sum_kernel(float *in, float *out, int size) {
  for (; size != 1; size = (size + 1) / 2) {
    for (int tid = threadIdx.x; tid < size / 2; tid += blockDim.x) {
      in[tid] += in[size - 1 - tid];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out = *in;
  }
}
```

启动kernel的代码
```c++
bitnoic_sort_kernel<<<1, 1024>>>(d_in, size);
fold_reduce_sum_kernel<<<1, 1024>>>(d_in, d_out, size);
```

## 总结

以上比较了多种方法对于浮点数累加问题的精度，同时对于部分算法进行了并行化的实现

希望这个教程能够抛砖引玉，帮助大家开拓思路