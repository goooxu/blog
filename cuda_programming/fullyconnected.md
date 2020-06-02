## 全连接层的实现（上）

这一讲我们来讨论全连接层的实现

先看看什么是全连接层

![image](https://user-images.githubusercontent.com/22703054/83033499-c5470e80-a069-11ea-85e1-f22a1a6e4e86.png)

全连接层有如下公式

前向
![image](https://user-images.githubusercontent.com/22703054/83034145-81083e00-a06a-11ea-8dcd-5485f460b9b0.png)

后向
![image](https://user-images.githubusercontent.com/22703054/83034223-9aa98580-a06a-11ea-895c-adf641933ae1.png)

其中
* X是输入，维度m×k
* Y是输出，维度m×n
* W和b是参数，维度分别是k×n和1×n

这一讲通过上下两篇分别讨论前向和后向的实现方法

### 前向

#### 方法一

最直观的方法，就是使用cublas的矩阵乘法(GEMM)API来实现`XW`，使用自定义Kernel来实现`+b`

注意，这里使用的矩阵都是列优先(Column Major)的

代码如下
```C++
__global__ void addBiasKernel(float *top, const float *bias, int m) {
  int offset = blockIdx.x * m;
  float b = __ldg(bias + blockIdx.x);
  for (int tid = threadIdx.x; tid < m; tid += blockDim.x) {
    top[offset + tid] += b;
  }
}

class FullyConnectedForwardOp {
public:
  void operator()(const Tensor &W_tensor, const Tensor &B_tensor,
                  const Tensor &X_tensor, Tensor &Y_tensor,
                  const Device &device) {

    int m = X_tensor.dim(0);
    int n = W_tensor.dim(1);
    int k = W_tensor.dim(0);

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasGemmEx(
        device.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        X_tensor.data_ptr<float>(), CUDA_R_32F, m, W_tensor.data_ptr<float>(),
        CUDA_R_32F, k, &beta, Y_tensor.data_ptr<float>(), CUDA_R_32F, m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

    addBiasKernel<<<n, min(m, 1024), 0, device.stream()>>>(
        Y_tensor.data_ptr<float>(), B_tensor.data_ptr<float>(), m);
  }
};
```

关于`cublasGemmEx`函数的参数说明可以参考[这里](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx)

测试结果（100次平均值）

维度|V100|A100
---|---|---
m=8192, n=1024, k=784|1108μs|770μs
m=8192, n=1024, k=1024|1366μs|990μs
m=8192, n=512, k=1024|676μs|508μs
m=8192, n=256, k=512|195μs|160μs
m=8192, n=1, k=256|22μs|34μs
总计|3367μs|2462μs

#### 方法二


首先想到的优化方法是降低计算的精度，也就是使用float16，实验证明，降低计算精度对最后的准确度的影响微乎其微，但对性能的提升巨大。

也就是将方法一中的`float`类型都改为`half`

代码如下 
```C++
__global__ void halfAddBiasKernel(half *top, const half *bias, int m) {
  int offset = blockIdx.x * m;
  half b = __ldg(bias + blockIdx.x);
  for (int tid = threadIdx.x; tid < m; tid += blockDim.x) {
    top[offset + tid] = __hadd(top[offset + tid], b);
  }
}

class FullyConnectedForwardOp {
public:
  void operator()(const Tensor &W_tensor, const Tensor &B_tensor,
                  const Tensor &X_tensor, Tensor &Y_tensor,
                  const Device &device) {
    int m = X_tensor.dim(0);
    int n = W_tensor.dim(1);
    int k = W_tensor.dim(0);

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasGemmEx(
        device.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        X_tensor.data_ptr<half>(), CUDA_R_16F, m, W_tensor.data_ptr<half>(),
        CUDA_R_16F, k, &beta, Y_tensor.data_ptr<half>(), CUDA_R_16F, m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    halfAddBiasKernel<<<n, min(m, 1024), 0, device.stream()>>>(
        Y_tensor.data_ptr<half>(), B_tensor.data_ptr<half>(), m);
  }
};
```

注意，这里`cublasGemmEx`使用的算法由`CUBLAS_GEMM_DEFAULT`变成了`CUBLAS_GEMM_DEFAULT_TENSOR_OP`，也就是在尝试用Tensor Core进行加速


测试结果（100次平均值）

维度|V100|A100
---|---|---
m=8192, n=1024, k=784|248μs|91μs
m=8192, n=1024, k=1024|290μs|105μs
m=8192, n=512, k=1024|148μs|61μs
m=8192, n=256, k=512|51μs|27μs
m=8192, n=1, k=256|9μs|13μs
总计|746μs|297μs

可见使用float16精度进行计算之后，所需时间只是float32精度的1/4和1/8左右，在A100上的优势更加明显

#### 方法三

`half`类型的浮点数是16bit宽，但CUDA提供了并行指令，可以同时处理两个`half`类型的变量

代码如下
```C++
__global__ void half2AddBiasKernel(half *top, const half *bias, int m) {
  half2 *top2 = reinterpret_cast<half2 *>(top);

  int offset = blockIdx.x * m;
  half2 b2 = __half2half2(__ldg(bias + blockIdx.x));

  for (int tid = threadIdx.x; tid < m; tid += blockDim.x) {
    top2[offset + tid] = __hadd2(top2[offset + tid], b2);
  }
}

class FullyConnectedForwardOp {
public:
  void operator()(const Tensor &W_tensor, const Tensor &B_tensor,
                  const Tensor &X_tensor, Tensor &Y_tensor,
                  const Device &device) {
    int m = X_tensor.dim(0);
    int n = W_tensor.dim(1);
    int k = W_tensor.dim(0);

    float alpha = 1.0f, beta = 0.0f;
    checkCudaErrors(cublasGemmEx(
        device.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        X_tensor.data_ptr<half>(), CUDA_R_16F, m, W_tensor.data_ptr<half>(),
        CUDA_R_16F, k, &beta, Y_tensor.data_ptr<half>(), CUDA_R_16F, m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    half2AddBiasKernel<<<n, min(m / 2, 1024), 0, device.stream()>>>(
        Y_tensor.data_ptr<half>(), B_tensor.data_ptr<half>(), m / 2);
  }
};

```

其中使用了`half2`类型和`__hadd2`指令，`__hadd2`指令可以同时对两个`half`类型的数做加法

测试结果（100次平均值）

维度|V100|A100
---|---|---
m=8192, n=1024, k=784|229μs|79μs
m=8192, n=1024, k=1024|272μs|93μs
m=8192, n=512, k=1024|138μs|57μs
m=8192, n=256, k=512|45μs|25μs
m=8192, n=1, k=256|8μs|13μs
总计|692μs|267μs

可见，使用`half2`类型，其性能比使用`half`类型又提高了不少

### 方法四

`+b`可以不使用自定义的Kernel，而是使用矩阵乘法(GEMM)来代替，方法如图所示

![image](https://user-images.githubusercontent.com/22703054/83492938-9eb02a00-a4e6-11ea-9914-59eb3dec350f.png)

因为cublas的GEMM API已经做过充分优化，可以认为在各种情况下都是很快的

代码如下
```C++
class FullyConnectedForwardOp {
public:
  void operator()(const Tensor &W_tensor, const Tensor &B_tensor,
                  const Tensor &X_tensor, const Tensor &ONES_tensor,
                  Tensor &Y_tensor, const Device &device) {

    int m = X_tensor.dim(0);
    int n = W_tensor.dim(1);
    int k = W_tensor.dim(0);

    float alpha = 1.0f, beta1 = 0.0f, beta2 = 1.0f;

    checkCudaErrors(cublasGemmEx(
        device.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, 1, &alpha,
        ONES_tensor.data_ptr<half>(), CUDA_R_16F, m, B_tensor.data_ptr<half>(),
        CUDA_R_16F, 1, &beta1, Y_tensor.data_ptr<half>(), CUDA_R_16F, m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    checkCudaErrors(cublasGemmEx(
        device.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        X_tensor.data_ptr<half>(), CUDA_R_16F, m, W_tensor.data_ptr<half>(),
        CUDA_R_16F, k, &beta2, Y_tensor.data_ptr<half>(), CUDA_R_16F, m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
};
```

这里需要预构建一个`ONES_tensor`的矩阵，它的维度是`mx1`，值都是`1`

测试结果（100次平均值）

维度|V100|A100
---|---|---
m=8192, n=1024, k=784|234μs|91μs
m=8192, n=1024, k=1024|270μs|104μs
m=8192, n=512, k=1024|132μs|66μs
m=8192, n=256, k=512|45μs|30μs
m=8192, n=1, k=256|9μs|14μs
总计|690μs|305μs

### 方法五

在方法四的基础上，两个矩阵乘法(GEMM)其实可以合并成同一个，如图所示

![image](https://user-images.githubusercontent.com/22703054/83493207-02d2ee00-a4e7-11ea-8ae4-a4a82ea57e78.png)

代码如下
```C++
__global__ void extendKernel(half *out, const half *in, int m, int k) {
  int offset = blockIdx.x * m;
  for (int tid = threadIdx.x; tid < m; tid += blockDim.x) {
    half val;
    if (blockIdx.x < k)
      val = in[offset + tid];
    else if (blockIdx.x == k)
      val = __float2half(1.0f);
    else
      val = __float2half(0.0f);

    out[offset + tid] = val;
  }
}

class FullyConnectedForwardOp {
public:
  void operator()(const Tensor &W_extend_tensor, const Tensor &X_tensor,
                  Tensor &X_extend_tensor, Tensor &Y_tensor,
                  const Device &device) {

    int m = X_tensor.dim(0);
    int n = W_extend_tensor.dim(1);
    int k = W_extend_tensor.dim(0);

    extendKernel<<<X_extend_tensor.dim(1), 1024, 0, device.stream()>>>(
        X_extend_tensor.data_ptr<half>(), X_tensor.data_ptr<half>(),
        X_tensor.dim(0), X_tensor.dim(1));

    float alpha = 1.0f, beta = 0.0f;

    checkCudaErrors(
        cublasGemmEx(device.cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                     &alpha, X_extend_tensor.data_ptr<half>(), CUDA_R_16F, m,
                     W_extend_tensor.data_ptr<half>(), CUDA_R_16F, k, &beta,
                     Y_tensor.data_ptr<half>(), CUDA_R_16F, m, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
};
```

首先需要使用`extendKernel`对`X_tensor`进行扩展，添加全为`1`的一列，使其与扩展的`W_tensor`中的`b`的一行相乘，来达到`+b`的目的

测试结果（100次平均值）

维度|V100|A100
---|---|---
m=8192, n=1024, k=784|225μs|98μs
m=8192, n=1024, k=1024|285μs|123μs
m=8192, n=512, k=1024|171μs|88μs
m=8192, n=256, k=512|66μs|29μs
m=8192, n=1, k=256|24μs|14μs
总计|771μs|352μs

上篇中主要探讨了几个全连接层前向的实现方法，下篇中我们将继续对后向的实现方法进行探讨