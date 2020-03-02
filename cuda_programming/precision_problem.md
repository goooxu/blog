# 精度问题

这一讲我们来关注一个在深度学习框架中很普遍的问题，精度问题

看下面这个问题

> 对32位符点数的数组求和，误差越低越好

首先，我们需要一个基准，我们使用了任意精度计算库[MPFR](https://www.mpfr.org/)来得到一个基准答案，代码如下：

```c++
float accumulate(const vector<float> &seq) {
    mpfr_t sum, t;
    mpfr_init2(sum, 1024);
    mpfr_init2(t, 1024);
    mpfr_set_flt(sum, 0.0f, MPFR_RNDD);
    for (auto i : seq) {
        mpfr_set_flt(t, i, MPFR_RNDD);
        mpfr_add(sum, sum, t, MPFR_RNDD);
    }

    float result = mpfr_get_flt(sum, MPFR_RNDD);
    mpfr_clear(sum);
    mpfr_clear(t);
    return result;
}
```

其中累加变量`sum`使用1024位的精度

现在我们来比较各种方法的优劣

## 方法一，32位串行累加

有代码

```c++
template <typename T> 
float accumulate(const vector<T> &s) {
  T a = T(0);
  for (const T &e : s)
    a += e;
  return a;
}
```

测试结果如下

数组长度|累计绝对误差|累计相对误差
---|---|---
1,024|0.013122|0.001410
10,000|0.127352|0.004792
1,000,000|13.021620|0.043306

> 累计的意思是指1000次测试的误差之和，下同

## 方法二，累加变量升级到64位

把方法一中的32位累加变量改成64位，可以得到好的多的精度误差，测试结果如下


数组长度|累计绝对误差|累计相对误差
---|---|---
1,024|0.001089|0.000042
10,000|0.003819|0.000043
1,000,000|0.036608|0.000043

## 方法三，先排序再累加

虽然使用64位累加变量可以得到好的多的精度误差，但是我还是想回到32位的范畴内讨论这个问题

产生误差的原因是由于符点数的二进制结构导致的，当一个比较大的符点数与一个比较小的符点数相加时，由于他们表示小数部分的位数不同，所以会产生误差，当多个符点数累加的时候，这个误差会被放大。

那可以想到一个方法，我们对原始数组按绝对值从小到大排序，累加变量开始累加绝对值输较小的数，然后逐渐累加经验值较大的数

代码如下

```c++
template<typename T>
float accumulate(vector<T> &seq) {
  sort(sc.begin(), sc.end(), [](float a, float b) { return abs(a) < abs(b); });
  T a = T(0);
  for (const T &e : s) a += e;
  return a;
}
```

测试结果如下

数组长度|累计绝对误差|累计相对误差
---|---|---
1,024|0.008213|0.000890
10,000|0.084594|0.003569
1,000,000|7.718806|0.036913

## 方法四，按平方根分区累加

虽然排序再串行累加的方法比起直接串行累加的方法精度上好了不少，但还不够好

那我想到，可以把原始数组按长度的平方根分区，先计算每个区的和，再累加，当然每个区使用同样的方法，再分成更小的区

代码如下

```c++
template <typename T>
float accumulate(const vector<T> &s, size_t l, size_t h) {
  size_t n = h - l;
  if (n == 1) {
    return s[l];
  } else if (n == 2) {
    return s[l] + s[l + 1];
  } else {
    size_t sqrtn = ceil(sqrt(n));
    float sum = 0.0f;
    while (l < h) {
      sum += square_root_merge_accumulate(s, l, min(l + sqrtn, h));
      l += sqrtn;
    }
    return sum;
  }
}
```

测试结果如下

数组长度|累计绝对误差|累计相对误差
---|---|---
1,024|0.003298|0.000389
10,000|0.015160|0.000618
1,000,000|0.438302|0.001552