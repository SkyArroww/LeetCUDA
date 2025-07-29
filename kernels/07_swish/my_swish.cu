#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

/*
 * CUDA Half精度数学函数说明：
 * 
 * CUDA提供了专门的half精度数学函数，这些函数直接在硬件层面支持，
 * 相比先转换为float再计算更高效：
 * 
 * 基础运算：
 * - __hadd(a, b)     : half精度加法 a + b
 * - __hsub(a, b)     : half精度减法 a - b  
 * - __hmul(a, b)     : half精度乘法 a * b
 * - __hdiv(a, b)     : half精度除法 a / b
 * - __hneg(a)        : half精度取反 -a
 * 
 * 数学函数：
 * - hexp(a)          : half精度指数函数 exp(a)
 * - hlog(a)          : half精度对数函数 log(a)
 * - hsqrt(a)         : half精度平方根函数 sqrt(a)
 * 
 * 类型转换：
 * - __float2half(f)  : float转half
 * - __half2float(h)  : half转float
 * 
 * 使用这些原生half函数的优势：
 * 1. 避免不必要的精度转换开销
 * 2. 充分利用现代GPU的half精度计算单元
 * 3. 减少寄存器使用和内存带宽需求
 */

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// Swish x: N, y: N y=x*sigmoid(x)

// FP32
__device__ __forceinline__ float swish(float x){
    return x / (1.0f + expf(-x));
}

__global__ void swish_f32_kernel(float *x, float *y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        y[idx] = swish(x[idx]);
    }
}

__global__ void swish_f32x4_kernel(float *x, float *y, int N){
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if(idx < N){
        float4 reg_x = FLOAT4(x[idx]), reg_y;
        reg_y.x = swish(reg_x.x);
        reg_y.y = swish(reg_x.y);
        reg_y.z = swish(reg_x.z);
        reg_y.w = swish(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// FP16
// Swish函数的数学推导和half精度实现解析：
//
// 1. Swish函数定义：swish(x) = x * sigmoid(x)
// 2. Sigmoid函数定义：sigmoid(x) = 1 / (1 + exp(-x))
// 3. 合并得到：swish(x) = x * (1 / (1 + exp(-x))) = x / (1 + exp(-x))
//
// Half精度CUDA实现分解：
// - __hneg(x)                    : 计算 -x (half精度取反)
// - hexp(__hneg(x))             : 计算 exp(-x) (half精度指数函数)
// - __float2half(1.0f)          : 将常数1.0转换为half精度
// - __hadd(1.0_half, exp(-x))   : 计算 1.0 + exp(-x) (half精度加法)
// - __hdiv(1.0_half, 1+exp(-x)) : 计算 1.0 / (1.0 + exp(-x)) (half精度除法)
// - __hmul(x, 1/(1+exp(-x)))    : 计算 x * (1/(1+exp(-x))) (half精度乘法)
//
// 最终实现：swish(x) = x / (1 + exp(-x))
__device__ __forceinline__ half swish_half(half x){
      return __hmul(x,                                    // x * 
                    __hdiv(__float2half(1.0f),            // 1.0 / 
                           __hadd(__float2half(1.0f),     // (1.0 + 
                                  hexp(__hneg(x)))));     //  exp(-x))
      // 分步解析：
      // 1. __hneg(x)           -> -x
      // 2. hexp(...)           -> exp(-x)  
      // 3. __float2half(1.0f)  -> 1.0 (half精度)
      // 4. __hadd(1.0, exp(-x)) -> 1.0 + exp(-x)
      // 5. __hdiv(1.0, ...)     -> 1.0 / (1.0 + exp(-x)) = sigmoid(x)
      // 6. __hmul(x, sigmoid(x)) -> x * sigmoid(x) = swish(x)
}

__global__ void swish_f16_kernel(half *x, half *y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        x[idx] = swish_half(x[idx]);
    }
}

__global__ void swish_f16x2_kernel(half *x, half *y, int N){
    int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    if(idx < N){
        half2 reg_x = HALF2(x[idx]), reg_y;
        reg_y.x = swish_half(reg_x.x);
        reg_y.y = swish_half(reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

__global__ void swish_f16x8_kernel(half *x, half *y, int N){
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    half2 reg_x_0 = HALF2(x[idx + 0]);
  half2 reg_x_1 = HALF2(x[idx + 2]);
  half2 reg_x_2 = HALF2(x[idx + 4]);
  half2 reg_x_3 = HALF2(x[idx + 6]);
  half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
  reg_y_0.x = swish_half(reg_x_0.x);
  reg_y_0.y = swish_half(reg_x_0.y);
  reg_y_1.x = swish_half(reg_x_1.x);
  reg_y_1.y = swish_half(reg_x_1.y);
  reg_y_2.x = swish_half(reg_x_2.x);
  reg_y_2.y = swish_half(reg_x_2.y);
  reg_y_3.x = swish_half(reg_x_3.x);
  reg_y_3.y = swish_half(reg_x_3.y);
  if ((idx + 0) < N) {
    HALF2(y[idx + 0]) = reg_y_0;
  }
  if ((idx + 2) < N) {
    HALF2(y[idx + 2]) = reg_y_1;
  }
  if ((idx + 4) < N) {
    HALF2(y[idx + 4]) = reg_y_2;
  }
  if ((idx + 6) < N) {
    HALF2(y[idx + 6]) = reg_y_3;
  }
}

__global__ void swish_f16x8_pack_kernel(half *x, half *y, int N){
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    if(idx + 7 < N){
        half pack_x[8], pack_y[8];
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
        
        #pragma unroll
        for(int i = 0; i < 8; i++){
            pack_y[i] = swish_half(pack_x[i]);
        }
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}


#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_SWISH(packed_type, th_type, element_type, n_elements)    \
  void swish_##packed_type(torch::Tensor x, torch::Tensor y) {                 \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      swish_##packed_type##_kernel<<<grid, block>>>(                           \
          reinterpret_cast<element_type *>(x.data_ptr()),                      \
          reinterpret_cast<element_type *>(y.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        swish_##packed_type##_kernel<<<grid, block>>>(                         \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        swish_##packed_type##_kernel<<<grid, block>>>(                         \
            reinterpret_cast<element_type *>(x.data_ptr()),                    \
            reinterpret_cast<element_type *>(y.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_SWISH(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SWISH(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SWISH(f16, torch::kHalf, half, 1)
TORCH_BINDING_SWISH(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SWISH(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SWISH(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(swish_f32)
  TORCH_BINDING_COMMON_EXTENSION(swish_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(swish_f16x8_pack)
}
