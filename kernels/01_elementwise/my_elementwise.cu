#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

// WARP_SIZE: GPU中一个warp包含的线程数，NVIDIA GPU标准为32个线程
// 一个warp是GPU调度的最小单位，所有线程在warp内同步执行
#define WARP_SIZE 32

// INT4: 4个32位整数的向量类型，总共128位(16字节)
// 构成: {x, y, z, w} 四个int32_t元素
// 用途: 用于向量化的整数运算，提高内存带宽利用率
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])

// FLOAT4: 4个32位浮点数的向量类型，总共128位(16字节)  
// 构成: {x, y, z, w} 四个float元素
// 用途: 用于向量化的单精度浮点运算，一次处理4个float值
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// HALF2: 2个16位半精度浮点数的向量类型，总共32位(4字节)
// 构成: {x, y} 两个half元素
// 用途: 用于向量化的半精度浮点运算，平衡精度和性能
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

// BFLOAT: 2个16位Brain Float的向量类型，总共32位(4字节)
// 构成: {x, y} 两个__nv_bfloat16元素
// 用途: 用于向量化的Brain Float运算，在深度学习中常用
#define BFLOAT(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

// LDST128BITS: 128位内存事务的向量类型，用于优化内存访问
// 构成: 4个float元素，总共128位(16字节)
// 用途: 用于最大化内存带宽利用率的加载/存储操作
// 
// 为什么需要128位对齐？
// 1. GPU内存事务大小: GPU通常以128位(16字节)为单位进行内存访问(硬件层面的设计)
// 2. 内存带宽优化: 128位事务能最大化内存带宽利用率
// 3. 缓存效率: 128位对齐的数据能更好地利用L1/L2缓存
// 4. 减少内存事务数量: 一次访问更多数据，减少总的内存访问次数
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        c[idx] = a[idx] + b[idx];
    }
    return;
}

// FP32x4: 4个float元素的向量化加法，实现128位内存事务优化
// 为什么需要FP32x4？
// 1. 内存对齐: 4个float = 4*32位 = 128位，正好匹配GPU的128位内存事务
// 2. 带宽优化: 一次内存访问处理4个float元素，最大化内存带宽利用率
// 3. 向量化: 利用GPU的向量化能力，提高计算效率
// 4. 缓存友好: 128位对齐的数据能更好地利用缓存行
// grid(N/256), block(256/4) - 每个线程处理4个元素，所以block大小减半
__global__ void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N){
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    
    // 检查是否完全在边界内 (处理4个元素)
    if ((idx + 3) < N) {
        // 安全情况: 所有4个元素都在边界内，可以安全加载
        float4 reg_a = FLOAT4(a[idx]);  // 一次加载4个float (128位)
        float4 reg_b = FLOAT4(b[idx]);  // 一次加载4个float (128位)
        float4 reg_c;
        
        // 执行向量化加法
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        
        FLOAT4(c[idx]) = reg_c;  // 一次存储4个float (128位)
    }
    else if (idx < N) {
        // 边界情况: 部分元素在边界内，需要逐个检查
        for (int i = 0; i < 4; i++) {
            if ((idx + i) < N) {
                c[idx + i] = a[idx + i] + b[idx + i];
            }
        }
    }
    // 如果 idx >= N，什么都不做
    return;
}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        // __hadd: CUDA内置的半精度浮点数加法函数
        // 功能: 对两个half类型数据进行加法运算
        // 参数: a[idx] - 第一个half操作数, b[idx] - 第二个half操作数
        // 返回: half类型的结果
        // 
        // 性能分析:
        // 1. 硬件支持时(SM_53+): 直接使用GPU的半精度运算单元，性能最优
        //    - 使用内联汇编: add.f16 指令
        //    - 无需类型转换，直接硬件加速
        // 
        // 2. 硬件不支持时: 转换为float进行运算，会有性能损失
        //    - half -> float 转换 (__half2float)
        //    - float 加法运算
        //    - float -> half 转换 (__float2half)
        //    - 总共需要3次转换操作，性能较差
        // 
        // 优势: 
        // 1. 硬件加速 - 在支持的GPU上直接使用半精度运算单元
        // 2. 数值稳定性 - 处理半精度浮点数的特殊值(如NaN, Inf)
        // 3. 精度保证 - 确保半精度运算的准确性
        // 4. 向后兼容 - 在不支持的GPU上仍能正常工作
        c[idx] = __hadd(a[idx], b[idx]);
    }
    return;
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N){
    int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    if(idx < N){
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        half2 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        HALF2(c[idx]) = reg_c;
    }
    return;
}

// FP16x8: 8个half元素的向量化加法，实现128位内存事务优化
// 为什么需要FP16x8？
// 1. 内存对齐: 8个half = 8*16位 = 128位，正好匹配GPU的128位内存事务
// 2. 带宽优化: 一次内存访问处理8个half元素，最大化内存带宽利用率
// 3. 计算密度: 每个线程处理更多数据，提高计算吞吐量
// 4. 缓存友好: 128位对齐的数据能更好地利用缓存行
__global__ void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N){
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    
    // 问题分析: 这里存在潜在的内存访问越界问题
    // 参考实现: 先加载数据，再检查边界 - 可能导致越界访问
    // 正确做法: 先检查边界，再安全加载数据
    
    // 检查是否完全在边界内 (处理8个元素)
    if ((idx + 7) < N) {
        // 安全情况: 所有8个元素都在边界内，可以安全加载
        half2 reg_a_0 = HALF2(a[idx + 0]);  // 加载第0,1个half
        half2 reg_a_1 = HALF2(a[idx + 2]);  // 加载第2,3个half  
        half2 reg_a_2 = HALF2(a[idx + 4]);  // 加载第4,5个half
        half2 reg_a_3 = HALF2(a[idx + 6]);  // 加载第6,7个half
        
        half2 reg_b_0 = HALF2(b[idx + 0]);
        half2 reg_b_1 = HALF2(b[idx + 2]);
        half2 reg_b_2 = HALF2(b[idx + 4]);
        half2 reg_b_3 = HALF2(b[idx + 6]);
        
        // 执行向量化加法
        half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
        reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
        reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);
        reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
        reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);
        reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
        reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);
        reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
        reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);
        
        // 安全存储所有结果
        HALF2(c[idx + 0]) = reg_c_0;
        HALF2(c[idx + 2]) = reg_c_1;
        HALF2(c[idx + 4]) = reg_c_2;
        HALF2(c[idx + 6]) = reg_c_3;
    }
    else if (idx < N) {
        // 边界情况: 部分元素在边界内，需要逐个检查
        // 逐个处理剩余的元素
        for (int i = 0; i < 8; i++) {
            if ((idx + i) < N) {
                c[idx + i] = __hadd(a[idx + i], b[idx + i]);
            }
        }
    }
}

__global__ void elementwise_add_f16x8_pack_kernel(half *a, half *b, half *c, int N){
    int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
    half pack_a[8], pack_b[8], pack_c[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

    #pragma unroll
    for(int i = 0; i < 8; i += 2){
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    if ((idx + 7) < N){
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
}

////////////////////////////////////////////////////////////////////////////////

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements) \
  void elementwise_add_##packed_type(torch::Tensor a, torch::Tensor b,         \
                                     torch::Tensor c) {                        \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(c, (th_type))                                     \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256 / (n_elements));                                          \
      dim3 grid((N + 256 - 1) / 256);                                          \
      elementwise_add_##packed_type##_kernel<<<grid, block>>>(                 \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        dim3 block(K / (n_elements));                                          \
        dim3 grid(S);                                                          \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256 / (n_elements));                                        \
        dim3 grid((N + 256 - 1) / 256);                                        \
        elementwise_add_##packed_type##_kernel<<<grid, block>>>(               \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)
TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)
TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
