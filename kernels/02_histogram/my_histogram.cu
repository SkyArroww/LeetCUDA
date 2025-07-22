#include <algorithm>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <tuple>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// Histogram - 直方图统计核函数
// grid(N/256), block(256) - 网格和块配置
// a: Nx1 - 输入数组，包含N个整数元素
// y: count histogram - 输出数组，存储每个值的出现次数
// a >= 1 - 输入数组中的值应该大于等于1
__global__ void histogram_i32_kernel(int *a, int *y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 边界检查：确保线程索引在有效范围内
    if(idx < N){
        // atomicAdd: 原子加法操作
        // 
        // 功能说明：
        // 1. 读取内存地址 y[a[idx]] 处的当前值
        // 2. 将该值加1
        // 3. 将结果写回内存地址 y[a[idx]]
        // 4. 返回原始值（这里我们不需要使用返回值）
        //
        // 为什么需要原子操作？
        // - 多个线程可能同时访问同一个内存位置 y[a[idx]]
        // - 如果没有原子操作，会出现竞态条件（race condition）
        // - 例如：两个线程同时读取值5，都加1后写回6，最终结果是6而不是7
        //
        // 原子操作保证：
        // - 读取-修改-写入操作是不可分割的
        // - 多个线程同时访问时，操作会串行执行
        // - 确保计数结果的正确性
        //
        // 性能影响：
        // - 原子操作比普通操作慢
        // - 当多个线程访问同一位置时，会产生冲突和等待
        // - 数据分布越不均匀，冲突越多，性能越差
        //
        // 参数说明：
        // - &y[a[idx]]: 指向输出数组y中索引为a[idx]的位置的指针
        // - 1: 要加的值（每次出现该元素就加1）
        atomicAdd(&y[a[idx]], 1);
    }
}

// Histogram + Vec4
// grid(N/256), block(256/4)
// a: Nx1, y: count histogram, a >= 1
__global__ void histogram_i32x4_kernel(int *a, int *y, int N){
    int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < N){
        int4 reg_a = INT4(a[idx]);
        atomicAdd(&(y[reg_a.x]), 1);
        atomicAdd(&(y[reg_a.y]), 1);
        atomicAdd(&(y[reg_a.z]), 1);
        atomicAdd(&(y[reg_a.w]), 1);
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

#define CHECK_TORCH_TENSOR_SHAPE(T, S0)                                        \
  if (((T).size(0) != (S0))) {                                                 \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define TORCH_BINDING_HIST(packed_type, th_type, element_type, n_elements)     \
  torch::Tensor histogram_##packed_type(torch::Tensor a) {                     \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);   \
    const int N = a.size(0);                                                   \
    std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0);         \
    torch::Tensor max_val = std::get<0>(max_a).cpu();                          \
    const int M = max_val.item().to<int>();                                    \
    auto y = torch::zeros({M + 1}, options);                                   \
    static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);               \
    const int NUM_BLOCKS = (N + 256 - 1) / 256;                                \
    dim3 block(NUM_THREADS_PER_BLOCK);                                         \
    dim3 grid(NUM_BLOCKS);                                                     \
    histogram_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<element_type *>(a.data_ptr()),                        \
        reinterpret_cast<element_type *>(y.data_ptr()), N);                    \
    return y;                                                                  \
  }

TORCH_BINDING_HIST(i32, torch::kInt32, int, 1)
TORCH_BINDING_HIST(i32x4, torch::kInt32, int, 4)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32)
  TORCH_BINDING_COMMON_EXTENSION(histogram_i32x4)
}