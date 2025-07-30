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

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

/*
对于输入索引 idx[i]，输出: output[i, :] = weight[idx[i], :]

数据流详解：
输入：idx[N] - N个token ID (每个都是标量，范围[0, vocab_size-1])
输出：output[N, emb_size] - N个embedding向量，每个向量维度为[1, emb_size]

例如：
- 输入序列：idx = [10, 25, 100] (3个token ID)
- 输出：output[3, 512] (假设emb_size=512)
  - output[0, :] = weight[10, :] (第10个token的embedding向量)
  - output[1, :] = weight[25, :] (第25个token的embedding向量) 
  - output[2, :] = weight[100, :] (第100个token的embedding向量)
*/

// FP32
__global__ void embedding_f32_kernel(const int *idx, float *weight, float *output, int n, int emb_size){
    // 线程索引计算
    int tx = threadIdx.x;  // 当前线程在block内的ID (0 到 emb_size-1)
    int bx = blockIdx.x;   // 当前block的ID (0 到 n-1，对应sequence中的position)
    int tid = bx * blockDim.x + tx;  // 全局线程ID (用于调试，此kernel中未使用)

    /*
      * 索引计算详解：
      * 
      * 1. 线程组织方式：
      *    - Grid维度：(n, 1, 1) - n个block，每个block处理一个sequence position
      *    - Block维度：(emb_size, 1, 1) - 每个block有emb_size个线程，每个线程处理embedding向量的一个元素
      * 
      * 2. 输入数据布局：
      *    - idx[n]: 长度为n的索引数组，idx[i]表示第i个position对应的token ID
      *    - weight[vocab_size, emb_size]: embedding权重矩阵，按行主序存储
      *    - output[n, emb_size]: 输出矩阵，按行主序存储
      * 
      * 3. 为什么用idx[bx]？
      *    - bx表示当前处理的sequence position (0到n-1)
      *    - idx[bx]获取该position对应的token ID
      *    - 例如：bx=0时，idx[0]是第一个token的ID；bx=1时，idx[1]是第二个token的ID
      */

    // 计算当前token在weight矩阵中的起始行偏移
    int offset = idx[bx] * emb_size;
    /*
      * offset计算原理：
      * - idx[bx]是token ID，范围[0, vocab_size-1]
      * - weight矩阵按行存储：weight[token_id][0], weight[token_id][1], ..., weight[token_id][emb_size-1]
      * - 第token_id行的起始地址 = token_id * emb_size
      * - 所以offset = idx[bx] * emb_size 就是该token对应embedding向量的起始位置
      * 
      * 例如：emb_size=512, token_id=100
      * - 该token的embedding向量在weight数组中的位置是[51200, 51711] (100*512 到 100*512+511)
      */

    // 执行embedding查找：从weight矩阵复制到output矩阵  
    output[bx * emb_size + tx] = weight[offset + tx];
    /*
      * 内存访问模式详解：
      * 
      * 读取 (weight[offset + tx])：
      * - offset: 当前token在weight矩阵中的行起始位置
      * - tx: 当前线程处理的embedding维度 (0到emb_size-1)
      * - weight[offset + tx]: 读取该token embedding向量的第tx个元素
      * 
      * 写入 (output[bx * emb_size + tx])：
      * - bx * emb_size: 输出矩阵中第bx行的起始位置
      * - tx: 该行内的列偏移
      * - output[bx * emb_size + tx]: 写入输出矩阵第bx行第tx列
      * 
      * 内存访问特点：
      * - 同一block内的线程访问连续的内存地址 (合并访问)
      * - 读取：weight[offset], weight[offset+1], ..., weight[offset+emb_size-1]
      * - 写入：output[bx*emb_size], output[bx*emb_size+1], ..., output[bx*emb_size+emb_size-1]
      * - 这种模式最大化内存带宽利用率
      */
}

__global__ void embedding_f32x4_kernel(const int *idx, float *weight, float *output, int n, int emb_size){
  int tx = 4 * threadIdx.x, bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  if(tx + 3 < emb_size){
      output[bx * emb_size + tx + 0] = weight[offset + tx + 0];
      output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
      output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
      output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
  }
}

__global__ void embedding_f32x4_pack_kernel(const int *idx, float *weight, float *output, int n, int emb_size){
  int tx = 4 * threadIdx.x, bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  LDST128BITS(output[bx * emb_size + tx]) = LDST128BITS(weight[offset + tx]);
}

__global__ void embedding_f16_kernel(const int *idx, half *weight, half *output, int n, int emb_size){
  int tx = threadIdx.x, bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f16x8_kernel(const int *idx, half *weight,
                                       half *output, int n, int emb_size) {
  int tx = threadIdx.x * 8;
  int bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  output[bx * emb_size + tx] = weight[offset + tx];
  output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
  output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
  output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
  output[bx * emb_size + tx + 4] = weight[offset + tx + 4];
  output[bx * emb_size + tx + 5] = weight[offset + tx + 5];
  output[bx * emb_size + tx + 6] = weight[offset + tx + 6];
  output[bx * emb_size + tx + 7] = weight[offset + tx + 7];
}

__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight, half *output, int n, int emb_size){
  int tx = 8 * threadIdx.x, bx = blockIdx.x;
  int offset = idx[bx] * emb_size;
  LDST128BITS(output[bx * emb_size + tx]) = LDST128BITS(weight[offset + tx]);
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define TORCH_BINDING_EMBEDDING(packed_type, th_type, element_type,            \
                                n_elements)                                    \
  void embedding_##packed_type(torch::Tensor a, torch::Tensor weight,          \
                               torch::Tensor o) {                              \
    CHECK_TORCH_TENSOR_DTYPE(a, (torch::kInt32));                              \
    CHECK_TORCH_TENSOR_DTYPE(weight, (th_type));                               \
    CHECK_TORCH_TENSOR_DTYPE(o, (th_type));                                    \
                                                                               \
    const int N = a.size(0);                                                   \
    const int emb_size = weight.size(1);                                       \
    dim3 block(emb_size / n_elements);                                         \
    dim3 grid(N);                                                              \
    embedding_##packed_type##_kernel<<<grid, block>>>(                         \
        reinterpret_cast<int *>(a.data_ptr()),                                 \
        reinterpret_cast<element_type *>(weight.data_ptr()),                   \
        reinterpret_cast<element_type *>(o.data_ptr()), N, emb_size);          \
  }

TORCH_BINDING_EMBEDDING(f32, torch::kFloat32, float, 1)
TORCH_BINDING_EMBEDDING(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f32x4_pack, torch::kFloat32, float, 4)
TORCH_BINDING_EMBEDDING(f16, torch::kHalf, half, 1)
TORCH_BINDING_EMBEDDING(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_EMBEDDING(f16x8_pack, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f32x4_pack);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8);
  TORCH_BINDING_COMMON_EXTENSION(embedding_f16x8_pack);
}
