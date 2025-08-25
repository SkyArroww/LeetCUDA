# NMS

## 0x00 说明

包含以下内容：

- [X] nms_kernel(CPU/GPU)
- [X] PyTorch bindings

nms cuda实现是最基础的版本，根据[官方源码](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu)可以进行进一步优化。

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 nms.py
```

输出:

```bash
-------------------------------------------------------------------------------------
                                        nboxes=1024
       out_nms: ['1021 ', '1022 ', '1023 '], len of keep: 950, time:0.26456594ms
    out_nms_th: ['1021 ', '1022 ', '1023 '], len of keep: 950, time:0.19218683ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        nboxes=2048
       out_nms: ['2045 ', '2046 ', '2047 '], len of keep: 1838, time:0.47256470ms
    out_nms_th: ['2044 ', '2045 ', '2047 '], len of keep: 1838, time:0.39437532ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        nboxes=4096
       out_nms: ['4092 ', '4093 ', '4095 '], len of keep: 3598, time:0.89909315ms
    out_nms_th: ['4093 ', '4094 ', '4095 '], len of keep: 3598, time:1.03515625ms
-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
                                        nboxes=8192
       out_nms: ['8189 ', '8190 ', '8191 '], len of keep: 7023, time:1.49935722ms
    out_nms_th: ['8189 ', '8190 ', '8191 '], len of keep: 7023, time:3.39094877ms
-------------------------------------------------------------------------------------
```

## 算法原理详解

### NMS (Non-Maximum Suppression) 算法原理

NMS算法是目标检测中的关键后处理步骤，主要解决检测过程中产生的重叠边界框问题。其核心思想是保留置信度最高的检测框，同时抑制与其高度重叠的其他检测框。

#### 数学原理

**1. IoU (Intersection over Union) 计算**

给定两个边界框 $A$ 和 $B$，其IoU定义为：

$$\text{IoU}(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}$$

其中交集区域的计算为：
- $x_1^{inter} = \max(x_1^A, x_1^B)$
- $y_1^{inter} = \max(y_1^A, y_1^B)$ 
- $x_2^{inter} = \min(x_2^A, x_2^B)$
- $y_2^{inter} = \min(y_2^A, y_2^B)$
- $\text{Area}_{inter} = \max(0, x_2^{inter} - x_1^{inter}) \times \max(0, y_2^{inter} - y_1^{inter})$

**2. NMS算法流程**

1. **排序**: 将所有检测框按置信度分数降序排列
2. **初始化**: 创建保留框列表 $\mathcal{K} = \emptyset$
3. **迭代抑制**:
   - 选择剩余框中置信度最高的框 $b_i$
   - 将 $b_i$ 加入保留列表: $\mathcal{K} = \mathcal{K} \cup \{b_i\}$
   - 移除所有与 $b_i$ 的IoU超过阈值 $\tau$ 的框：
     $$\mathcal{R} = \mathcal{R} \setminus \{b_j | \text{IoU}(b_i, b_j) > \tau\}$$
4. **重复**直到 $\mathcal{R} = \emptyset$

#### 算法复杂度

- **时间复杂度**: $O(N^2)$，其中 $N$ 是检测框数量
- **空间复杂度**: $O(N)$

### Kernel实现分析与优化策略

#### 1. `nms_kernel` - 基础并行化实现

**实现特点:**
```cpp
__global__ void nms_kernel(const float *boxes, const float *scores, int *keep,
                           int num_boxes, float iou_threshold)
```

**优化策略:**

**a) 线程并行化**
- 每个线程处理一个边界框
- 线程ID映射: `idx = blockId * threadsPerBlock + threadId`
- 避免了串行处理的性能瓶颈

**b) 预排序优化**
```cpp
// 主机端预排序，避免GPU端复杂排序操作
auto order_t = std::get<1>(scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
auto boxes_sorted = boxes.index_select(0, order_t).contiguous();
```

**c) 内存访问优化**
- 使用连续内存布局存储边界框坐标
- 线程内局部变量存储当前框坐标，减少重复访存：
```cpp
float x1 = boxes[idx * 4 + 0];  // 一次性加载当前框坐标
float y1 = boxes[idx * 4 + 1];
float x2 = boxes[idx * 4 + 2]; 
float y2 = boxes[idx * 4 + 3];
```

**d) 早期退出优化**
```cpp
if (iou > iou_threshold) {
    keep[idx] = 0;  // 发现抑制条件立即退出
    return;
}
```

**e) 分支减少**
- 通过 `if (keep[i] == 0) continue;` 跳过已被抑制的框
- 减少不必要的IoU计算

#### 2. 性能特征分析

**优势:**
- **并行度高**: 所有线程同时工作，充分利用GPU并行计算能力
- **内存效率**: 预排序减少GPU内存操作复杂度
- **计算密集**: IoU计算完全在GPU上进行

**局限性:**
- **负载不均衡**: 不同线程的循环次数差异很大（索引越大循环越多）
- **内存访问模式**: 存在一定的内存访问不规律性
- **竞争条件**: 多线程同时访问 `keep` 数组可能存在竞争

#### 3. 与官方实现对比

根据README中的性能测试结果，在某些情况下自实现版本性能优于PyTorch官方实现：

- **nboxes=8192**: 自实现 1.499ms vs 官方 3.391ms
- 这主要得益于简化的实现逻辑和针对性的优化

#### 4. 进一步优化方向

**a) 块级并行优化**
- 使用共享内存缓存频繁访问的边界框数据
- 块内线程协作减少全局内存访问

**b) Warp级优化**  
- 利用warp内线程同步特性优化分支处理
- 使用warp-level primitives提高效率

**c) 内存合并访问**
- 优化内存访问模式提高带宽利用率
- 使用向量化内存操作（如 `float4`）

**d) 动态并行**
- 对于大规模数据，可考虑使用CUDA动态并行特性
- 减少主机-设备同步开销

这个实现展示了NMS算法从CPU到GPU的基本并行化思路，虽然相对简单但在特定场景下已能获得不错的性能表现。
