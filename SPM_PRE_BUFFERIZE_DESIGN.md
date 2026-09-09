# Bufferize 之前的 IR 形态：分层 tiling 与 double-buffer software pipeline 设计

本文定义 `SPM_SRAM_SPILLING_DESIGN.md` 的**输入**：进入 One-Shot Bufferize 之前，IR 必须长成什么样，
以及产生这个形态的两个算法——**tiling** 和 **software pipeline**，各自为 L2 和 L1 实例化一次。

`SPM_SRAM_SPILLING_DESIGN.md`（下称**下游文档**）已经冻结，本文**不修改**它，只引用和细化。
唯一一处需要修正的地方（下游文档 §5.3 推荐的 ring 表示）在本文 §8.2 单独列出，并附实测证据。

本文所有关键 IR 断言都用本仓库的 `build/bin/mlir-opt`（LLVM 24.0.0git，assertions on）实测过；
实验清单、命令和结论见 §9。凡标注「**实测**」的结论都可复现，凡标注「读码」的是从源码推断。

---

## 0. 前置上下文

### 0.1 目标硬件的内存层级

| 层级 | 性质                                                   |
| ---- | ------------------------------------------------------ |
| DRAM | 外部内存                                               |
| L2   | software-managed SRAM                                  |
| L1   | software-managed SRAM，硬件 compute 唯一可直接读写的层 |

三条硬件事实决定了本文的全部结构：

1. **没有硬件 cache。** L1/L2 的每一个字节由编译器决定放什么，地址在编译期全静态分配。
2. **只有相邻层可以搬运**：DRAM ↔ L2 ↔ L1。DRAM ↔ L1 直连在硬件上不存在。
3. **compute 只碰 L1**：任何计算 op 的 operand 和 `outs` 最终必须由 L1 allocation 支撑。

NPU 的计算架构（ISA、阵列形状、向量单元）与本文无关，不需要知道。

### 0.2 本文负责什么，不负责什么

| 事项                                                                                            | 归属                                                   |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| L2 tiling / L1 tiling 的算法骨架、产出 IR、后置条件                                             | **本文 §3**                                     |
| 每个算子的 tile size 动态决策                                                                   | **外部已解决的 oracle**，本文只定义接口（§3.1） |
| 把多个算子融进一个 tiled loop（tile-and-fuse）的融合决策                                        | **外部已解决的 oracle**，本文只定义接口（§3.1） |
| software pipeline / double buffer 的完整算法                                                    | **本文 §4、§5**（本文的重点）                  |
| pre-bufferize IR 契约与 verifier                                                                | **本文 §7**                                     |
| One-Shot Bufferize 契约、per-space arena、静态 offset、spilling                                 | 下游文档 §7–§9                                      |
| tensor placement normal form（`alloc_tensor(memory_space)` / `materialize_in_destination`） | 下游文档 §2、§3                                      |

### 0.3 两个「已解决」的 oracle

用户已有这两个算法，本文只把它们当作**保证成功的黑盒**，并把它们钉在 MLIR 现成的钩子上：

```text
TileSizeOracle(clusterRoot, tier, budgetBytes) -> tileSizes[]
  语义：在 budgetBytes 之内，返回让本层 SRAM 尽量吃满、重复加载尽量少的 tile size。
  前提：只要存在可行 tile size，它一定能找到。
  落点：scf::SCFTilingOptions::tileSizeComputationFunction
        （mlir/include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h:60）

FusionOracle(sliceOp, producerResult, tier) -> {fuse?, yieldProducerReplacement?}
  语义：决定某个 producer 是否融进当前 tiled loop 以减少 memory traffic。
  落点：scf::SCFTileAndFuseOptions::fusionControlFn（同上 :301-:311）
```

两个 oracle 的返回值都不携带层级语义——层级只由 §3.2 第 4 步的 materialization intent 表达。
本文其余部分对这两个函数只有一个要求：**它们的容量预算必须按 §4.6 的 `slotCount` 计费**，
否则 tile 选得刚好塞满一层，double buffer 一上来就超容量（下游文档 §10 的 512 KiB 反例就是这个）。

---

## 1. 结论速查

| 问题                                               | 结论                                                                                                           | 依据                                                               |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| tiling 和 pipeline 是否每层各跑一遍                | **是。** `Round(L2)` 然后 `Round(L1)`                                                                | 用户约束                                                           |
| pipeline 的**结构改写**是否也每层各做一遍    | **不是。** 每层只做 plan，结构展开在两层 plan 都完成后统一做一次，由内向外                               | §2.3（否则 L1 slot 翻倍）                                         |
| v0 的 slot 表示                                    | **K 个独立 `alloc_tensor` root**，不是一个 ring root + K 个 `extract_slice`                          | §5.1，**实测** T1c/T1g 失败、T1e/T9 通过                    |
| ring（单 root + slot 维度）还能用吗                | 能，但必须把整个 ring 当一个 loop-carried tensor value 并用`insert_slice` 写 slot；这会给跨层边引入第二种 op | §5.1，**实测** T1f                                          |
| 尾块 / 不整除怎么处理                              | 优先`transform.loop.peel` 让主循环静态整除；尾块 slotCount=1 不流水                                          | §3.3，**实测** T10 显示 stock tiling 会产生 `tensor<?x?>` |
| 循环 trip count 必须静态吗                         | **不必。** 预取可以用 `scf.if` 谓词化，且不产生额外 buffer                                             | §5.5，**实测** T8                                           |
| 能不能用`%iv mod K` 动态选 slot                  | **不能。** 硬失败                                                                                        | §5.1，**实测** T6                                           |
| 能不能靠交换 iter_arg 位置轮转                     | **不能。** 硬失败                                                                                        | §5.4，**实测** T3b                                          |
| copy 之后能不能继续 yield 写前的版本               | **不能。** 硬失败                                                                                        | §5.4，**实测** T3                                           |
| 循环内`tensor.empty` 的后果                      | 变成循环内、**且 memory space 为 0** 的 `memref.alloc`                                                 | §3.6，**实测** T4                                           |
| v0 用不用上游`scf::pipelineForLoop`              | **不用。** 它做 rotation 而不是 K-way unroll，slot 绑定不是静态残差                                      | §4.9                                                              |
| 「double buffer」对应 Triton 的`num_stages` 多少 | Triton`numBuffers = stageDiff`，所以两份 buffer 对应 `num_stages=3`；本文改用 `slotCount` 做主参数       | §4.1                                                              |

---

## 2. 每层一轮：Round(T) 的结构

### 2.1 两轮的内容

```text
Round(L2):
  1. TileForTier(L2)              §3
  2. MaterializeTensorStorage(L2) 下游文档 §4.2
  3. PlanPipeline(L2)             §4   —— 只标注，不改结构

Round(L1):
  4. TileForTier(L1)              §3
  5. MaterializeTensorStorage(L1) 下游文档 §4.2
  6. PlanPipeline(L1)             §4   —— 只标注，不改结构

之后（只做一次）:
  7. ResolveTensorConflicts       下游文档 §6
  8. PlanStorageAndSpills ↔ PlanPipeline 单调迭代   下游文档 §5.2
  9. MaterializePipeline          §5   —— 由内向外展开
 10. VerifyPreBufferizeForm       §7
```

第 1–3 步只看 DRAM ↔ L2 这条边，第 4–6 步只看 L2 ↔ L1 这条边。两轮共用同一套代码，
差别全部由一个 `tier` 参数和 target description 提供的 `capacity/alignment/bank` 决定。

### 2.2 为什么 tiling 必须两轮而不是一次两级 tiling

一次性 `tile_sizes [64, 64]` + `tile_sizes [32, 32]` 在语法上当然可以（**实测** T10），但两层的决策变量不同：

- 容量不同 → tile size oracle 的预算不同；
- 融合决策不同：一个 producer 值得融进 L2 loop，不一定值得融进 L1 loop（融进去会占 L1 的 scratch）；
- 需要 materialize 的值不同：L2 轮产生 DRAM→L2 的 stream，L1 轮产生 L2→L1 的 stream 加 accumulator/scratch；
- 尾块策略可以不同：L2 层 peel，L1 层 mask。

所以两轮是两次独立的 `tileConsumerAndFuseProducersUsingSCF` 调用，中间夹一次 materialization，
不是一次调用配两组 tile size。

### 2.3 为什么 pipeline 的结构展开不能跟着轮次做（关键论证）

假设严格按「L2 tiling → L2 pipeline（含结构展开）→ L1 tiling → L1 pipeline」执行。
L2 的结构展开会把 outer loop **2-way unroll**（§5.1 的静态 slot 要求逼出来的，见 §5.3），于是 outer body
里出现两份完整的计算区域 A、B。接下来 L1 tiling 分别对 A、B 做 tiling，得到**两个** inner loop；
L1 pipeline 再分别给这两个 inner loop 建 slot，于是出现**两套 L1 ring**。

这四个 L1 slot 无法靠 arena 复用掉：按下游文档 §5.5，slot allocation 必须提到所有重复执行 scope 之外，
即 outer loop 之前；两套 ring 的 SSA handle 都横跨整个 outer loop，lifetime 完全重叠，
静态 planner 只能给它们四个不同 offset。结果是 **L1 占用翻倍**，而 L1 恰恰是最紧的一层。

修正办法就是下游文档 §5.2 已经给出的 plan / materialize 分离：

```text
PlanPipeline(T)        只在 loop 和 op 上写 schedule 标注（stage / cluster / slotCount / role）
MaterializePipeline    在两层 plan 都冻结后，一次性由内向外展开所有被标注的 loop
```

由内向外展开时，L1 loop 先展开：inner loop 2-way unroll，L1 slot 提到 outer loop 之外，
通过 outer loop 的 `iter_args` 传进 inner loop。然后 outer loop 展开：2-way unroll 复制的是
**已经展开好的 inner loop**，两份副本引用**同一组** L1 slot root。L1 占用不翻倍，只有代码体积翻倍。

这一点不是推理，是实测：**T9**（§6）就是这个形态，bufferize 后 L1 weight slot 恰好 2 份，
两个 inner loop 共享它们。

顺带说明：Triton 的 pipeliner 直接 `isOuterLoop(forOp)` 就 bail
（`ScheduleLoops.cpp:41`，"Don't pipeline outer loops"），它从不处理这个问题。
我们必须处理，因为 DRAM→L2 的预取本质上就发生在 outer loop 上。

### 2.4 与下游文档 §6 pass pipeline 的对照

下游文档 §6 把流程写成一条线性链。本文的两轮结构是它的细化，不是替代：

| 下游文档 §6                                                                  | 本文                                                                                                                                          |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `L2 tiling + L2 materialization intent` → `MaterializeTensorStorage(L2)` | Round(L2) 第 1–2 步，一致                                                                                                                    |
| `L1 tiling + L1 materialization intent` → `MaterializeTensorStorage(L1)` | Round(L1) 第 4–5 步，一致                                                                                                                    |
| `PlanPipeline ↔ PlanStorageAndSpills（单调迭代）`                          | 拆成`PlanPipeline(L2)`（第 3 步）、`PlanPipeline(L1)`（第 6 步）和第 8 步的联合迭代。§6 的单个 `PlanPipeline` 读作「两层的 plan 之和」 |
| `MaterializePipeline`                                                       | 第 9 步，一次，由内向外。§6 未规定展开顺序，本文规定为内→外                                                                                 |
| `VerifyTensorPlacementAndCapacity`                                          | 第 10 步`VerifyPreBufferizeForm`（§7）是它的可机械检查版本                                                                                 |

唯一的实质性差异是 `MaterializePipeline` 的**位置**：§6 把它放在两轮 tiling 之后，本文同意；
本文额外规定它**只运行一次**、且**内层先展开**，并解释了原因（§2.3）。

---

## 3. TileForTier(T)：算法

### 3.1 输入

```text
TileForTier(tier T, target description D, TileSizeOracle O_t, FusionOracle O_f):
  输入 IR: canonical DPS tensor IR（T=L2）或 Round(L2) 的输出（T=L1）
  预算:    budget(T) = capacity(T)
                     - reservedBytes(T)            // ABI / runtime / bank 保留
                     - residentBytes(T)            // 已决定常驻本层的 loop-invariant 值
           且交给 O_t 时必须按 slotCount 折算，见 §3.1.1
```

#### 3.1.1 预算与 slotCount 的前馈约定

tile size 和 slotCount 互相依赖：tile 越大越省 traffic，但 double buffer 要两份。
为了不形成循环依赖，v0 用**固定前馈假设 + 单调回退**：

```text
交给 O_t 的预算按角色折算：
   stream 类（每迭代换内容的输入 tile / 输出 tile）  按 2 份计费
   resident 类（loop-invariant，整个 loop 只搬一次） 按 1 份计费
   accumulator / scratch                            按 1 份计费

即：sum_stream(2 * alignedBytes(tile)) + sum_other(alignedBytes(tile)) <= budget(T)
```

`PlanPipeline` 之后如果实际 slotCount 与假设不符（例如某条流水被判定不可流水，slotCount 降为 1），
容量只会变松，不需要回退。反向（需要更多 slot）在 v0 不可能发生，因为 v0 的 lookahead 固定为 1。
若容量仍不满足，按下游文档 §7.4 的顺序处理：先降 depth（2→1），再回退 tile size，最后报硬错误。

### 3.2 步骤

```text
1. 收集本层的 tiling scope 和候选 root：
     walk 所有实现 TilingInterface 的 op，按 O_f 给出的 cluster 分组，
     每个 cluster 有一个 root consumer。

2. 对每个 cluster：
     tileSizes = O_t(root, T, budget(T))
     若 tileSizes 全 0（不需要在本层再分块，例如 op 本来就装得下）则跳过。

3. 调用 tile-and-fuse：
     scf::SCFTileAndFuseOptions opts;
     opts.tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);   // 必须 ForOp，见 §3.7
     opts.tilingOptions.setTileSizeComputationFunction(wrap(O_t));
     opts.setFusionControlFn(wrap(O_f));
     scf::tileConsumerAndFuseProducersUsingSCF(rewriter, root, opts);
   产出：scf.for 循环嵌套 + body 内多个已切块的小算子 + 边界上的
         tensor.extract_slice / tensor.insert_slice。

4. 尾块规范化 NormalizeTileShapes（§3.3）。这一步必须在 5 之前。

5. 标注 materialization intent（下游文档 §4.2 的 spm.materialize 字典属性）：
     对每个需要进入本层的 tile value 标 {space = T, role = ..., pipeline_candidate = ...}
     role 取值：resident | stream | accumulator | scratch | writeback

6. 交给 MaterializeTensorStorage(T)（下游文档 §4.2）。它消费并删除标注，插入
     bufferization.alloc_tensor <{memory_space = T}>
   以及（仅当原内容会被读取时）
     bufferization.materialize_in_destination
   纯 output/accumulator destination 只插 allocation，不插初始 copy。
   仓库已有 transform.structured.promote_tensor 的实现正是这个语义
   （LinalgTransformOps.cpp:443-495：alloc_tensor + 按 mayBeRead 决定是否插 materialize），
   可以直接抽成公共 utility。

7. 运行 §3.5 的后置条件检查。
```

> **第 6 步的一条实现要求**：promote 必须按 **destination use** 建 allocation，而不是按 value。
> `promote_tensor` 用的是 `replaceAllUsesExcept`（`LinalgTransformOps.cpp:491`），一个 value 一份
> allocation：若同一个 tile value 同时是两个 DPS op 的 `outs`（例如 `linalg.fill` 与
> `linalg.generic` 共用一个 `outs`），就只会建出一份 allocation，在 tensor 层留下 **destination 分叉**——
> §5.7 的 HoistOnly 只能报错，One-Shot 也只能插 copy 把它救回来（多一份循环内 alloc）。
> 多个 op 只是**读**同一个 tile（`ins`）不受影响，共享一份 promote 是对的。
> 见 `SPM_HOIST_ONLY_EXAMPLE.md` §7.2.2。

### 3.3 尾块规范化（必须做，有实测依据）

**实测 T10**：对 `linalg.matmul` 做两级 `transform.structured.tile_using_for`（`[64,64]` 然后 `[32,32]`），
在 shape 不整除（130×64 × 64×96）时，产出的 tile 类型是**动态的**：

```mlir
%2 = affine.min affine_map<(d0) -> (-d0 + 130, 64)>(%arg3)
%3 = affine.min affine_map<(d0) -> (-d0 + 96, 64)>(%arg5)
%extracted_slice   = tensor.extract_slice %arg0[%arg3, 0]  [%2, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
%extracted_slice_5 = tensor.extract_slice %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<130x96xf32> to tensor<?x?xf32>
// 而且内层 loop 的上界就是这两个动态值：
%10 = scf.for %arg7 = %c0_9 to %2 step %c32 iter_args(...) -> (tensor<?x?xf32>) { ... }
```

两个后果都致命：

1. `bufferization.alloc_tensor` 直接作用在 `tensor<?x?xf32>` 上就是**动态大小的 SRAM 分配**，
   违反下游文档 §11 不变量 10。
2. 内层 loop 的 trip count 变成动态，2-way unroll 的残差绑定无从谈起。

因此 `TileForTier` 之后必须紧跟 `NormalizeTileShapes`，策略按优先级：

```text
S1. Peel（默认，推荐）
    transform.loop.peel（SCFTransformOps.td:171，"updates the given loop so that its step
    evenly divides its range and puts the remaining iteration into a separate loop"）
    结果：主循环 tile 全静态、trip count 静态；尾循环单独一份。
    尾循环标 pipeline_candidate = false，slotCount = 1，不流水（§5.7）。
    代价：代码体积；收益：主循环拿到干净的静态残差绑定，且内层 loop 的 trip count 也变回静态。

S2. 静态最大 slot + 动态有效子范围（下游文档 §4.4 的做法）
    用 value-bounds 求 affine.min 的上界（= tile size），按上界建 allocation，
    循环内用 tensor.extract_slice 取实际有效范围参与计算，越界部分靠 padding/mask/predication。
    可直接参考 transform.tensor.make_loop_independent（TensorTransformOps.td:131）。
    适用于：不希望代码翻倍、或尾块占比极小的维度。

S3. Pad 到 tile size
    适用于：element type 允许无害 padding、且 pad 的 traffic 可接受。
```

硬规则，写进 verifier：

- 任何 `alloc_tensor` 的 result type 必须**静态**；
- 被标 `pipeline_candidate = true` 的 loop，其 lb/ub/step 必须静态，或者已经过 S1 peel；
- 尾块**绝不**允许生成一个动态大小的 L1/L2 allocation。

### 3.4 reduction 与 accumulator

reduction 维度切块会引入一个跨迭代的 accumulator。它是「循环内 alloc」的一个独立来源，处理方式与 stream 不同：

```text
若 root 有 reduction 维且需要在本层切它：
  - v0 用 SCFTilingOptions::reductionStrategy = FullReduction，
    accumulator 表达为 scf.for 的 iter_arg，alloc 在循环之前；
  - accumulator 的 alloc 天然在循环外，slotCount = 1，不参与 rotation；
  - init（linalg.fill）也必须在循环外，不能每迭代 fill 一个新 tensor.empty；
  - accumulator 是 loop-carried recurrence：按下游文档 §7.8，它不可 spill，
    且在 PlanPipeline 里必须整体落在同一个 stage（§4.5 规则 L5）。
```

`PartialReductionOuterReduction` 等策略会产生额外的 partial-result tensor，
那些 tensor 同样要走 §3.5 的分类，v0 不启用。

### 3.5 后置条件与 verifier（TileForTier(T) 出口 gate）

```text
P1. body 内每个 hardware compute op 的 operand/outs 都能追溯到某个 space=T'（T' ⊑ T）的 root，
    或者是本轮待 materialize 的 tile value（还带着 spm.materialize 标注）。
P2. 所有待 promote 的 tile value 的类型静态（§3.3）。
P3. body 内单个迭代的本层占用（按 §3.1.1 折算）<= budget(T)。
    这是「颗粒度一定放得下」的机械化定义，也是 tiling 算法的正确性判据。
P4. 所有 spm.materialize 标注在离开本轮之前都被消费掉（未消费标注 = 硬错误）。
P5. 没有残留的 tensor.empty（§3.6）。
P6. 循环类型是 scf.for（§3.7）。
```

### 3.6 循环内 allocation 的来源全表

这张表是完备性的核心：**tiling 之后每一种会导致「循环内 allocation」的来源，都必须在这里有归属**，
否则它会一路活到 One-Shot 变成循环内 `memref.alloc`。

| #   | 来源                                                        | 归属           | 处理                                                |
| --- | ----------------------------------------------------------- | -------------- | --------------------------------------------------- |
| A1  | 每迭代换内容的输入 tile（stream）                           | §5.3 rotation | slotCount = 2，2-way unroll                         |
| A2  | 每迭代产生的输出 tile / writeback staging                   | §5.3 或 §5.7 | 有跨迭代 in-flight 需求则 2，否则 1                 |
| A3  | reduction accumulator                                       | §3.4          | 循环外 alloc + iter_arg，slotCount = 1              |
| A4  | 融合进来的 producer 的中间结果（scratch/temp）              | §5.7          | hoist-only，slotCount = 1                           |
| A5  | `linalg.fill` / DPS op 的 `tensor.empty` destination    | §3.6.1        | 先 eliminate，剩下的转`alloc_tensor`              |
| A6  | transpose / pack / layout 变换的临时 buffer                 | §5.7          | 同 A4                                               |
| A7  | 尾块 padding buffer                                         | §3.3 S3       | 按上界静态分配，slotCount = 1                       |
| A8  | `insertTensorCopies` 为消解 RaW 插入的 copy 的 allocation | 下游文档 §6   | 规范化成 alloc + materialize 后重新参加 §4 的 plan |
| A9  | spill/fill 新增的 home/transit allocation                   | 下游文档 §7.5 | 重新参加 §4、§5（下游文档 §11 不变量 17）        |
| A10 | pipeline 自己在 prologue 里引入的 copy                      | §5.5          | 不新增 allocation，只写已有 slot                    |

#### 3.6.1 `tensor.empty` 的处理次序（有实测，次序会影响正确性）

**实测 T4a**：循环内一个 `tensor.empty` 作为 `linalg` 的 `outs`，直接 bufferize 的结果是

```mlir
scf.for ... {
  %alloc = memref.alloc() alignment = 64 : memref<32x64xf16>   // 循环内，而且 memory space = 0
  linalg.fill ins(%cst) outs(%alloc)
  linalg.matmul ins(%subview, %arg1) outs(%alloc)
  ...
}
```

两个问题同时发生：循环内分配 + **落到默认 space 0**（既不是 L1 也不是 L2），
后者会直接把下游文档 §9.1 的 per-space 分桶弄坏。

**实测 T4b/T4c**：`-eliminate-empty-tensors` 会把这个 destination 换成
`tensor.extract_slice %arg4[...]`（即最终会被 insert 回去的那块 subset），
之后 bufferize 的 `memref.alloc` 数量为 **0**。

但这里有个陷阱：被换上来的 destination 是**外层目标 tensor 的 subset**。在我们的分层语境里，
外层目标可能在 L2 甚至 DRAM——于是 compute 直接往 L2/DRAM 写，违反「compute 只碰 L1」。

因此次序必须是：

```text
1. 先跑 empty-tensor elimination（-eliminate-empty-tensors），把能消掉的 destination 消掉；
2. 再跑 MaterializeTensorStorage(T)：把 compute 的 outs 显式 promote 到 L1（alloc_tensor + 不带初始 copy），
   并把「写回外层 destination」表达成一条独立的 materialize_in_destination 边；
3. 仍然剩下的 tensor.empty 用 -empty-tensor-to-alloc-tensor 转成 alloc_tensor，
   补上 memory_space，然后按 §5.7 hoist。
```

反过来（先 materialize 再 eliminate）会让 elimination 把刚建好的 L1 staging 又消掉。
这一条写进 pass pipeline 的顺序约束，并在 §7 的 verifier 里检查「compute 的 outs 是 L1 root」。

### 3.7 循环类型必须是 `scf.for`

`SCFTilingOptions::LoopType` 支持 `ForOp` / `ForallOp` / `CustomOp`
（`TileUsingInterface.h:48`）。本文一律用 `ForOp`：

- `scf.forall` 表达的是**无序并行**，语义上不允许「上一迭代预取、这一迭代消费」的跨迭代依赖，
  software pipeline 无从下手；
- 上游 pipeliner 和 Triton 的 pipeliner 都只处理 `scf.for`。

如果前端因为其他原因先产生了 `scf.forall`，必须在 `PlanPipeline` 之前用
`mlir/lib/Dialect/SCF/Transforms/ForallToFor.cpp` 转成 `scf.for`。
v0 不支持在 `scf.forall` 上做流水。

---

## 4. PlanPipeline(T)：分析与调度

`PlanPipeline` **不改结构**，只在 IR 上写标注。结构改写全部在 §5。
算法骨架直接对应 Triton 的分解，逐段给出对应关系。

### 4.1 术语先钉死：`slotCount` 而不是 `numStages`

Triton 里物理 buffer 数是

```text
numBuffers = asyncLoad.stageDiff                       (LowerLoops.cpp:356, :376)
stageDiff  = stage(firstUse) - stage(def)              (LowerLoops.cpp:443)
buffer shape = [numBuffers] ++ tileShape               (PipeliningUtility.cpp:417-419)
```

而 `stage` 来自 latency 的最长路径（§4.4），对一条 `load -> dot` 链，
`loadLatency = (numStages - 1) / (maxIndirectionLevel + 1)`（`AssignLatencies.cpp:127`），
于是 `num_stages=2` 给出 `stageDiff = 1`，即**只分配一份 shared buffer**；
真正的两份 buffer 对应 `num_stages=3`。

原因（读码）：`scheduleKeyOps` 把 cluster 按 stage 逆序分配
（"ops with higher stage numbers are assigned first... roughly reverse program order"，
`ScheduleLoops.cpp:219-223`），所以 Triton 的 steady-state body 里 **compute 在前、issue copy 在后**，
一份 buffer 就够，但 DMA 只能和 back-edge 重叠。我们要的是 **issue copy 在前、compute 在后**，
让 DMA 和 compute 真重叠，代价就是多一份 slot。

为了避免「num_stages=2 到底是几份 buffer」这种歧义，本文**不用 `numStages` 作主参数**，改用：

```text
slotCount(buf) = buf 在 steady-state body 中同时被 reserve 的不同迭代实例数
               = 1 (正在被消费的那份) + lookahead (提前发起的迭代数)
v0: lookahead = 1  =>  stream 类 slotCount = 2   （= 用户说的 double buffer）
    非流水的 loop-local buffer slotCount = 1
```

用户要求的「只支持 double buffer，不考虑 num_stages > 2」在本文的记法里就是
**`lookahead` 固定为 1、`slotCount ∈ {1, 2}`**。

### 4.2 Candidate discovery

```text
candidates(loop, T) =
  { c : c 是 bufferization.materialize_in_destination
      且 c 在 loop body 的顶层（不在嵌套 region 内）
      且 destination 的 allocation root 的 memory_space == T
      且 destination root 定义在 loop body 内（loop-local staging）}
```

按方向再分类（方向由 source/destination root 的 space 推导，下游文档 §2.3 的表）：

| 类别           | source space | dest space | 典型                                                    |
| -------------- | ------------ | ---------- | ------------------------------------------------------- |
| `in-stream`  | 慢层         | T          | DRAM→L2 tile load（T=L2）、L2→L1 subtile fill（T=L1） |
| `out-stream` | T            | 慢层       | L1→L2 writeback、L2→DRAM store                        |
| `local`      | T            | T          | 层内 copy，通常不值得流水                               |

`in-stream` 是 lookahead 的对象（提前发起）；`out-stream` 是 lagging 的对象（延后完成）。
v0 只对 `in-stream` 做 rotation；`out-stream` 在 v0 按同步处理，slotCount 由 §4.6 计算
（通常是 1，除非同一个 slot 在下一迭代就被覆写）。

### 4.3 Latency assignment

```text
AssignLatencies(loop, T):
  for c in candidates(loop, T):
      if not canPrefetch(c):        latency(c) = 0   // 不流水
      elif bytes(c) < D.dmaMinOverlapBytes:  latency(c) = 0   // 太小，DMA 开销 > 收益
      else:                         latency(c) = 1   // v0 固定 1
```

`canPrefetch(c)` 是**唯一实质性的分析**，v0 的判据：

```text
C1. c 的 source 的所有索引都能在「提前一个迭代」的位置重新计算：
    索引表达式对 induction variable 是 affine 的，且其余 operand 循环不变或定义在 loop 外。
    实现上：把 source 的 backward slice 取出来，检查 slice 内每个 op 要么循环不变，
    要么只依赖 iv；然后能用 iv+step 重新实例化这个 slice。
C2. c 的 source 不依赖同一迭代内计算出来的值（否则没法提前）。
    典型反例：source 的地址来自上一个算子的输出（间接寻址 / gather）。
    Triton 用 maxIndirectionLevel 把这种情况的 latency 摊薄（AssignLatencies.cpp:127），
    v0 直接判 latency = 0。
C3. c 不在 loop-carried recurrence 的环上（下游文档 §7.8 最后一条）。
C4. c 的 destination root 没有被同一 body 内其他 candidate 共享。
```

`canPrefetch` 为假不是错误，只是这条边不流水：它的 destination 走 §5.7 的 hoist-only 路径，slotCount = 1。

### 4.4 Stage assignment（直接移植 Triton 的最长路径公式）

Triton `scheduleKeyOps`（`ScheduleLoops.cpp:152-247`）的核心是：

```text
distance(op) = latency(op) + max over in-loop users u of distance(u)      // 到 yield 的最长延迟路径
               （无 in-loop user 时 max 项取 0）
maxDistance  = max over latency ops of distance(op)
stage(op)    = maxDistance - distance(op)
```

对我们 v0 的典型形状（一条 `in-stream` copy，latency 1，喂给 compute）：

```text
distance(compute) = 0 + 0 = 0
distance(copy)    = 1 + 0 = 1
maxDistance = 1  =>  stage(copy) = 0,  stage(compute) = 1
```

之后照 Triton 的顺序补齐（`ScheduleLoops.cpp:351-386` 的 `scheduleLoop`）：

```text
1. schedule = scheduleKeyOps(...)                     stage 来自最长路径
2. schedulePrologueAndEpilogue(...)                   把 body 里已有的 scf.if 推到两端 cluster
3. scheduleDependencies(...)                          backward 传播：def 的 stage <= use 的 stage
4. scheduleDistanceOneDependencies(...)               通过 iter_arg 的 distance-1 依赖 -> stage+1，
                                                      放进当前 cluster 之前的新 cluster
5. scheduleRemainingToLastStage(...)                  其余 op 归到最后一个 stage，
                                                      并保证 use 不早于 def 的 cluster
6. 把 (stage, cluster) 序列化到 IR 属性上              §4.7
```

cluster 的用途是 stage 内部的排序。**我们对 cluster 的约定与 Triton 相反**（§4.1）：
`in-stream` copy 所在的 cluster 必须排在同 stage 的 compute **之前**，这样展开后 DMA 才和 compute 重叠。
这一条是 v0 的 `slotCount = stageDiff + 1` 的来源。

嵌套 loop 的处理：内层 `scf.for` 作为**一个整体**参与外层的 stage 分配。
上游 pipeliner 也是这个约定——它只要求「被分配 stage 的 op 的 owning block 是 loop body block」
（`LoopPipelining.cpp:176-184`），嵌套 region 内部的 op 不分配 stage，容器 op 拿一个 stage。

### 4.5 合法性检查与 bail-out

综合 Triton `isSafeToPipeline`（`ScheduleLoops.cpp:36-48`）和上游
`initializeLoopInfo`（`LoopPipelining.cpp:99-205`），v0 的 bail-out 条件：

```text
L1. loop-carried 依赖距离 > 1                       -> 不流水
L2. body 内有屏障 / assert / print / 未知副作用 op   -> 不流水
L3. body 顶层有 op 拿不到 stage                     -> 硬错误（说明 §4.4 有 bug）
L4. 静态 trip count < slotCount                     -> 不流水（退回 §5.7）
     trip count 动态时不 bail：走 §5.5 的谓词化路径
L5. loop-carried recurrence（accumulator）跨了 stage -> 硬错误；recurrence 必须整体在最后一个 stage
L6. candidate 的 destination root 不在 body 内       -> 已经是 hoisted slot，不需要 rotation
L7. body 里有 early exit / scf.while                 -> v0 不流水
```

L4 的「不流水」意味着：这个 loop 的所有 loop-local allocation 走 §5.7 的 hoist-only 路径，
仍然满足 pre-bufferize 契约，只是没有 DMA/compute 重叠。**正确性从不依赖流水成功**。

### 4.6 slotCount 计算

```text
for each loop-local allocation root a of space T:
    copies  = { c in candidates : dest root of c == a }
    uses    = { u : u 读 a 或读 a 的 subset }
    if 所有 copies 的 latency 都是 0:
        slotCount(a) = 1                                     // hoist-only，§5.7
    else:
        d = max over (c, u) of ( stage(u) - stage(c) )        // = Triton 的 stageDiff
        slotCount(a) = d + 1                                  // +1 见 §4.1
        // v0: d 恒为 1  =>  slotCount = 2
assert slotCount(a) <= 2                                      // v0 上界
```

把 `slotCount(a)` 和 `alignedBytes(a)` 写到 `a` 上，供下游文档 §7 的容量 planner 按
`slotCount × alignedTileBytes` 计费（下游文档 §11 不变量 11）。

### 4.7 schedule 的落地形式

`PlanPipeline` 的输出全部是 IR 属性，不改结构。这一点和 Triton 完全一致——Triton 在
`ScheduleLoops` 结束时 `schedule.serialize(forOp)`，在 `expandLoops` 开头
`schedule.deSerialize(forOp)`（`SoftwarePipeliner.cpp:97`），中间还夹着 `lowerLoops`。

建议的属性（放在 loop op 和 body 内 op 上）：

```text
loop 上:   spm.pipeline = { lookahead = 1, slotCount = 2, tailPolicy = peel|predicate, tier = l1|l2 }
op 上:     spm.stage = i64, spm.cluster = i64
alloc 上:  spm.slot_count = i64, spm.role = stream|resident|accumulator|scratch|writeback
```

`MaterializePipeline` 必须消费并删除这些属性；残留标注是硬错误（和下游文档 §4.2 对
`spm.materialize` 的要求一致）。

### 4.8 与容量 planner 的单调迭代

按下游文档 §5.2，`PlanPipeline` 和 `PlanStorageAndSpills` 交替运行直到收敛，
一次编译内只允许三种单调决策：

```text
- residency 从快层往慢层移动；
- lookahead / slotCount 只减不增（2 -> 1）；
- tile size 由外层重试机制只减不增。
```

本文补一条实现细节：容量 planner 因 spill 插入的新 copy（下游文档 §7.5）会成为新的 candidate，
必须重新跑 §4.2–§4.6。为保证终止，给每个 loop 记一个 `pipelineRevision`，
每轮只允许 slotCount 下降；若某轮既要降 slotCount 又要加 copy 且仍不可行，
直接退回 tiling（下游文档 §7.9），不再迭代。

### 4.9 为什么 v0 不用上游 `scf::pipelineForLoop`

上游 `mlir/lib/Dialect/SCF/Transforms/LoopPipelining.cpp` 提供了完整的 expander，
`PipeliningOption`（`Transforms.h`）有 `getScheduleFn` / `predicateFn` / `peelEpilogue` /
`supportDynamicLoops` / `annotateFn`，Triton 的 `PipelineExpander.cpp` 就是它的 fork。
但它做的是 **rotation**：

```text
createKernelLoop / createKernel 里，一个跨 stage 的值会按
  version = maxStage - lastUseStage + 1                     (LoopPipelining.cpp:608)
产生多份 valueMapping，通过额外的 iter_args 传递；
kernel 的上界改成 newUb = ub - maxStage * step              (LoopPipelining.cpp:445-450)
```

也就是说，**它把「第 i 次迭代用哪一份」编码成随迭代轮转的 SSA 版本映射**，
而 v0 需要的是「残差 j 静态绑定 slot j」。把 rotation 的结果再变成静态残差绑定，
还是要在外面套一层 2-way unroll；直接 unroll-then-rewrite（§5.3）更短也更好验证。

所以 v0 自己写 `MaterializePipeline`。上游 expander 保留为后续选项：
当我们允许动态 ring 索引（下游文档 §14 的第一条后续项）时，它就直接可用了。

---

## 5. MaterializePipeline：结构改写

### 5.1 v0 的 slot 表示：K 个独立 allocation root

下游文档 §5.3 **推荐**的形态是「一个 ring root + K 个静态 disjoint `extract_slice`，slice 作为 iter_args」。
实测表明这个形态在真实流水调度下**会被 One-Shot 判为不可原地化**。

**实测 T1c**（ring + 两个 slice 作 iter_args + prologue 填 ping + 真实流水顺序）：

```mlir
%alloc   = memref.alloc() : memref<2x32x64xf16, 1>            // ring
%subview = memref.subview %alloc[0,0,0] ... // slot0 的 subview 被建出来后就废弃了
%alloc_0 = memref.alloc() : memref<32x64xf16, 1>              // <== 多出来的一份 buffer
memref.copy %subview_3, %alloc_0                              // prologue 写到了这份新 buffer
```

原因：`%pong = extract_slice %ring[1]` 是对 `%ring` 值的一次读，而 prologue 往 `%ping` 的 buffer 里写
等于覆写 `%ring` 的内容；`%pong` 之后还要用，One-Shot 只能把其中一个 slot 拷出去。
**实测 T1g** 把 prologue 改成先 `insert_slice` 到 ring 值再取两个 slice，问题仍然存在（多一份 alloc + 一次 copy）。

三个可行形态的实测对比：

| 形态                                                                                                                       | 实测结果                                                                                                                        | 评价              |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| ring root + K 个 slice 作独立 iter_args（下游文档 §5.3 字面形态）                                                         | **失败**：多一份 alloc + 一次 copy（T1c、T1g）                                                                            | 不可用            |
| 整个 ring 作**单个** loop-carried tensor value，slot 写用 `tensor.insert_slice`，读用 `extract_slice` 取最新版本 | **通过**：2 个 alloc（ring + acc），所有 slot 访问是静态 offset 的 `memref.subview`，循环内 0 alloc、0 额外 copy（T1f） | 可用，但见下      |
| **K 个独立 `alloc_tensor` root**，各自作 iter_arg                                                                  | **通过**：K 个 alloc，循环内 0 alloc、0 额外 copy（T1e、T9）                                                              | **v0 采用** |

选 K 个独立 root 的理由：

1. 跨层内容迁移保持只有一种 op。ring-as-single-value 形态必须用 `tensor.insert_slice` 把慢层数据写进
   ring 值（因为 `materialize_in_destination` 写进一个 `extract_slice` 的结果**不会**产生更新后的父 tensor 版本），
   这会给「跨层边」引入第二种 op，和下游文档 §11 不变量 5（所有跨 placement edge 都经过
   `materialize_in_destination`）冲突。
2. 独立 root 之间不存在父值失效问题，prologue/epilogue 里在循环外写 slot 是平凡合法的（T1e/T9 实测）。
3. 「物理不相交」（下游文档 §5.6 条件 4）对不同 root 平凡成立，不需要任何 subset disjointness 证明。
4. 下游文档 §11 的 20 条不变量没有任何一条要求 K 个 slot 共享一个 root；§7.1 只是说
   「view 默认按整个底层 root 计费，只有静态证明 disjoint 的 subset 才能分别建 reservation」——
   独立 root 直接就是分别建 reservation。

「K 个 slot 连续放置」这个诉求并没有丢，它只是搬到了正确的位置：**它是 arena planner 的 layout hint**，
属于下游文档 §9 的范围。`PlanPipeline` 给同一组 slot 打一个 `spm.slot_group = <id>` 属性，
arena planner 可以选择把同组 slot 连续、同 alignment class 放置。这比在 tensor 层伪装成一个 root 更干净：
tensor 层不需要任何 disjointness 推理，物理连续性由决定物理地址的那一层负责。

v0 的 slot 表示因此是：

```mlir
// slotCount = 2 的 stream（这里是 L1 weight subtile）
%w1a = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l1>}> : tensor<64x32xf16>
%w1b = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l1>}> : tensor<64x32xf16>
```

> 实现提示（实测）：本 checkout 里 `memory_space` 是 inherent property，必须写成
> `bufferization.alloc_tensor() <{memory_space = ...}>`；写成普通 attr-dict
> `alloc_tensor() {memory_space = ...}` 是**解析错误**
> （"inherent attribute 'memory_space' cannot be parsed from attr-dict when strict properties
> in assembly format is enabled"）。下游文档的写法是对的。

### 5.2 slot allocation 放在哪里

严格按下游文档 §5.5：**最接近但位于所有重复执行 scope 之外**。具体规则：

```text
hoistPoint(a) = 紧邻 outermostRepetitiveAncestor(a) 之前
  其中 outermostRepetitiveAncestor(a) = 从 a 的定义点往外走，
  最外层那个会重复执行 a 的 region 的 owner op。

推论：
 - 内层 loop 的 slot，若跨外层迭代复用（我们的 L1 slot 就是），提到外层 loop 之前；
 - 两个顺序执行的 top-level loop 各自在自己前面 alloc，arena 才能复用同一 offset；
 - 若 v0 的 static planner 只接受 function entry block 里的 alloc，
   则统一放进 entry block，但仍应尽量靠近 owning loop（下游文档 §5.5 第三条）。
```

**`hoistPoint` 的实现必须走白名单，不能直接拿 `getEnclosingRepetitiveRegion` 反复外推。**
该 API 把 `scf.forall` 也算成 repetitive region（`SCF/Transforms/BufferizableOpInterfaceImpl.cpp:106-113`：
`ForallOpInterface::isRepetitiveRegion` 只看 step 数）。盲目外推会把 alloc 提到 `forall` 之外，
**所有并发迭代共享同一份 L1 buffer = data race**，而且 bufferize 不报错、Q1 与「循环内 0 alloc」
的判据也照样通过。§3.7 只禁止了本层 tiling 产出 `forall`，不能假设 hoist 路径上没有别人产的 `forall`。
因此外推只跨 `scf.for` / `scf.if`；碰到 `scf.forall` / `scf.while` / 任何其他 repetitive region owner
就停在它里面，此时若 alloc 仍在 repetitive region 内（Q1 不满足）则报错。
算法与反例见 `SPM_HOIST_ONLY_EXAMPLE.md` §7.2.1。

被提升之后，slot 必须通过**每一层** loop 的 `iter_args` 往里传。T9 实测了两层的情况：
L1 slot 在函数顶层分配，穿过 outer loop 的 `iter_args`，再穿过两个 inner loop 的 `iter_args`。

### 5.3 2-way unroll + rotate：精确算法

```text
ExpandLoop(loop L, plan P):                    // P 来自 §4，lookahead = 1, K = slotCount = 2
  前置：L 已通过 §4.5；L 的 lb/ub/step 静态或 tailPolicy = predicate
  记 S = P.candidates_instream(L)              // 需要 rotation 的 in-stream copy
  记 A = { dest root of c : c in S }           // 需要 K 个 slot 的 allocation root

  // ---- 1. 建 slot 并提升 ----
  for a in A:
      在 hoistPoint(a) 处建 K 个独立 alloc_tensor（§5.1、§5.2），命名 a[0], a[1]
      删除 a 原来的循环内 alloc_tensor
  for a in A(slotCount == 1) 以及所有非 candidate 的 loop-local alloc:
      走 §5.7 的 hoist-only

  // ---- 2. 建 prologue（lookahead = 1，只需要预热 slot 0）----
  在 L 之前插入 S 中每条 copy 的「第 0 次迭代版本」：
      把 c 的 source 的 backward slice 用 iv := lb 重新实例化（§4.3 的 C1 保证可行）
      发射 c' = materialize_in_destination <source@lb> in a[0]
      记 initA[a] = c' 的 result           // 后面作为 iter_arg 的初值
  若 tailPolicy = predicate 且 trip count 可能为 0：
      整个 prologue 用 scf.if (lb < ub) 包起来，else 分支 yield 未写的 a[0]（§5.5 证明这样不产生 copy）

  // ---- 3. 建 steady-state loop ----
  newStep = 2 * step
  newUb   = tailPolicy == peel ? ub                    // peel 已保证 (ub-lb) % newStep == 0
                               : ub                    // predicate 路径不改上界
  iter_args = [ for a in A: a[0] slot 版本, a[1] slot 版本 ]   // 位置固定，见 §5.4
            ++ [ 其他原有 iter_args（accumulator 等） ]
  body 由两个 half 顺序拼成，half j ∈ {0, 1}：
      ivj      = iv + j * step                        // 本 half 消费的迭代号
      slotCur  = a[j]                                 // 静态残差绑定：half j 消费 slot j
      slotNext = a[(j + 1) % 2]
      ivNext   = iv + (j + 1) * step                  // 本 half 预取的迭代号

      // (a) 先发预取，让它和 (b) 的 compute 重叠  —— cluster 约定，§4.1
      for c in S:
          若 ivNext 一定在范围内（peel 路径的 j == 0，或静态可证）:
              slotNext' = materialize_in_destination <source@ivNext> in slotNext
          否则:
              slotNext' = scf.if (ivNext < ub) {
                            yield materialize_in_destination <source@ivNext> in slotNext
                          } else {
                            yield slotNext                    // 未写，原样传出
                          }
      // (b) 本 half 的 compute：把原 body 里 stage == maxStage 的 op 复制过来，
      //     iv -> ivj，对 a 的读 -> slotCur，其余 SSA 按 §5.4 接线
      emit body_ops(stage == maxStage) with iv := ivj

  yield：每个 slot 位置 yield 它自己那一路的最新版本（§5.4），accumulator yield 最新版本

  // ---- 4. epilogue ----
  tailPolicy == peel:      peel 出来的尾循环（§3.3 S1）本身就是 epilogue，slotCount = 1，不流水
  tailPolicy == predicate: 不需要独立 epilogue——最后一个 half 的预取被谓词关掉即可
```

关于「为什么必须 unroll 2 份」：v0 要求残差 → slot 的映射静态。三条替代路线都实测不通或被下游文档禁止：

- `%iv mod K` 动态索引 ring：下游文档 §5.3 明确禁止，且需要跨迭代 disjoint 证明；
- `arith.select` / `scf.if` 在两个 slot 之间动态选：**实测 T6 硬失败**
  （`'tensor.extract_slice' op not bufferizable under the given constraints: cannot avoid RaW conflict`）；
- 交换 iter_arg 位置轮转：**实测 T3b 硬失败**
  （`Yield operand #0 is not equivalent to the corresponding iter bbArg`）。

代价是 body 体积 ×2（嵌套两层则最内层 ×2，因为内层先展开、外层复制的是整个内层 loop，见 §5.6）。
这是静态 slot 绑定的必要代价，不是可以优化掉的实现瑕疵。

### 5.4 tensor SSA version threading 的三条硬规则

allocation 被提升之后，每次 copy/compute 仍然产生**新的** tensor SSA 内容版本。三条规则，
每条都有实测的反例证据：

```text
R1. 消费者必须消费 copy 的 result，不能继续消费写入前的 destination。
    反例实测 T3（yield 写前的 %p 而不是 %p1）:
      error: 'bufferization.materialize_in_destination' op not bufferizable
             under the given constraints: cannot avoid RaW conflict

R2. slot 在 iter_args 里的位置必须稳定：slot j 的版本永远 yield 回位置 j。
    反例实测 T3b（交换两个 slot 的 yield 位置）:
      error: Yield operand #0 is not equivalent to the corresponding iter bbArg

R3. 谓词化的 copy 两个分支必须 yield 同一个 slot 的版本（then: 写后版本，else: 原版本）。
    正例实测 T8：bufferize 后两个分支都 yield %arg4，无 alloc、无 copy。
```

R1/R2 就是下游文档 §5.4 的两条禁令，这里给出它们的实际诊断信息，便于实现时写 lit 测试的 `CHECK`。

### 5.5 尾部：peel 还是 predicate

两条路都可用，实测都干净。

**predicate 路线（实测 T8）**：`scf.if` 里做 `materialize_in_destination`，else 分支原样 yield slot。
bufferize 结果：

```mlir
%2 = scf.if %1 -> (memref<32x64xf16, 1>) {
  memref.copy %subview, %arg4 : ... to memref<32x64xf16, 1>
  scf.yield %arg4 : memref<32x64xf16, 1>
} else {
  scf.yield %arg4 : memref<32x64xf16, 1>
}
```

零 allocation、零额外 copy。**这条实测结论有一个重要推论：v0 不需要静态 trip count。**
上游 pipeliner 在 trip count 动态且没有 `predicateFn` 时会直接 bail
（`LoopPipelining.cpp:148-150`），而我们在 tensor 层的「谓词」就是一个 `scf.if`，
天然可用，不需要额外的 `predicateFn` 基础设施。

必须区分清楚两件事——这是 T6 和 T8 的差别，也是整个 v0 slot 模型的关键：

|                                                                 | 合法性                          |
| --------------------------------------------------------------- | ------------------------------- |
| 动态决定**往哪个 slot 写**（slot 选择依赖 iv）            | **非法**，T6 硬失败       |
| 动态决定**要不要往一个静态确定的 slot 写**（谓词化 copy） | **合法且无开销**，T8 通过 |

选择建议：

```text
tailPolicy = peel       当 trip count 静态、且代码体积可接受时首选。
                        主循环没有任何 scf.if，DMA 发射无分支；
                        而且 peel 让内层 loop 的 trip count 也变回静态（§3.3 的实测动机）。
tailPolicy = predicate  当 trip count 动态、或 peel 造成的代码膨胀不可接受时使用。
```

trip count 静态且 < slotCount（即 0 或 1 次）时按 §4.5 的 L4 完全不流水。

### 5.6 嵌套：由内向外展开

```text
MaterializePipeline(func):
  loops = 所有带 spm.pipeline 标注的 scf.for，按嵌套深度**降序**排序
  for L in loops:                     // 内层先
      ExpandLoop(L, plan(L))
```

内层先展开的两个理由：

1. 外层 2-way unroll 复制的是**已经展开好的**内层 loop，两份副本引用**同一组**内层 slot root
   → 内层 SRAM 不翻倍（§2.3）。若外层先展开，内层 tiling/plan 会各自看到两份 body，产生两套 slot。
2. 内层 slot 的 `hoistPoint` 在外层 loop 之外（§5.2），这要求外层 loop 此时还是一个完整的、
   可以往前插入的 op；先展开外层会把这个插入点打散成 prologue + 新 loop 两段。

外层展开时，内层 loop 整体作为一个 stage 单元参与（§4.4 末段）。内层 slot 的版本要在外层的
`iter_args` 里占固定位置，并且外层的两个 half 各自把内层 loop 的结果串行接下去——
T9 的 `%IA#0..3` 传给第二个 inner loop 的 `iter_args` 就是这个接线。

### 5.7 slotCount = 1 的情形：hoist-only

不是每个循环内 allocation 都需要 double buffer。§3.6 表里的 A3/A4/A6/A7 都属于
「内容生命周期完全落在一次迭代内、不跨迭代 in-flight」，它们只需要**一份** slot：

```text
HoistOnly(a, L):
  前置 1：a 是 L 的 loop-local allocation root，slotCount(a) == 1
  前置 2：a 已经存在。allocation root 由 MaterializeTensorStorage 在 §3.2 第 6 步创建；
          本路径不创建、不删除、不合并 alloc_tensor，进出数量 1:1
  1. 把 a 的 alloc_tensor 移到 hoistPoint(a)（§5.2，注意那里的白名单：不许跨 scf.forall）
  2. 给 L（以及所有中间层 loop）加一个 iter_arg 承载 a 的版本
  3. body 内对 a 的第一次写改成写这个 iter_arg，版本链的**终端**值 yield 回同一位置
  4. 若 a 在每次迭代开始时都被完整覆写（例如先 linalg.fill 再 matmul），不需要任何跨迭代内容假设；
     若不是完整覆写，它大概率是 accumulator（A3，按 §3.4 处理）——但这里只 warning，不报错：
     alloc_tensor 的内容是 undefined，把上一迭代的残留穿进来是对 UB 的合法 refine，
     所以这一条不影响正确性，只用来探测上游的分类错误
  5. 检查版本链线性：a 与每个中间 version 都只有一个 use，且该 use 是 DPS-init use。
     不满足（destination 分叉 / 覆写后的迟到 read）则报错退出——这种 IR 在 tensor 层本来就需要
     两份 buffer，正确的修法是让 MaterializeTensorStorage 按 destination use 而不是按 value promote。
     同一个 version 被多个 op 当 ins 读（读分叉）不受影响，仍然只要一份
```

这条路径没有 unroll、没有 prologue、没有谓词，代码体积不变。
**大部分循环内 allocation 走的是这条路**，rotation 只用于真正需要 DMA/compute 重叠的 stream。
T9 里的 L1 output slot 和 L2 output staging slot 就是这条路径（bufferize 后各 1 份）。

因为 allocation 的份数不由本路径决定（总数由 `MaterializeTensorStorage` 按 destination 数 1:1 决定，
每个 root 几份由 §4.6 的 `slotCount` 决定），**HoistOnly 不需要完整的 read-after-write conflict 分析**：
它只需要沿 DPS destination chain 做一次线性性检查，复杂度 O(链长)，不需要 alias、不需要全局 liveness、
不需要跨 root 的干涉图。真要「省 alloc」（同 shape、生命周期不重叠的 slot 共用物理内存）属于 arena
planner（下游文档 §9），真要「少 promote」属于 `MaterializeTensorStorage`。
完整论证、失败模式表与 lit 清单见 `SPM_HOIST_ONLY_EXAMPLE.md` §7.0 / §7.2 / §7.3.1。

### 5.8 后置条件（MaterializePipeline 出口 gate）

对应下游文档 §5.6，并补上可机械检查的判据：

```text
Q1. 任何 repetitive region 内不存在 memory_space 为 L1/L2 的 alloc_tensor。
Q2. 任何 repetitive region 内不存在 tensor.empty。
Q3. 每个原「每迭代一个逻辑 tile」的 allocation 都变成了固定 K 个 slot，K = slotCount ∈ {1,2}。
Q4. K 与重叠 stage 的最大并发实例数一致（§4.6 的公式重算一遍，结果必须相同）。
Q5. 同一时刻可能被 DMA / compute 同时使用的 slot 是不同的 allocation root（§5.1 下 v0 平凡成立）。
Q6. 每个 copy 的 result 都通过 SSA / iter_args 传到了消费者（R1）。
Q7. 每个 slot 在 iter_args 里位置稳定（R2）；每个谓词化 copy 的两分支 yield 同一 slot（R3）。
Q8. prologue / 尾部覆盖了少于 K 次迭代和非整除尾部。
Q9. 所有 spm.pipeline / spm.stage / spm.cluster 标注已删除。
Q10. 对最终 IR 重跑一次 §3.1.1 的容量核算（下游文档 §5.6 第 7 条）。
```

---

## 6. 完整实例：两级流水的 GEMV（实测通过）

场景（缩小到便于实测的尺寸，结构与下游文档 §10 一致）：

```text
input  : tensor<1x64xf16>      DRAM，小、循环不变 -> L2 和 L1 各常驻一份
weight : tensor<64x512xf16>    DRAM，完整 weight 永不进 SRAM
output : tensor<1x512xf16>     DRAM

L2 N-tile   = 128  -> L2 weight tile    = 64x128
L1 N-subtile = 32  -> L1 weight subtile = 64x32
两层都 slotCount = 2；output 路径 L1 -> L2 -> DRAM，slotCount = 1
```

### 6.1 pre-bufferize IR（这是本文要交付的「bufferize 之前 IR 应该长什么样」）

下面是**实测文件 T9** 的骨架（完整文件见 §9；这里省略重复的第二个 half 的内部细节，
用 `// … 与 half A 对称` 标出）。为了和下游文档一致，这里写成
`#spm.memory_space<l1|l2>`；实测时用的是等价的整数 space `1`/`2`（因为该 attribute 尚未实现）。

```mlir
func.func @gemv_2tier(%in_mem: memref<1x64xf16, #spm.memory_space<dram>>,
                      %w_mem: memref<64x512xf16, #spm.memory_space<dram>>,
                      %out_mem: memref<1x512xf16, #spm.memory_space<dram>>) {
  %in = bufferization.to_tensor %in_mem restrict : memref<...> to tensor<1x64xf16>
  %w  = bufferization.to_tensor %w_mem restrict  : memref<...> to tensor<64x512xf16>

  // ---- resident input：DRAM -> L2 -> L1，只搬一次，在所有循环之外 ----
  %in_l2_s = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l2>}> : tensor<1x64xf16>
  %in_l2   = bufferization.materialize_in_destination %in in %in_l2_s : (...) -> tensor<1x64xf16>
  %in_l1_s = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l1>}> : tensor<1x64xf16>
  %in_l1   = bufferization.materialize_in_destination %in_l2 in %in_l1_s : (...) -> tensor<1x64xf16>

  // ---- 所有 slot，全部在两层循环之外（§5.2）----
  %w2a = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l2>}> : tensor<64x128xf16> // L2 stream slot 0
  %w2b = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l2>}> : tensor<64x128xf16> // L2 stream slot 1
  %w1a = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l1>}> : tensor<64x32xf16>  // L1 stream slot 0
  %w1b = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l1>}> : tensor<64x32xf16>  // L1 stream slot 1
  %o1s = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l1>}> : tensor<1x32xf16>   // L1 out, slotCount=1
  %o2s = bufferization.alloc_tensor() <{memory_space = #spm.memory_space<l2>}> : tensor<1x32xf16>   // L2 staging, slotCount=1

  // ---- 外层 prologue：L2 slot 0 <- weight[:, 0:128] ----
  %wt0  = tensor.extract_slice %w[0, 0] [64, 128] [1, 1] : tensor<64x512xf16> to tensor<64x128xf16>
  %w2a0 = bufferization.materialize_in_destination %wt0 in %w2a : (...) -> tensor<64x128xf16>

  // ================= 外层（L2 级）2-way unrolled steady state =================
  %R:6 = scf.for %n = %c0 to %c512 step %c256                     // step = 2 * 128
      iter_args(%W2A = %w2a0, %W2B = %w2b,                        // L2 slot 版本，位置固定
                %W1A = %w1a,  %W1B = %w1b,                        // L1 slot 版本穿过外层
                %O1  = %o1s,  %O2  = %o2s)
      -> (tensor<64x128xf16>, tensor<64x128xf16>, tensor<64x32xf16>,
          tensor<64x32xf16>, tensor<1x32xf16>, tensor<1x32xf16>) {

    // ---------- 外层 half 0：先发 L2 预取（和下面整个内层循环重叠），再算 W2A ----------
    %nA   = arith.addi %n, %c128 : index
    %wtA  = tensor.extract_slice %w[0, %nA] [64, 128] [1, 1] : tensor<64x512xf16> to tensor<64x128xf16>
    %W2B1 = bufferization.materialize_in_destination %wtA in %W2B : (...) -> tensor<64x128xf16>

    // 内层 prologue：L1 slot 0 <- W2A[:, 0:32]
    %sA0  = tensor.extract_slice %W2A[0, 0] [64, 32] [1, 1] : tensor<64x128xf16> to tensor<64x32xf16>
    %W1A0 = bufferization.materialize_in_destination %sA0 in %W1A : (...) -> tensor<64x32xf16>

    // ---- 内层（L1 级）2-way unrolled steady state ----
    %IA:4 = scf.for %nn = %c0 to %c128 step %c64                  // step = 2 * 32
        iter_args(%P = %W1A0, %Q = %W1B, %OP = %O1, %SP = %O2)
        -> (tensor<64x32xf16>, tensor<64x32xf16>, tensor<1x32xf16>, tensor<1x32xf16>) {

      // 内层 half 0：先发 L1 预取，再在 P 上算
      %m1 = arith.addi %nn, %c32 : index
      %s1 = tensor.extract_slice %W2A[0, %m1] [64, 32] [1, 1] : tensor<64x128xf16> to tensor<64x32xf16>
      %Q1 = bufferization.materialize_in_destination %s1 in %Q : (...) -> tensor<64x32xf16>

      %OPz = linalg.fill ins(%zero : f16) outs(%OP : tensor<1x32xf16>) -> tensor<1x32xf16>
      %OP1 = linalg.matmul ins(%in_l1, %P : tensor<1x64xf16>, tensor<64x32xf16>)
             outs(%OPz : tensor<1x32xf16>) -> tensor<1x32xf16>
      %SP1 = bufferization.materialize_in_destination %OP1 in %SP : (...) -> tensor<1x32xf16>   // L1 -> L2
      %colA = arith.addi %n, %nn : index
      %dstA = memref.subview %out_mem[0, %colA] [1, 32] [1, 1] : memref<...> to memref<1x32xf16, strided<[512,1], offset: ?>, ...>
      bufferization.materialize_in_destination %SP1 in writable %dstA : (...) -> ()             // L2 -> DRAM

      // 内层 half 1：预取被谓词化（§5.5），再在 Q1 上算
      %m2  = arith.addi %nn, %c64 : index
      %okI = arith.cmpi slt, %m2, %c128 : index
      %P1  = scf.if %okI -> (tensor<64x32xf16>) {
        %s2 = tensor.extract_slice %W2A[0, %m2] [64, 32] [1, 1] : tensor<64x128xf16> to tensor<64x32xf16>
        %mm = bufferization.materialize_in_destination %s2 in %P : (...) -> tensor<64x32xf16>
        scf.yield %mm : tensor<64x32xf16>
      } else {
        scf.yield %P : tensor<64x32xf16>                      // R3：两分支 yield 同一 slot
      }
      // … 与 half 0 对称：fill / matmul(Q1) / L1->L2 / L2->DRAM

      scf.yield %P1, %Q1, %OQ1, %SQ1 : ...                    // R2：位置固定
    }

    // ---------- 外层 half 1：L2 预取谓词化，然后在 W2B1 上再跑一个内层 loop ----------
    %nB   = arith.addi %n, %c256 : index
    %okO  = arith.cmpi slt, %nB, %c512 : index
    %W2A1 = scf.if %okO -> (tensor<64x128xf16>) {
      %wtB = tensor.extract_slice %w[0, %nB] [64, 128] [1, 1] : tensor<64x512xf16> to tensor<64x128xf16>
      %mm  = bufferization.materialize_in_destination %wtB in %W2A : (...) -> tensor<64x128xf16>
      scf.yield %mm : tensor<64x128xf16>
    } else {
      scf.yield %W2A : tensor<64x128xf16>
    }
    // 内层 prologue（第二份）+ 第二个内层 loop %IB，iter_args 从 %IA#0..3 接下去
    // … 与 half 0 对称

    scf.yield %W2A1, %W2B1, %IB#0, %IB#1, %IB#2, %IB#3 : ...
  }
  return
}
```

注意这份 IR 里体现的每一条设计决策：

- 完整 weight 只在 DRAM；L2 只见 `64x128` tile，L1 只见 `64x32` subtile（不同颗粒度，下游文档 §4.1）；
- 所有 `alloc_tensor` 都在两层循环之外，L1 slot 穿过外层 `iter_args` 进内层；
- 两层各自 2-way unroll，残差静态绑定 slot；
- 预取一律在本 half 的 compute **之前**发（§4.1 的 cluster 约定）；
- 越界预取用 `scf.if` 谓词化，两分支 yield 同一 slot（R3）；
- output 走 L1 → L2 → DRAM 两跳，从不 L1 → DRAM 直连；
- output 两个 slot 的 `slotCount` 都是 1（hoist-only，§5.7），不是所有东西都 double buffer。

### 6.2 bufferize 之后（实测输出）

```text
$ build/bin/mlir-opt t9_nested.mlir -one-shot-bufferize
```

函数顶层恰好 8 个 allocation，**两个 loop body 内 0 个 allocation**：

```mlir
%alloc   = memref.alloc() alignment = 64 : memref<1x64xf16, 2>     // L2 resident input
%alloc_0 = memref.alloc() alignment = 64 : memref<1x64xf16, 1>     // L1 resident input
%alloc_1 = memref.alloc() alignment = 64 : memref<64x128xf16, 2>   // L2 weight slot 0
%alloc_2 = memref.alloc() alignment = 64 : memref<64x128xf16, 2>   // L2 weight slot 1
%alloc_3 = memref.alloc() alignment = 64 : memref<64x32xf16, 1>    // L1 weight slot 0
%alloc_4 = memref.alloc() alignment = 64 : memref<64x32xf16, 1>    // L1 weight slot 1
%alloc_5 = memref.alloc() alignment = 64 : memref<1x32xf16, 1>     // L1 out (slotCount = 1)
%alloc_6 = memref.alloc() alignment = 64 : memref<1x32xf16, 2>     // L2 staging (slotCount = 1)
```

`memref.copy` 共 19 条，全部是设计里预期的搬运（3 条循环外 + 外层每迭代 16 条），
**One-Shot 没有插入任何一条额外 copy**。这正是下游文档 §8.2 要求的
`number of implicit SRAM allocation decisions == 0`，也是下游文档 §8.3 那个回归测试的可执行形式
（只是 slot 用独立 root 而不是 ring，见 §8.2）。

### 6.3 容量核算

```text
L2 = 1x64xf16 (resident input, 128 B)
   + 2 x 64x128xf16 (weight slots, 2 x 16 KiB)
   + 1x32xf16 (staging, 64 B)
   + alignment/bank padding
L1 = 1x64xf16 (resident input, 128 B)
   + 2 x 64x32xf16 (weight slots, 2 x 4 KiB)
   + 1x32xf16 (out, 64 B)
   + alignment/bank padding
```

按 §3.1.1，`TileSizeOracle` 在选 L1 N-subtile = 32 时看到的 weight 预算已经是
`2 x alignedBytes(64x32xf16)`，而不是一份。下游文档 §10 的反例（把 512 KiB 的 L2 tile 原样放进 L1）
在这个模型里根本不会被选出来，因为 oracle 的预算里那一项已经乘了 2。

---

## 7. Pre-bufferize IR 契约（`VerifyPreBufferizeForm`）

进入 One-Shot Bufferize 之前必须全部通过。每条都是机械可检查的。

### 7.1 类型与 placement（引用下游文档 §2、§3，不重述）

```text
V1.  函数体内只有 builtin tensor / memref；没有 tier-specific type，没有 tier encoding。
V2.  每个被实体化的 tensor 都能唯一追溯到一个 alloc_tensor 或一个 to_tensor 的 DRAM ABI root。
V3.  每个 alloc_tensor 都有显式 memory_space。
     （实测：one-shot-bufferize 的 must-infer-memory-space=true 在缺失时会硬报
      "could not infer memory space" + "failed to bufferize op"，所以这条也可以直接由 bufferize 兜底，
      但应在 verifier 里提前失败以给出更好的诊断。）
V4.  所有跨 space 的内容迁移都由 materialize_in_destination 表达；没有跨 space 的 insert_slice。
V5.  没有 DRAM <-> L1 直连的 materialize。
V6.  compute op 的所有 operand 和 outs 都由 L1 root 支撑（§3.6.1 的次序陷阱就是靠这条兜住）。
V7.  控制流 join 不合并不同 space 的 root。
V8.  没有 alloc_tensor() copy(...) 的紧凑形式。
```

### 7.2 静态性与 allocation 位置（本文新增的部分）

```text
V9.  每个 alloc_tensor 的 result type 完全静态（§3.3）。
V10. 任何 repetitive region（scf.for / scf.while / 任何 RegionBranchOpInterface 的重复 region）内
     不存在 memory_space 为 L1/L2 的 alloc_tensor。          == Q1
V11. 任何 repetitive region 内不存在 tensor.empty。           == Q2
V12. 每个 L1/L2 allocation root 都带 slotCount ∈ {1, 2}，
     且 sum over roots of (slotCount * alignedBytes) 满足每层容量。
V13. 每个 slot 组的成员是不同的 allocation root（v0）。       == Q5
```

### 7.3 SSA version threading（本文新增，对应 §5.4）

```text
V14. 对每个 materialize_in_destination c：c 的 destination 在 c 之后没有被除 c.result 之外的
     use 读到（否则就是 R1 违规）。
V15. 对每个承载 slot 的 iter_arg 位置 j：yield 到位置 j 的值，其 allocation root 与
     iter_arg j 的 init 的 root 相同（R2）。
V16. 对每个 yield tensor 的 scf.if：所有分支 yield 的值 root 相同（R3）。
V17. 对每个承载 slot 的 iter_arg 位置 j：yield 到 j 的值是该 root 版本链的**终端** DPS result。
     （V15 的「同 root」挡不住 yield 中间版本——fill 的 result 与 matmul 的 result root 相同。）
V18. 每个 L1/L2 allocation root 与它的使用点之间的路径上不存在 scf.forall 或其他无序并行 region：
     跨越它 hoist 会让并发迭代共享同一份 buffer，而 V10 和「循环内 0 alloc」都检查不出来（§5.2）。
```

### 7.4 建议的兜底检查

`VerifyPreBufferizeForm` 通过之后，再跑一次 `one-shot-bufferize` 的 analysis-only
（`test-analysis-only=true`，配合 `print-conflicts`）。如果分析报告仍需新增 L1/L2 allocation，
说明 §5 的改写有 bug——在这里硬失败，不允许最终 bufferize 静默生成循环内 `memref.alloc`
（下游文档 §6 末段、§11 不变量 19）。

---

## 8. 与下游文档的接口

### 8.1 本文完全遵守的部分

下游文档 §2（tensor placement normal form）、§3（placement provenance）、§7（容量模型与 spilling）、
§8（One-Shot 契约）、§9（bufferize 后的静态分配）本文一字不改，只作为前置/后继引用。
下游文档 §11 的 20 条不变量，本文的 §7 契约是其中第 1–14 条在 pre-bufferize 阶段的可检查化。

### 8.2 需要修正的一处：§5.3 的 ring 表示

**下游文档 §5.3 推荐的形态在真实流水调度下不成立**，这是本文唯一与它冲突的结论，证据在 §5.1
（实测 T1c、T1g）。具体地，§5.3 的这段：

> 推荐把 K 个 slot 打包成一个带前导 slot 维度的 allocation …
> `%ring` 位于相关循环之前，`%ping` 和 `%pong` 是静态可证不相交的两个 subset。
> 这样静态 planner 只看到一个大小明确的 allocation root，One-Shot 则把两个 slice 降成同一个
> memref 的两个 disjoint subview。

最后一句在「两个 slice 只作为 iter_args 初值、且循环内各自被写」的情况下**不成立**：
One-Shot 会为其中一个 slot 另开一份 buffer 并插入一次 copy。根因是 tensor 值语义——
往 `%ping` 的 buffer 里写等于覆写 `%ring` 的内容，而 `%pong` 这个对 `%ring` 的读还活着。

建议的修正（不改变 §5.3 之外的任何内容）：

```text
v0 canonical slot 表示 = K 个独立的 alloc_tensor root，各自作 iter_arg。
「K 个 slot 连续放置」降级为 arena planner 的 layout hint（spm.slot_group 属性），属于 §9 的范围。
若确实要保留单 ring root，则整个 ring 必须作为**单个** loop-carried tensor value，
slot 写用 tensor.insert_slice、读用 extract_slice 取最新版本（实测可行，T1f），
但这会让跨层边多出一种 op，与 §11 不变量 5 冲突。
```

§11 的不变量**一条都不用改**：没有任何一条要求 K 个 slot 共享一个 root；
§5.6 的条件 4（同时使用的 slot 必须物理不相交）在独立 root 下平凡成立；
§8.3 的回归测试只需把「一个 `memref.alloc` + 两个 `memref.subview`」改成
「两个 `memref.alloc`」，其判据（**循环体内不新增 `memref.alloc`**）不变——
§6.2 的实测就是这个判据的两层版本。

### 8.3 §6 pass pipeline 的细化版

```text
Canonical DPS tensor IR
  ↓
[Round L2]
  TileForTier(L2)                        §3
  NormalizeTileShapes(L2)                §3.3    <-- 本文新增的必需步骤
  EliminateEmptyTensors                  §3.6.1  <-- 必须在 Materialize 之前
  MaterializeTensorStorage(L2)           下游 §4.2
  PlanPipeline(L2)                       §4      （只标注）
  ↓
[Round L1]
  TileForTier(L1)                        §3
  NormalizeTileShapes(L1)                §3.3
  EliminateEmptyTensors
  MaterializeTensorStorage(L1)           下游 §4.2
  PlanPipeline(L1)                       §4      （只标注）
  ↓
ResolveTensorConflicts                   下游 §6
  ↓
PlanStorageAndSpills ↔ PlanPipeline      下游 §5.2、§7（单调迭代；新增 copy 回到 §4.2）
  ↓
MaterializePipeline                      §5      （一次，由内向外，§5.6）
  ↓
VerifyPreBufferizeForm                   §7
  ↓
one-shot-bufferize analysis-only 兜底     §7.4
  ↓
One-Shot Bufferize                       下游 §8
  ↓
buffer-deallocation-pipeline + optimize-allocation-liveness
  ↓
StaticMemoryPlan（per memory space arena）下游 §9
  ↓
跨空间 memref.copy -> DMA；后端 lowering  下游 §9.4
```

与下游文档 §6 的差异只有三处，都是新增而非改写：`NormalizeTileShapes`、
`EliminateEmptyTensors` 的强制位置、`MaterializePipeline` 的「一次 + 内→外」。

---

## 9. 实验记录（可复现）

所有实验用 `build/bin/mlir-opt`（LLVM 24.0.0git，assertions on）。`memory_space` 用整数
（`1` = L1，`2` = L2），因为 `#spm.memory_space` 尚未实现；整数与 attribute 形式对 bufferize 行为无差别。
文件在本次会话的 scratchpad 目录下（`t*.mlir`），关键 IR 已内嵌在本文对应小节。

| #   | 测试内容                                                                                                               | 命令                                                   | 结果                                                                                                                            |
| --- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| T1  | 循环外 ring + 两个 slice 作 iter_args，**非流水顺序**（每个 slot 先写后读）                                      | `-one-shot-bufferize`                                | 通过：2 个 alloc（ring + acc），ring 降成 2 个静态 offset（0 / 2048）的 subview，循环内 0 alloc。但这个顺序不是真实流水，见 T1c |
| T1b | 两个独立 root，非流水顺序                                                                                              | 同上                                                   | 通过：3 个 alloc，循环内 0 alloc                                                                                                |
| T1c | 循环外 ring + 两个 slice 作 iter_args，**真实流水顺序**（half A 写 pong 读 ping） + prologue                     | 同上                                                   | **失败**：多出 `%alloc_0 : memref<32x64xf16,1>`，prologue 写到新 buffer                                                 |
| T1e | **两个独立 root**，真实流水顺序 + prologue + 谓词化尾部                                                          | 同上                                                   | **通过**：3 个 alloc（2 slot + acc），循环内 0 alloc、0 额外 copy → **v0 采用**                                    |
| T1f | 整个 ring 作单个 loop-carried value，`insert_slice` 写 slot                                                          | 同上                                                   | 通过：2 个 alloc，所有 slot 访问是静态 offset subview                                                                           |
| T1g | ring，prologue 改用`insert_slice`，再取两个 slice 作 iter_args                                                       | 同上                                                   | **失败**：仍多一份 alloc + 一次 copy                                                                                      |
| T2  | `alloc_tensor` 在循环体内                                                                                            | 同上                                                   | 循环内`memref.alloc() : memref<32x64xf16, 1>`（设计禁止的形态）                                                               |
| T3  | yield 写入前的 slot 版本                                                                                               | 同上                                                   | **硬错误**：`materialize_in_destination op not bufferizable … cannot avoid RaW conflict`                               |
| T3b | 交换两个 slot 的 yield 位置                                                                                            | 同上                                                   | **硬错误**：`Yield operand #0 is not equivalent to the corresponding iter bbArg`                                        |
| T4a | 循环内`tensor.empty` 作 `linalg` 的 outs                                                                           | 同上                                                   | 循环内`memref.alloc()`，且 **memory space = 0**                                                                         |
| T4b | 同上，先`-eliminate-empty-tensors`                                                                                   | `-eliminate-empty-tensors`                           | destination 被换成`extract_slice %arg4[...]`（外层目标的 subset，可能在 L2/DRAM）                                             |
| T4c | `-eliminate-empty-tensors -one-shot-bufferize`                                                                       | 同上                                                   | `memref.alloc` 数量 = **0**                                                                                             |
| T6  | `scf.if` 按 `iv % 2` 在同一 ring 的两个 slice 间动态选 slot                                                        | `-one-shot-bufferize`                                | **硬错误**：`tensor.extract_slice op not bufferizable … cannot avoid RaW conflict`                                     |
| T7  | `alloc_tensor` 缺 `memory_space`                                                                                   | `-one-shot-bufferize="must-infer-memory-space=true"` | **硬错误**：`could not infer memory space` + `failed to bufferize op`                                                 |
| T8  | 谓词化预取：`scf.if` then 分支 materialize、else 分支原样 yield slot                                                 | `-one-shot-bufferize`                                | **通过**：两分支都 yield 同一 memref，0 alloc、0 额外 copy → v0 不需要静态 trip count                                    |
| T9  | **两层嵌套**：L2 外层 + L1 内层各 2-way unroll，L1 slot 提到外层之外、穿过两层 iter_args，output 走 L1→L2→DRAM | 同上                                                   | **通过**：函数顶层恰好 8 个 alloc，两个 loop body 内 0 个 alloc，19 条 copy 全部为预期搬运                                |
| T10 | 两级`transform.structured.tile_using_for`（`[64,64]` 然后 `[32,32]`），shape 不整除                              | `-transform-interpreter`                             | tile 类型是`tensor<?x64xf32>` / `tensor<?x?xf32>`，内层 loop 上界为动态 `affine.min` → §3.3 的规范化是必需的            |

源码依据（本次核对过的行号，LLVM 侧为本仓库，Triton 侧为 `../triton`）：

```text
mlir/include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h:46-141   SCFTilingOptions
                                                              :279-311 SCFTileAndFuseOptions / fusionControlFn
mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h                  PipeliningOption 全字段
mlir/lib/Dialect/SCF/Transforms/LoopPipelining.cpp:99-205               initializeLoopInfo 的全部前置条件
                                                  :396-460, :595-632    rotation / iter_arg 版本机制
mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td:272 transform.structured.promote_tensor
mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp:443-495     其实现 = alloc_tensor + 条件 materialize
mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td:171       transform.loop.peel
mlir/include/mlir/Dialect/Tensor/TransformOps/TensorTransformOps.td:131 transform.tensor.make_loop_independent
mlir/include/mlir/Dialect/Bufferization/IR/BufferizationOps.td:195-231  materialize_in_destination 语义
mlir/include/mlir/Dialect/Bufferization/Transforms/Passes.td:145        must-infer-memory-space

../triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp:175-197  pass 顺序
../triton/.../Pipeliner/AssignLatencies.cpp:127                        loadLatency 公式
../triton/.../Pipeliner/ScheduleLoops.cpp:36-48                        isSafeToPipeline（含"不流水外层循环"）
                                          :52-99                       scheduleDistanceOneDependencies
                                          :100-140                     scheduleRemainingToLastStage
                                          :152-247                     scheduleKeyOps（最长路径 -> stage）
                                          :351-386                     scheduleLoop 驱动顺序
../triton/.../Pipeliner/LowerLoops.cpp:356, :376, :443                 numBuffers = stageDiff
../triton/.../Pipeliner/PipeliningUtility.cpp:410-427                  createAlloc：leading dim = distance
```

---

## 10. v0 边界与后续

v0 明确支持：

- 两轮 tiling（L2、L1），`scf.for`，tile-and-fuse，静态 tile 或 peel 后静态；
- `slotCount ∈ {1, 2}`，`lookahead = 1`（= double buffer）；
- rotation（2-way unroll）与 hoist-only 两条路径，自动分类（§4.6）；
- 嵌套两层流水，内层 slot 穿过外层 `iter_args`；
- 尾部 peel 或谓词化，trip count 可以是动态的；
- L1 → L2 → DRAM 的 output staging。

后续再做（都不需要推翻本文的 IR 主轴）：

| 后续项                                                     | 需要改什么                                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `lookahead > 1` / `slotCount > 2`                      | §4.6 的公式已经是一般式；`ExpandLoop` 的 2-way unroll 变 K-way；容量按 K 计费     |
| 动态`%iv mod K` 选 slot                                  | 需要跨迭代 subset disjoint 证明；届时可直接改用上游`scf::pipelineForLoop`（§4.9） |
| 间接寻址 / gather 的流水                                   | `canPrefetch` 的 C2 放宽，参考 Triton 的 `maxIndirectionLevel` 摊薄              |
| `scf.forall` 上的流水                                    | 需要先定义并行语义下的「跨迭代预取」，v0 直接禁止（§3.7）                           |
| 异步 DMA 的 token / wait 进入 reservation                  | 下游文档 §7.2、§9.4；本文的 slot reservation 已经按「保留到 transfer 完成」设计    |
| 非整除尾块的 mask/predication lowering                     | §3.3 的 S2/S3，目前只定义了形态没有定义 mask op                                     |
| 同 shape slot 的物理复用（单次迭代内生命周期不重叠）       | hoist 把 live range 拉长到整个嵌套，arena planner 需要 per-iteration liveness ＋「读前整块覆写」判定；v0 高估峰值，靠 §4.8 单调迭代兜（`SPM_HOIST_ONLY_EXAMPLE.md` §7.6） |
| 跨 loop 的 slot 复用（顺序执行的两个 loop 共享 slot root） | §5.2 已经允许（各自在自己前面 alloc），靠 arena planner 复用 offset                 |

---

## 11. 建议实现顺序

每一步都带一个可执行 gate，前一步禁止的形态在后一步必须立刻报错，不留给后续 pass 猜。

```text
 1. VerifyPreBufferizeForm 的 V1–V16（§7）先写。它是所有后续步骤的裁判。
    gate：拿 §6.1 的 T9 IR 当正例，T1c/T2/T3/T3b/T4a/T6/T7 当反例，全部跑通。
 2. TileForTier 的骨架：包 tileConsumerAndFuseProducersUsingSCF，把两个 oracle 接上。
    gate：两轮之后 §3.5 的 P1–P6 通过。
 3. NormalizeTileShapes：先只实现 S1（peel）。
    gate：T10 那种不整除的输入，主循环的 tile 类型全静态。
 4. EliminateEmptyTensors 的位置固定 + §3.6.1 的次序检查。
    gate：V6（compute outs 是 L1 root）在 eliminate 之后仍然成立。
 5. MaterializeTensorStorage：抽 transform.structured.promote_tensor 的实现为公共 utility。
    gate：纯 output promotion 只有 alloc、没有初始 copy（下游文档 §12 的 Output-only 行）。
 6. PlanPipeline 的 candidate discovery + canPrefetch + latency（§4.2、§4.3）。
    gate：标注正确性用 lit 测试比对 spm.stage / spm.slot_count。
 7. PlanPipeline 的 stage 分配（§4.4，照抄 Triton 的最长路径）+ §4.5 的 bail-out。
    gate：L1–L7 每条都有一个 lit 测试证明「不流水但仍然合法」。
 8. MaterializePipeline 的 HoistOnly 路径（§5.7）。这是最简单也最常用的一条。
    gate：V10/V11/V17/V18 通过，且 bufferize 后循环内 0 alloc；alloc_tensor 进出数量 1:1；
          不跨 scf.forall 提升（SPM_HOIST_ONLY_EXAMPLE.md §7.3 的 G6-G8）。
 9. MaterializePipeline 的 ExpandLoop（§5.3），先只做 tailPolicy = peel。
    gate：单层的 T1e 形态，bufferize 后 alloc 数 = slot 数 + 其他，循环内 0 alloc。
10. tailPolicy = predicate（§5.5）。
    gate：动态 trip count 的输入也能流水，且 bufferize 后无额外 copy（T8 形态）。
11. 内→外的嵌套展开（§5.6）。
    gate：T9 的两层形态，L1 slot 恰好 2 份（不是 4 份）——这是 §2.3 论证的可执行判据。
12. 接容量 planner 的单调迭代（§4.8），把 spill 新增的 copy 送回 §4.2。
    gate：depth 2 -> 1 的回退路径有测试；下游文档 §12 容量那一行通过。
```

对应下游文档 §13 的实现顺序：本文的第 2–5 步落在它的第 3–5 步，第 6–11 步落在它的第 7–8 步，
第 1 步落在它的第 2 步。两份顺序不冲突，本文只是把「pipeline」那两步展开成了 6 小步。
