# Shared-Memory Allocation 设计抽取：可移植到 NPU SRAM 分配器的核心部件

先说结论：Triton 的 SMEM 分配链路里，**跟 GPU 强绑定的东西其实很少**——绝大部分核心算法（`Allocation.cpp`/`Alias.cpp`）本身就是纯粹的图论/区间算法，不引用任何 CUDA/LDS 概念。真正 GPU-specific 的部分，全部集中在两处：

1. `defaultAllocationAnalysisScratchSizeFn` 里对 `ConvertLayoutOp`/`ReduceOp`/`WarpSpecializeOp` 等具体算子的字节数计算；
2. `partitionSize`/`neighbors`/`AsyncRegions` 这套为 warp-specialize + LDS bank 分区设计的可选分支。

把这两块摘掉之后，剩下的就是可以直接搬到 NPU software-managed SRAM 分配器上的东西。

下面按"可以几乎照抄的算法核心 → IR 层的两个 op → 两个分析（alias / liveness） → 排除清单 → 建议的 NPU 端骨架"组织。所有引用都带文件路径，方便直接去源码里对照。

---

## 1. Buffer 抽象：`BufferT`

`include/triton/Analysis/Allocation.h:170-198` 里的核心数据结构，完全硬件无关：

```cpp
struct BufferT {
  enum class BufferKind { Explicit, Scratch, Virtual };
  BufferKind kind;
  BufferId  id;
  Operation *owner;   // 产生/拥有这块内存的 op
  size_t size;
  size_t alignment;
  size_t offset;      // 分配算法要填的结果
  SmallVector<BufferT*> neighbors;  // 分区专用，v0 不需要，见第 5 节
};
```

三种 kind，对应要移植的场景：

- **Explicit** —— 由用户可见的 `local_alloc` 产生，大小直接来自 memdesc 类型（shape × dtype 位宽），这是 v0 唯一需要的一种。
- **Scratch** —— 某些算子在 lowering 时临时需要一块内存（不是用户写的 `local_alloc`，是编译器插的），大小由一个可插拔函数 `AllocationAnalysisScratchSizeFn` 给出。**这个"某算子需要临时 buffer"的抽象本身是通用的**，但 Triton 默认实现里给出的具体大小公式（reduce/scan/gather/convert_layout 的 swizzling 字节数……）全部是 GPU layout 相关，要删掉换成自己的。v0 可以先不接任何算子（函数直接返回 0），需要时再一个个加。
- **Virtual** —— 代表一次 `tt.call`，把被调函数的整个 SMEM 占用打包成"调用点上的一块 scratch"，从而让跨函数调用的栈式布局能纳入同一套区间算法（`lib/Analysis/Allocation.cpp:270-277`）。**只有当 NPU IR 存在"一个 kernel 里多个函数互相调用、且共用一块 SRAM 预算"时才需要**；如果每个 kernel 编译单元本身就是一个平坦的函数，直接不实现这一种即可。

## 2. IR 层：`local_alloc` / `local_dealloc`

定义在 `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:154-218`。这里有一个**极其关键、容易移植错的点**：

> `local_dealloc` 是**可选**的。文档原话："If you don't explicitly dealloc a buffer, the compiler assumes it's deallocated at the first point that post-dominates all uses of the alloc."

也就是说——**决定一块 buffer 什么时候"死"的，从来不是扫描 `local_dealloc` 这个 op，而是标准的 SSA use-def 活跃区间分析**（下一节的 `mlir::Liveness`）。`local_dealloc` 存在的意义只是：

1. 给用户一个显式提前释放的手段（比如提前腾出空间给下一块大 buffer，即使 SSA 值理论上还没到 post-dominate 点）；
2. 配合 `MemFree<SharedMemory>` 的 side-effect 声明，让 CSE/DCE 之类的通用 MLIR pass 知道"这之后用这个 memdesc 是 UB"，是一个正确性/别名工具，不是分配算法的输入。

**移植建议**：NPU 的 `npu.local_alloc` / `npu.local_dealloc` 完全可以照搬这个语义划分——`local_dealloc` 只做验证/提示用途，真正喂给分配算法的活跃区间必须来自下面的 liveness 分析，不要设计成"扫 dealloc 位置来定生命周期结束"，否则一旦用户（或者自己的 lowering pass）忘记插 dealloc，会悄悄分配出错误重叠的 buffer。

其余字段基本可以直接照搬：`alignment`（可选 attr，缺省有默认值）、`src`（可选初始化值，没有则要求结果必须是 `mutable`）、`isSharedMemoryAlloc()`（用内存空间 attr 区分"这是不是要分配的那种内存"——NPU 上换成自己的 memory-space attr 判断即可）。

## 3. Alias 分析

`include/triton/Analysis/Alias.h` + `lib/Analysis/Alias.cpp`，一个基于 MLIR dataflow 框架的**前向 sparse 分析**，格（lattice）是 `AliasInfo = 一个 SSA 值可能来自哪些 local_alloc 的集合`。之所以需要它：一个 memdesc 值不一定字面上就是 `local_alloc` 的结果——它可能是从 `scf.for` 的 `iter_args`/`scf.yield` 传出来的、从 `arith.select` 两个分支里选出来的、或者是某个"view"类 op（切片/reinterpret）包了一层。如果不追踪这些别名，liveness 分析会漏掉真正的存活范围。

规则非常短（`lib/Analysis/Alias.cpp:23-64`），完全和 GPU 无关，可以整体照搬逻辑：

```
只有 local_alloc 产生新的 alloc-id（种子）
带 MemDescViewTrait 的 op（切片/子视图类）→ 直接继承 operand[0] 的别名集合
select 类 op → 取两个分支别名集合的并集
poison → 空集合
其他任何返回同种"共享内存 memdesc 类型"结果的 op → 视为未知/最保守（触发断言，
    意味着"新增了一种能产生 memdesc 的 op，必须显式教会这个分析怎么处理它"）
```

**移植建议**：这个"未知 op 触发断言"的兜底设计值得保留——它逼着每加一种能产生/传递 SRAM 描述符的新 op 时，都必须显式决定它是种子、透传、合并还是保守，而不是被动漏掉。

## 4. Liveness 分析

在 `lib/Analysis/Allocation.cpp:414-455`。分两层，都是通用算法：

1. **后序编号**：对根 op 做一次 `WalkOrder::PostOrder` 遍历，给每个 op 分配递增 ID。这一步的意义是保证"父 op（比如整个 `scf.for`）的 ID 一定大于它所有子 op 的 ID"，这样如果一个值在循环体外定义、循环体内使用，它的活跃区间能正确地覆盖整个循环，而不会在子 op 遍历时提前"看起来死了"。
2. **区间计算**：直接用 **MLIR 自带的 `mlir::Liveness`**（`mlir/Analysis/Liveness.h`，完全 dialect-agnostic，不需要自己写）算出每个 SSA 值的"活跃于哪些 op"集合，取这些 op 里最小/最大后序号 → 得到 `[minId, maxId)` 区间。

三种 buffer 的区间来源不同，最后都汇总进同一张 `bufferRange` 表：

- **Explicit**：直接是该值自身的 `mlir::Liveness` 区间（`lib/Analysis/Allocation.cpp:344-360`）。
- **Alias 扩展**：如果一个值是某 buffer 的别名（第 3 节），把这个别名值自己的区间也 union 进那个 buffer 的区间（`lib/Analysis/Allocation.cpp:365-380`）——这就是为什么循环里被当成 iter_arg 传递的 buffer，它的区间会被正确地撑到整个循环范围。
- **Scratch**：不跑 `mlir::Liveness`，直接定义为"只活在拥有它的那个 op 执行的瞬间"，即 `[opId, opId+1)`（`lib/Analysis/Allocation.cpp:399-402`）——因为这类 buffer 从头到尾都是编译器内部产物，一次用完就扔。

**移植建议**：这一整块（后序编号 + `mlir::Liveness` + 三路区间合并）**可以几乎原样照搬**，因为它只依赖 MLIR 核心设施和自己的 `BufferT`/`AliasInfo`，不涉及任何 GPU 概念。

## 5. Offset 分配算法（图着色）

`lib/Analysis/Allocation.cpp:498-799`，实现的是论文 *Algorithms for Compile-Time Memory Optimization*，三段式，**同样是纯区间/图算法，和 GPU 无关**（前提是把 partition/neighbor 那几处摘掉，见下）：

**第一步 `calculateStarts`**：按 size 从大到小排序（大 buffer 先摆，减少被小 buffer 碎片化的概率），用一个 `multimap<offset, 可用活跃区间>` 的"空闲槽表"贪心摆放：每次取最低的空闲 offset，找出"整段生命周期都落在这个槽的可用区间内、且不会被更早的槽覆盖"的候选 buffer，选中一个摆进去，再把这个槽剩下的空闲区间（按被摆入 buffer 的生命周期切开）重新插回槽表。这一步只给出一个**初始猜测**，不保证没有重叠。

**第二步 `buildInterferenceGraph`**：两个 buffer 只要"物理 offset 区间相交"且"活跃区间也相交"，就连一条边（`lib/Analysis/Allocation.cpp:681-735` 里还有一段"AsyncRegions/同一 op 的并发子区域即使活跃区间理论上不重叠也要连边"的规则——这是给 warp-specialize 用的，NPU 如果有多个执行引擎并发跑同一个 op 的不同分支，未来可能用得上，v0 不需要）。

**第三步 `allocate`**：对着上面的冲突图做**贪心图着色**（first-fit：给每个节点找一个和它所有已着色邻居都不同的最小颜色编号），同色节点保证互不冲突；然后把每种颜色内、真正会冲突的节点依次往后推（推到"和它冲突的邻居的 offset+size 之后"）。

**收敛**：因为第三步的"往后推"可能让原本不重叠的 offset 区间变得重叠（制造出新的冲突），所以外层是一个 `do { allocate(); buildInterferenceGraph(); } while (!interference.empty())` 的不动点循环（`lib/Analysis/Allocation.cpp:521-527`）——每轮 offset 只增不减，冲突只减不增，源码里没有给出严格的收敛证明/上界，只是注释说"最终会到达不动点"，工程上这么多年也确实一直够用。

**移植时要留意**：`calculateStarts`/`buildInterferenceGraph`/`allocate` 这套图着色算法本身完全不感知硬件容量，只管把 buffer 摆紧凑，不会因为"分配到多少字节"而做任何判断。Triton 里真正的容量检查在完全不同的层——`python/triton/compiler/compiler.py:470-472`，在 kernel 加载时把 MLIR 模块上的 `ttg.shared`（`AllocateSharedMemory` pass 写入的值）和驱动查询到的硬件上限（`third_party/nvidia/backend/driver.c:163-167` 查询的 `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`）比较，超了就抛 `OutOfResources` 异常终止，**没有任何自动降级/spill 机制**，完全依赖用户自己改小 block size/num_stages 重新编译。也就是说，如果 NPU SRAM 容量比典型 GPU SMEM 小很多、或者不希望把"超预算"这种事甩给上层重新编译解决，那么无论是这种"编译后置校验+报错"的方式，还是更进一步的"编译期就感知容量、必要时 spill"的方式，都是 Triton 原设计里没有、需要自己另外设计的部分。

## 6. 明确不建议移植的部分（对应"其他复杂用法一律不要"）

| 内容 | 在哪 | 为什么不要 |
|---|---|---|
| `async_copy_global_to_local` / `async_commit_group` / `async_wait` | TritonGPUOps.td | cp.async 双缓冲流水线专用，和"分配"无关，是另一层软件流水优化 |
| TMA（`async_tma_copy_global_to_local`）+ mbarrier | third_party/nvidia | 硬件描述符 + 异步完成通知机制，Hopper 专属 |
| `Membar`（barrier 插入分析） | lib/Analysis/Membar.cpp | 解决的是"多线程/多 warp 观察顺序"问题，是同步语义而不是分配语义；如果 NPU 也有多个并发执行单元访问同一块 SRAM，这是**独立于分配之外的下一个课题**，不要和 offset 分配算法糅在一起 |
| `WarpSpecializeOp` / `AsyncRegions` trait / `neighbors` / `partitionSize` | Allocation.cpp 里散落的分支 | 服务于"同一块逻辑 buffer 必须物理落在不同 LDS bank 分区"这个 GPU-specific 需求；如果 NPU SRAM 确实分 bank 且有类似诉求，可以按同样思路后加，但不是 v0 该背的复杂度 |
| `defaultAllocationAnalysisScratchSizeFn` 里的具体算子分支 | Allocation.cpp:95-147 | Reduce/Scan/Gather/Histogram/ConvertLayout(swizzling)/Atomic/TensormapCreate 全部是 GPU layout/硬件相关的字节数公式，一个都不能直接搬 |
| `PartitionedSharedEncodingAttr` 相关 buffer 拆分逻辑 | Allocation.cpp:215-240 | 同上，分区专用 |
| `ModuleAllocation` 的调用图跨函数拼接 | Allocation.h:261-305 | 只有多函数共享一个 SRAM 预算时才需要，见第 1 节 Virtual buffer 部分 |

## 7. 建议的 NPU 端最小骨架

按这个顺序实现，每一步都能独立跑通、独立测试：

1. `npu.local_alloc` / `npu.local_dealloc`（严格照抄语义划分：dealloc 只做验证，不参与生命周期计算）。
2. `SRAMAliasInfo` + `SRAMAliasAnalysis`：直接照抄 `AliasInfo`/`SharedMemoryAliasAnalysis` 的 4 条规则，把 `LocalAllocOp` 换成 `npu.local_alloc`。
3. `Buffer`（对应 `BufferT`，先只做 `Explicit` 一种 kind，字段砍到 `id/owner/size/alignment/offset`）。
4. Liveness：后序编号 + 直接调用 `mlir::Liveness` + 三路区间合并（alias 那一路用第 2 步的分析结果）。
5. Offset 分配：`calculateStarts` → `buildInterferenceGraph` → `allocate` 不动点循环，几乎可以照抄，只是把 partition/neighbor 相关的判断整段删掉。
6. 补一道 Triton 没有的容量检查：`allocate()` 收敛后比较 `sharedMemorySize` 和 NPU SRAM 实际容量，超了就报错（或者先占位，等以后再决定要不要做 spill-to-DRAM）。
7. 需要给某个 NPU 专属算子（比如某种归约/重排）分配临时空间时，再引入 `Scratch` kind + 自己的 `ScratchSizeFn`，一个算子一个算子地加，不要一开始就照抄 Triton 那一大坨。

这样每一步都对应 Triton 里边界清晰的一小块代码，出问题时也容易回头对照原实现排查是算法搬错了还是 NPU 语义本身不一样。
