# Load/Store Op 与 MemDesc 类型设计抽取：可移植到 NPU 的核心部件

本文是 [NPU_SRAM_ALLOCATION_PORTING.md](NPU_SRAM_ALLOCATION_PORTING.md) 的姊妹篇、也是更早一步的基础篇。上一篇假设"`local_alloc`/`local_dealloc`/`memdesc` 已经存在"，只讲分配算法；这一篇往回退一步，把 **`!ttg.memdesc` 类型本身，以及在它上面操作的 op 家族（load/store/view/gather-scatter）** 扒开，回答"设计 NPU 自己的 `mydesc` 类型和 load/store op 时，哪些能照抄、哪些必须换掉"。

先说结论：和上一篇发现的规律一致——**IR 层的类型定义、op 定义、op 的 verifier，几乎完全和 GPU 无关**；真正 GPU-specific、大量硬编码 SIMT 概念（thread/warp/lane、bank 冲突、swizzle）的部分，**全部集中在 `TritonGPUToLLVM` 这一层的 lowering 代码里**，也就是"如何把一个逻辑上的 tensor↔shared memory 数据搬运，翻译成每个线程该读写哪个地址"这件事。IR 定义可以照抄，lowering 必须按 NPU 自己的执行模型重新设计。

---

## 1. `MemDescType`：六个字段拆解

定义在 `include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:23-85`：

```tablegen
def TTG_MemDescType : TTG_TypeDef<"MemDesc", "memdesc", [ShapedTypeInterface]> {
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType,
    "Attribute":$encoding,
    "Attribute":$memorySpace,
    "bool":$mutableMemory,
    ArrayRefParameter<"int64_t">:$allocShape
  );
}
```

逐个字段看要不要搬：

| 字段 | 含义 | 移植建议 |
|---|---|---|
| `shape` | 逻辑形状（这个 view 看到的形状） | 照抄，纯 shape 概念 |
| `elementType` | 元素类型 | 照抄 |
| `memorySpace` | 用哪种内存空间的 attr 区分（对应 GPU 的 `SharedMemorySpaceAttr`，`TritonGPUAttrDefs.td:1496`） | 照抄这个"用 attr 而不是硬编码 enum 区分内存空间"的设计，换成自己的 `NPUSRAMSpaceAttr` |
| `mutableMemory` | 是否可写（不可变的只能在 alloc 时初始化一次） | 照抄，纯语义概念，和硬件无关 |
| `allocShape` | **底层实际分配的完整形状**，与 `shape`（当前 view 看到的形状）分离 | **强烈建议保留**，这是支持"软件流水线多缓冲（multi-buffering）"的关键设计，见下 |
| `encoding` | 描述这块内存里数据的物理排布方式 | **这是全部复杂度的来源**，见第 5 节 |

**关于 `allocShape` vs `shape` 这个分离**（不要漏看，容易被当成"GPU 特有的奇怪字段"跳过）：`local_alloc` 出来的 buffer，`shape == allocShape`（`lib/Dialect/TritonGPU/IR/Ops.cpp:890-903` 的 `verifyAllocOp` 强制这一点）。但如果这块内存是一个"多缓冲"buffer——比如软件流水线里开了 3 份双缓冲，`allocShape = [3, M, N]`，通过 `memdesc_index` 取出第 `i` 份时结果类型是 `shape = [M, N]`，但 `allocShape` 仍然记录着完整的 `[3, M, N]`（`lib/Dialect/TritonGPU/IR/Ops.cpp:1128-1170` 的 `MemDescIndexOp::verify` 强制"只能对 `allocShape.size() == rank` 的顶层 buffer 做 index，不能对已经 index 过一次的再 index"）。**这个机制和 GPU 无关，是纯粹的"环形多缓冲区寻址"抽象**——如果 NPU 打算做软件流水线（比如 DMA 预取和计算重叠），这个字段是直接可以复用的设计，不要因为它出现在 GPU 代码里就以为是 GPU 专属概念。

**类型自身的 `verify()`**（`lib/Dialect/TritonGPU/IR/Types.cpp:91-131`）也基本通用，可以照抄的规则：
- `shape` 不能是 rank 0，不能有维度是 0；
- 元素位宽必须是 1 或者 ≥8（不允许诡异的 sub-byte 类型，除了 i1）；
- `allocShape` 维度数必须 ≥ `shape` 维度数；
- `encoding` 必须实现 `LayoutEncodingTrait`，且必须是"合法的内存 encoding 类型"之一——GPU 这里列了三种：`SharedEncodingTrait`（真正的 shared memory）、`TensorMemoryEncodingAttr`、`TensorMemoryScalesEncodingAttr`（后两种是 Hopper/Blackwell 的 Tensor Memory，NPU 不需要，直接不实现即可）；
- rank 检查：`SharedEncodingTrait` 场景下要求 `encoding.getRank() == shape.size()` 或 `shape.size() - 1`（后者对应"顶层多缓冲 buffer，encoding 的 rank 不含那个多缓冲维度"这一情况）。

## 2. Op 家族总览：谁是"load/store"，谁是"view"，谁可以先不做

`TritonGPUOps.td` 里所有围绕 `memdesc` 转的 op，按角色分成四组：

| 组别 | Op | 文件位置 | v0 是否需要 |
|---|---|---|---|
| **分配** | `local_alloc` / `local_dealloc` | `TritonGPUOps.td:154-218` | 需要（上一篇已覆盖） |
| **View（不搬内存，只换描述符）** | `memdesc_index` / `memdesc_subslice` / `memdesc_trans` / `memdesc_reshape` / `memdesc_reinterpret` | `TritonGPUOps.td:219-357` | **`memdesc_index`/`memdesc_subslice` 建议需要**（多缓冲、tile 切片场景很常见）；`trans`/`reshape`/`reinterpret` 可以按需后加 |
| **真正的数据搬运（load/store 本体）** | `local_load` / `local_store` | `TritonGPUOps.td:359-398` | **v0 核心，必须要** |
| **可选的花活** | `local_gather` / `local_scatter` / `local_atomic_scatter_rmw`（`TritonGPUOps.td:400-508`）、`async_copy_global_to_local`/`async_commit_group`/`async_wait`（`TritonGPUOps.td:48-151`） | 同上 | **v0 不需要**，见第 7 节 |

## 3. `local_load` / `local_store`：IR 层定义（可以整体照抄）

```tablegen
def TTG_LocalLoadOp : TTG_Op<"local_load", [LocalLoadTrait]> {
  let arguments = (ins
    Arg<TTG_MemDescType, "", GenericSharedRead>:$src,
    Optional<TTG_AsyncToken>:$token
  );
  let results = (outs TT_Tensor:$result);
  let hasVerifier = 1;
}

def TTG_LocalStoreOp : TTG_Op<"local_store"> {
  let arguments = (ins
    TT_Tensor:$src,
    Arg<TTG_MemDescType, "", GenericSharedWrite>:$dst
  );
  let hasVerifier = 1;
}
```

（`TritonGPUOps.td:359-398`）这里面完全没有任何 GPU 概念：

- **语义**：`local_load` 把一个 memdesc（"内存里的数据"）搬进一个 `TT_Tensor`（"寄存器/计算域里的数据"）；`local_store` 反过来。这个"内存域"vs"计算域"两种值类型的区分，是 Triton IR 的基本设计，NPU 上大概率也需要类似的两分（SRAM 里的 buffer vs. 参与计算指令的 operand），可以照抄这个建模思路。
- **`token` 操作数**（`local_load` 上可选）：用来接上一个异步操作（比如 `async_copy_global_to_local` 返回的 `AsyncToken`），表示"必须等这个 token 代表的异步搬运完成才能读"。如果 NPU v0 不做异步拷贝，这个操作数可以先留空/不接。
- **`GenericSharedRead`/`GenericSharedWrite`**（`TritonGPUMemoryEffects.td:20-23`）：只是给 MemDescType 操作数标注 MLIR 的 side-effect（读/写哪个 `Resource`），让通用的 MLIR pass（CSE/DCE/调度）知道不能把读写搬运指令乱序。这是纯 MLIR 机制，照抄。
- **`LocalLoadTrait`**：注意看它的 C++ 实现（`include/triton/Dialect/TritonGPU/IR/Traits.h:35-39`）——**这个 trait 是空的**，不做任何验证，纯粹是给 `Alias.cpp`、pipeliner 之类的分析 pass 用来"识别一个 op 是不是 local_load 家族"的标记（`local_load`/`local_gather` 都打了这个 tag）。移植时如果 NPU 也有多种"从 SRAM 读出到计算域"的 op，用同样的空 trait 打标签，方便后续别名/调度分析统一处理。

## 4. Verifier 清单：哪些通用，哪些依赖 encoding

`local_load`/`local_store` 的 `verify()`（`lib/Dialect/TritonGPU/IR/Ops.cpp:920-945`）调用了两个共享的 helper，都在同一个文件里（`Ops.cpp:875-918`），**全部和 GPU 无关，可以照抄**：

1. **`verifyMemoryOpTypes`**（`Ops.cpp:875-888`）：检查 `src`/`dst` 的元素类型和 shape 必须完全一致——纯粹的"内存搬运不改变逻辑内容"约束。
2. **`verifySharedMemoryRank`**（`Ops.cpp:905-918`）：检查 tensor 一侧的 rank 要等于 memdesc encoding 声明的 rank（`LayoutEncodingTrait::getRank()`）。这里唯一依赖的是 `LayoutEncodingTrait` 这个接口本身（"encoding 得知道自己的 rank"），不依赖具体是哪种 encoding。
3. `local_store` 额外检查 `getDst().getType().getMutableMemory()`——不能往不可变内存写，纯类型系统层面的约束。

再往上一层，还有两个跑在**所有** TTG op 上的通用 trait（`TTG_Op` 基类自带，`TritonGPUOps.td:27-30`）：

- **`VerifyMemDescLayoutsTrait`**（`lib/Dialect/TritonGPU/IR/Traits.cpp:51-97`）：遍历一个 op 所有 operand/result，如果类型带 `memdesc` 的 encoding，就调用该 encoding 所属 dialect 的 `DialectVerifyTensorLayoutInterface::verifyMemDescLayout`。这是一个**插件式的钩子**——"验证规则本身"是留给具体 encoding 定义的，框架层只负责"发现有 encoding 就去问它自己合不合法"。NPU 侧照抄这个钩子框架即可，具体验证逻辑挂在自己的 encoding attr 上。
- **`verifyEquivalentMemDescType`**（`Traits.cpp:13-41`，被 `scf.for`/`scf.if` 之类控制流 op 用来检查 `iter_args` 前后类型一致）：逐字段比较 `shape`/`allocShape`/`elementType`/`memorySpace`/`mutableMemory`，`encoding` 相等就直接过，不相等则调用 `DialectInferLayoutInterface::verifyLayoutsAreEqual` 做语义等价判断（比如两个 encoding 参数不同但描述的其实是同一种物理排布）。同样是"框架给钩子，具体规则留给 encoding 自己"的模式。

**结论：verifier 这一层的设计模式（"通用字段直接比较 + encoding 相关的判断转发给 encoding 自己实现的接口"）本身是可以整体照抄的架构，不需要重新发明。**

## 5. Encoding：真正需要你自己设计的地方

`MemDescType` 的 `encoding` 字段必须实现两个 attr interface（`TritonGPUAttrInterfaces.td:10-39`）：

```tablegen
def LayoutEncodingTrait : AttrInterface<"LayoutEncodingTrait"> {
  let methods = [
    InterfaceMethod<"...", "CGAEncodingAttr", "getCGALayout">,
    InterfaceMethod<"...", "unsigned", "getRank", (ins), [{}], [{
      return $_attr.getCGALayout().getRank();
    }]>
  ];
}
def SharedEncodingTrait : AttrInterface<"SharedEncodingTrait"> {
  let methods = [
    InterfaceMethod<"Return the default alignment for the layout.",
                    "int32_t", "getAlignment", (ins), [{}], [{ return 16; }]>,
  ];
}
```

**惊喜之处**：`SharedEncodingTrait` 这个接口本身**只有一个方法**——`getAlignment()`，默认实现直接返回 16。也就是说，"一个 encoding 要成为合法的 shared-memory encoding"这件事，框架层面的要求极低。真正的复杂度全部在**具体子类怎么实现**：

| Encoding 实现 | 文件位置 | 是否移植 | 原因 |
|---|---|---|---|
| `SwizzledSharedEncodingAttr` | `TritonGPUAttrDefs.td:10-206` | **不建议直接抄**，但要理解思路 | 用 `vec`/`perPhase`/`maxPhase` 参数描述"异或换址"来避免 GPU shared memory 的 32-bank 冲突。这是**GPU bank 冲突模型专属**的数学——NPU SRAM 有没有类似的 bank 冲突、冲突规则是什么，需要重新推导公式，不能照抄参数含义 |
| `PaddedSharedEncodingAttr` | `TritonGPUAttrDefs.td:208-230` | **思路值得借鉴** | 用"每隔 N 个元素插入 M 个 padding"来错开地址、避免冲突，比 swizzle 更简单直接。如果 NPU SRAM 的冲突模式简单（比如就是简单的 bank 数取模），这种 padding 方案比异或 swizzle 更容易验证正确性，是更合适的起点 |
| `NVMMASharedEncodingAttr` | `TritonGPUAttrDefs.td:428+` | 不需要 | Hopper/Blackwell `wgmma`/`tcgen05` 指令要求的特定内存排布，硬件指令绑定 |
| `AMDRotatingSharedEncodingAttr` | `TritonGPUAttrDefs.td:498+` | 不需要 | AMD LDS 特有的另一种防冲突方案 |
| `PartitionedSharedEncodingAttr` | `TritonGPUAttrDefs.td:330+` | 不需要（上一篇已排除） | warp-specialize 场景下把逻辑 buffer 拆到不同物理分区 |

**移植建议**：v0 阶段最省事的做法是**先做一个"trivial" encoding**——不做任何 swizzle/padding，就是行主序（或者按 NPU 实际的 DMA/访存约束定一个简单的固定 stride 布局），`getAlignment()` 返回 NPU DMA 要求的对齐字节数。等后面发现真的有 bank 冲突或访存效率问题，再对着 `PaddedSharedEncodingAttr` 的思路（比 swizzle 简单很多）加 padding 规则，而不是一上来就抄 swizzle 的异或数学。

## 6. 真正 GPU-specific 的部分：`local_load`/`local_store` 的 LLVM lowering

这是本文最想强调的一点——**op 的 td 定义和 verify 都不难，难的、也是唯一真正硬编码了 GPU 执行模型的地方，在 `lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp`。**

以 `LocalLoadOpConversion::matchAndRewrite`（`MemoryOpToLLVM.cpp:168-209`）为例，核心是这几行：

```cpp
auto regLayout = toLinearLayout(regTy).removeZeroBasesAlongDim(str_attr("register"));
auto sharedLayout = toLinearLayoutIgnoringPadding(memDescTy);
auto cvt = invertAndComposeBlockLocal(sharedLayout, regLayout);
auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                              smemObj, rewriter, targetInfo,
                              makeSharedLoadEmitter(targetInfo, op));
```

这几行在做的事情，翻译成人话：

1. `regTy`（`local_load` 结果的 tensor 类型）带着一个"**这个 tensor 的每个元素归哪个 warp 的哪个 lane 的哪个寄存器持有**"的分布式布局（`DistributedEncodingTrait`，见 `TritonGPUAttrInterfaces.td:41-69` 里"CTA→Warp→Thread→寄存器"四级层级的描述）。这是 SIMT 执行模型独有的概念——**成千上万个线程各自独立执行同一条指令，每个线程只处理自己那一份寄存器数据**，所以"哪个元素归哪个线程"是必须显式建模的东西。
2. `memDescTy` 的 shared encoding 描述"这个元素物理上存在 shared memory 的哪个字节偏移"（含 swizzle）。
3. `invertAndComposeBlockLocal` 把这两个布局复合、求逆，算出**"这个正在执行的线程，要读的这几个寄存器元素，对应的 shared memory 地址是多少"**——这是一个线性代数问题（`LinearLayout` 是 Triton 内部专门为这类 GF(2) 线性布局设计的工具，`triton/Tools/LayoutUtils.h`）。
4. `lowerLocalLdSt` 最终为**每个线程**生成实际的 `ld.shared`/`st.shared` LLVM 指令。

**为什么这一整套不能照抄**：它的存在前提是"一个 tensor 的数据分散在成百上千个并发执行、各自持有私有寄存器的硬件线程上，需要用矩阵求逆算出每个线程该读哪"。如果 NPU 的执行模型不是这种细粒度 SIMT（比如是一个/少数几个执行引擎，通过显式的向量指令 + 地址生成器访问 SRAM，不存在"成千上万线程各自算一份地址"的情况），那么这一整套 `LinearLayout` 求逆机制**大概率是不需要的**——NPU 的 lowering 应该直接是"根据 `shape`/`strides`（由 encoding 给出）和当前要访问的下标，用一个地址生成公式算出 SRAM 偏移"，是标量地址计算，而不是"给定线程 ID，反解出它该读哪个字节"这种为大规模并行设计的间接寻址问题。

**移植建议**：
- **保留**：`local_load`/`local_store` op 本身的语义（内存域↔计算域搬运）、`MemDescType`、encoding 的"必须能告诉你元素在缓冲区里的地址"这个职责本身。
- **重新设计**：地址计算的**实现方式**。不需要 `LinearLayout`/寄存器分布式布局这套"从并发线程视角反推地址"的机制，换成 NPU 自己执行模型下"直接根据 shape+encoding 计算线性/物理地址"的公式即可，会简单得多。
- 如果 NPU 确实存在类似"多个并行执行单元各自独立访存"的情况（比如多个 DMA 通道、多条 vector lane），那可以在需要时再引入一个简化版的"分布式布局"，但不必一开始就照搬 GPU 三级/四级 warp-thread 层级的复杂度。

## 7. `SharedMemoryObject`：地址表示的可移植部分

Lowering 时，一个 memdesc 值在 LLVM 层被翻译成一个 `SharedMemoryObject`（`include/triton/Conversion/TritonGPUToLLVM/Utility.h:371-451`），核心字段是：

```cpp
class SharedMemoryObject {
  SmallVector<Value> bases;   // shared memory 基地址指针（分区场景下多个）
  Type baseElemType;
  SmallVector<Value> offsets; // i32，相对基地址的偏移，view 类 op 会累加这个
};
```

- **`bases` + `offsets` 的组合**（`Utility.h:444-450`）：这是"一个内存描述符 = 基地址 + 偏移"的最基础表示，和 GPU 无关，**照抄**。`memdesc_index`/`memdesc_subslice` 这些 view op 在 lowering 时就是往 `offsets` 里累加常量/变量偏移，不搬实际数据——这个"view op 只改变偏移记录、不触碰内存"的实现思路也直接可用。
- **`getShmemOffset`/`getMaskSpanOffsets`/`getCSwizzleOffset`** 这些方法（`Utility.h:415-442`）：内部要把 offset 换算成"考虑了 swizzle/padding 之后的实际字节地址"，这部分逻辑依赖具体 encoding 的换址公式，随第 5 节的 encoding 重新设计而重新实现,但对外的接口形态（"给一个逻辑 offset，问 SRAM 里的地址"）可以照抄。
- `bases` 是 `SmallVector` 而非单个 `Value`，是为了给 `PartitionedSharedEncoding`（NPU 不需要）场景多个物理分区各有各的基地址；NPU v0 直接假设永远只有一个 base 即可。

## 8. 明确排除清单（v0 不做）

| 内容 | 位置 | 为什么先不做 |
|---|---|---|
| `local_gather` / `local_scatter` / `local_atomic_scatter_rmw` | `TritonGPUOps.td:400-508` | 按索引张量做随机读写/原子读改写，是"在 `local_load`/`local_store` 之上"的高级能力；先把最基础的规整访存跑通，索引类访存等有明确需求（比如某个 NPU 算子确实要用）再加，其 lowering（`MemoryOpToLLVM.cpp:20-49, 267-360`）同样构建在第 6 节那套线程级地址反解机制之上，复杂度更高 |
| `async_copy_global_to_local` / `async_commit_group` / `async_wait` | `TritonGPUOps.td:48-151` | cp.async 硬件指令专属的"发起异步拷贝、之后再等待完成"两阶段模型；`local_load`/`local_store` 是同步语义，足够作为 v0 起点。等 NPU 明确了自己的 DMA/异步搬运原语长什么样，再单独设计对应的 op，不要硬套 GPU 的 async token 模型 |
| `memdesc_trans` / `memdesc_reshape` / `memdesc_reinterpret` | `TritonGPUOps.td:275-357` | 都是"view 不搬数据"的变体，`memdesc_index`/`memdesc_subslice` 覆盖了最常见的"多缓冲取一份"/"切片"需求，其余三种按后续具体算子需要再加 |
| `NVMMASharedEncodingAttr` / `AMDRotatingSharedEncodingAttr` / `PartitionedSharedEncodingAttr` | `TritonGPUAttrDefs.td` | 硬件指令绑定或 warp-specialize 专属，第 5 节已说明 |
| `TensorMemoryEncodingAttr` / `TensorMemoryScalesEncodingAttr` 这条 `MemDescType::verify` 分支 | `Types.cpp:110-130` | NVIDIA Blackwell Tensor Memory 专属的另一种内存空间，和 shared memory 是平行概念，NPU 不需要 |
| `LinearLayout` 求逆机制（`invertAndComposeBlockLocal`/`toLinearLayout`） | `MemoryOpToLLVM.cpp` 全文, `Tools/LayoutUtils.h` | 第 6 节详述，SIMT 专属，NPU 应换成直接的地址计算公式 |

## 9. 建议的实现顺序

承接上一篇的顺序（那篇是第 1-2 步的细化），这里把"load/store + memdesc"单独铺开：

1. 定义 `!npu.mydesc<shape x elemType, #encoding, #space>` 类型，字段照抄 `MemDescType` 的六个（`allocShape` 一起搬，为将来的多缓冲留口子），先只支持一种 trivial encoding（行主序 + NPU 要求的对齐，`getAlignment()` 硬编码返回该值）。
2. 把 `MemDescType::verify` 里通用的检查（rank≥1、无 0 维、元素位宽合法、`allocShape.size() >= shape.size()`）照抄过来；encoding 相关分支砍到只剩自己那一种。
3. 定义 `npu.local_load` / `npu.local_store`，参数、`GenericSharedRead/Write` 式的 side-effect 标注、`verifyMemoryOpTypes`/`verifySharedMemoryRank` 两个 helper 照抄。
4. 定义 `npu.memdesc_index`（多缓冲取一份）和 `npu.memdesc_subslice`（切片），verifier 逻辑（"encoding 相同"、"allocShape 匹配"、"非拆分维度 offset 必须为 0"）照抄 `MemDescIndexOp`/`MemDescSubsliceOp::verify` 的检查项，具体数值公式按自己的 shape 规则改。
5. **重点**：`npu.local_load`/`npu.local_store` 的 lowering，不抄 `LinearLayout` 那一套，直接写"给定 encoding + 逻辑下标 → SRAM 物理地址"的标量公式（对应 `SharedMemoryObject::getShmemOffset` 的角色），因为大概率 NPU 没有"每个线程各自算地址"的需求。
6. `SharedMemoryObject`（base + offset 的表示）可以照抄结构，作为 lowering 层内部的辅助类型。
7. 等到有明确的算子需求（比如某种归约/重排要用到索引访存），再引入 `local_gather`/`local_scatter` 或 NPU 自己的异步搬运原语，不要一开始就把 GPU 全部花活搬过来。

和上一篇一样，这样拆下来，每一步都对应 Triton 源码里边界清晰的一块，方便后续对照排查。
