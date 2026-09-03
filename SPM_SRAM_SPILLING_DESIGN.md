# 基于 Tensor 的 SPM 分层存储、流水化与静态 SRAM 分配设计

本文重新定义 L1 / L2 / DRAM 的 IR 表示、两级 tiling、software pipeline / double buffer、
spilling、One-Shot Bufferize 和静态地址分配之间的边界。

本版**整体取代**此前基于 `!spm.l1mem`、`!spm.l2mem`、`!spm.dram` 以及
`spm.l1_load/store`、`spm.l2_load/store` 的方案。旧版中与这些 type/op 绑定的 canonical form、
alias 规则和分层 lowering 不再是实现依据；`NPU_SRAM_ALLOCATION_PORTING.md` 中与表示无关的
打包算法仍可作为参考。

核心原则只有一句话：

> **tensor 表达值，allocation 表达存储层级，copy 表达内容迁移，最终 arena view 表达物理地址。**

---

## 0. 本版的确定结论

| 问题 | 结论 |
|---|---|
| Tensor 层是否需要三套 memory type | **不需要。** Bufferize 前统一使用 builtin `tensor<...>` |
| L1 / L2 / DRAM 放在哪里表达 | 放在显式 allocation 的 `memory_space` 属性上，不放进 tensor type |
| 是否需要三套 alloc op | **不需要。** 统一使用 `bufferization.alloc_tensor` |
| 是否需要四套 load/store op | **不需要。** 统一使用 `bufferization.materialize_in_destination` |
| spill/fill 是否需要专用 op | **不需要。** 它们仍然是显式 destination copy |
| view/subview 是否需要 SPM op | **不需要。** Tensor 层用 `tensor.extract_slice` 等标准 op，Bufferize 后用 `memref.subview` |
| software pipeline 在何时运行 | 两轮 tiling 和显式 tensor allocation 之后，One-Shot Bufferize 和物理分配之前 |
| 双缓冲如何表达 | 循环外一个固定大小的 slot/ring allocation，加静态 disjoint slice 和正确的 tensor SSA 版本传递 |
| offset 放在哪里 | 不放在 tensor alloc 上；Bufferize 后由每个 memory space 的 arena planner 产生 `memref.view`/offset |
| One-Shot Bufferize 是否需要扩展自定义 tensor type/op | 正常路径**不需要**；复用标准 Bufferization/Tensor/Linalg/SCF 接口 |

允许在 Bufferize 后出现：

```mlir
memref<64x64xf16, #spm.memory_space<l1>>
```

这不是重新发明 `!spm.l1mem`。它是 builtin `memref` 的标准 memory-space 参数，是后端区分地址空间所必需的。
真正被删除的是三套自定义 shaped type 以及围绕它们复制出来的整套接口和 op 家族。

建议 SPM dialect 最小只提供一个可读的 memory-space attribute：

```text
#spm.memory_space<dram>
#spm.memory_space<l2>
#spm.memory_space<l1>
```

它们到硬件 address-space number 的映射来自统一的 target description，不在各 pass 中重复写死。
若 MVP 想连这个 attribute 都暂时不实现，也可以直接使用 target description 提供的整数 memory-space ID；
两种写法只影响可读性和后端 attribute conversion，不改变本文的数据模型。

---

## 1. 三个容易混淆的“分配”

后续讨论必须区分三件不同的事：

1. **Tensor storage declaration**：
   `bufferization.alloc_tensor` 声明“这个 tensor 内容将拥有一块新的物理存储”，但尚未决定地址。
2. **Buffer allocation**：
   One-Shot Bufferize 把上面的 op 机械转换成带 memory space 的 `memref.alloc`。
3. **Static physical allocation**：
   最后的 SRAM planner 根据 lifetime、容量、对齐和 bank 约束，把多个 `memref.alloc` 打包到 L1/L2 arena，
   并用 `memref.view` 表达最终 offset。

software pipeline 所提升的是第 1 类 allocation。它把“每个动态迭代似乎创建一次的 tile storage”改写成
“循环外固定 K 个物理 slot”。One-Shot Bufferize 只负责把这份已经确定的 storage graph 机械转成 memref；
最后的 planner 才决定不同 allocation 是否可以复用同一段地址。

如果不先做 slotization/hoisting，循环体里的 `alloc_tensor` Bufferize 后就是循环体里的 `memref.alloc`；
静态 planner 无法把一个动态执行次数的 allocation 当成固定数量的 SRAM buffer。这正是本版把
software pipeline 放在 Bufferize 前的根本原因。

---

## 2. Tensor Placement Normal Form

### 2.1 类型不携带层级

在整个 tiling、placement、spilling 和 pipeline 阶段，数据类型始终是普通 tensor：

```mlir
tensor<4096x64xf16>
```

禁止用下列方式编码存储层级：

- `!spm.l1mem<...>` / `!spm.l2mem<...>` / `!spm.dram<...>`；
- `tensor<..., #spm.l1>` 之类的 tensor encoding；
- 为相同 shape/element type 发明三个互不兼容的 tensor-like type。

不用 tensor encoding 的原因不只是语法简洁。Encoding 是 tensor type identity 的一部分，会传播到
`scf.for` iter_args、`tensor.extract_slice`、DPS result 和控制流 join；跨层 copy 会因此变成 type conversion，
迫使大量标准 op 或 BufferizableOpInterface 做额外适配。`alloc_tensor.memory_space` 已经提供了所需信息，
没有理由再把同一事实复制到 tensor type。

### 2.2 层级属于 allocation root

显式 allocation 使用标准 op：

```mlir
%l2_storage = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l2>}> : tensor<4096x64xf16>

%l1_storage = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l1>}> : tensor<4096x32xf16>
```

`bufferization.alloc_tensor` 有两个正好符合本设计的性质：

- 它在 tensor SSA 中开启一个新的 allocation root，不与其他 buffer alias；
- 它已有原生 `memory_space` 属性，One-Shot Bufferize 会把该属性转换成结果 memref 的 memory space。

同一个 `tensor<...>` SSA type 因而可以由 DRAM、L2 或 L1 storage 支撑。值的 placement 由它将来
bufferize 到哪个 allocation root 决定，而不是从 tensor type 上读取。

### 2.3 内容迁移只有一个表示

所有显式 copy 统一使用：

```mlir
%l2_value = bufferization.materialize_in_destination
    %dram_tile in %l2_storage
    : (tensor<4096x64xf16>, tensor<4096x64xf16>)
      -> tensor<4096x64xf16>

%l1_value = bufferization.materialize_in_destination
    %l2_subtile in %l1_storage
    : (tensor<4096x32xf16>, tensor<4096x32xf16>)
      -> tensor<4096x32xf16>
```

它比普通 `tensor.insert_slice` 更适合作为存储边界：它明确要求 source 的内容必须 materialize 到指定
destination 的未来 buffer；如果这个 destination 不能被原地使用，Bufferize 应失败，而不是悄悄换一块地址。

方向完全由 source/destination allocation root 的 memory space 推导：

| Source | Destination | 含义 |
|---|---|---|
| DRAM | L2 | input load / prefetch |
| L2 | L1 | tile load / fill |
| L1 | L2 | writeback / spill |
| L2 | DRAM | output store / spill |
| 同一层 | 同一层 | 普通 copy |

硬件不支持 DRAM 与 L1 直连，因此 provenance verifier 必须拒绝这类边。合法化只能展开成
DRAM ↔ L2 ↔ L1 两段，每段都有自己的显式 destination allocation。

Tensor destination 形式会返回更新后的 tensor SSA 值。后续用户必须消费这个 result，而不是继续消费
写入前的 destination：

```mlir
// 正确。
%loaded = bufferization.materialize_in_destination %src in %storage
    : (tensor<32xf16>, tensor<32xf16>) -> tensor<32xf16>
%r = linalg.generic ... ins(%loaded : tensor<32xf16>) ...
```

旧版“copy 无结果、靠 mutable memdesc 原地改内容”的假设在 tensor IR 中不成立，本版彻底删除。

### 2.4 `alloc_tensor copy(...)` 不进入稳定形态

虽然标准 op 允许下面的紧凑写法：

```mlir
%x = bufferization.alloc_tensor() copy(%src)
    <{memory_space = #spm.memory_space<l1>}> : tensor<32xf16>
```

本设计的稳定形态禁止它。allocation 和 copy 必须拆开，因为 software pipeline 要把 allocation 提升到循环外，
但 copy 通常依赖循环 induction variable，必须留在循环的某个 pipeline stage 中。

统一规范化为：

```mlir
%storage = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l1>}> : tensor<32xf16>
%x = bufferization.materialize_in_destination %src in %storage
    : (tensor<32xf16>, tensor<32xf16>) -> tensor<32xf16>
```

### 2.5 不再设计 tensor-level dealloc

不新增 `spm.l1_dealloc` / `spm.l2_dealloc` / `spm.dram_dealloc`，也不要求 tiling/pipeline 手写
`bufferization.dealloc_tensor`。Tensor 阶段的 storage planner 根据真实 use、控制流和 pipeline schedule
计算 reservation；Bufferize 后再由标准 deallocation pipeline 产生实际 `memref.dealloc`，并由
`optimize-allocation-liveness` 把它移动到最后一次真实使用之后。静态 arena planner 最终会消费对应的
alloc/dealloc。

因此不存在“dealloc 是注释还是 use”的双重语义，也不需要修改 `mlir::Liveness` 去忽略某个自定义 op。

### 2.6 函数边界固定为 DRAM ABI

v0 推荐在 kernel 边界直接使用带 DRAM memory space 的 memref，再用标准 bridge 进入 tensor 世界：

```mlir
func.func @kernel(
    %input_mem: memref<4096xf16, #spm.memory_space<dram>>,
    %output_mem: memref<4096xf16, #spm.memory_space<dram>>) {
  %input = bufferization.to_tensor %input_mem restrict
      : memref<4096xf16, #spm.memory_space<dram>> to tensor<4096xf16>
  %output = bufferization.to_tensor %output_mem restrict writable
      : memref<4096xf16, #spm.memory_space<dram>> to tensor<4096xf16>
  // Tensor computation...
  return
}
```

这样 One-Shot 无需从无标注的 tensor function argument 猜 memory space，也不需要把 memory space 塞进
tensor encoding。若前端必须保留 tensor signature，则应先用单独的 ABI conversion pass 将所有 kernel
arguments/results 明确映射到 DRAM；不能依赖普通内部 tensor 的默认值。

---

## 3. Placement provenance：类型系统不再检查的事由分析检查

统一 tensor type 后，`tensor<...>` 本身不再回答“数据在哪一层”。需要一个很薄的 placement provenance
分析。它追踪的是未来 buffer 的 allocation root，而不是重新实现 One-Shot 的 RaW 分析。

基本规则：

1. `bufferization.alloc_tensor` 创建新 root，space 取其 `memory_space`。
2. `bufferization.to_tensor` 的 root/space 来自底层 memref。
3. `materialize_in_destination` 的 result root 等于 destination root；source 与 result **内容相同但不 alias**。
4. `tensor.extract_slice`、`tensor.cast`、reshape/view 类 op 传播 root，但记录 subset。
5. Linalg 等 DPS result 的 root 等于对应 `outs` operand 的 root。
6. `scf.for` / `scf.if` / block argument 的所有 incoming value 必须能合并到兼容的 root/space；
   不同 memory space 不能直接 join，必须先 materialize 到共同 destination。
7. 无法追溯到显式 root 的内部 tensor 在最终 normal form 中非法。

这个分析可以直接调用 `BufferizableOpInterface` 的 alias/equivalence 查询；不需要为 SPM 再维护一套
“seed/view/select/其他断言”的 dialect-specific alias 格。

placement verifier 至少检查：

- hardware compute 的所有 buffer operand 和 `outs` 最终都由 L1 root 支撑；
- 每条跨空间边都由 `materialize_in_destination` 显式表示；
- DRAM ↔ L1 不直连；
- destination shape、element type 和有效范围匹配；
- 不在控制流 join 中合并不同 space 的 root；
- L1/L2 allocation 的 shape 静态可知，或已转换成带静态上界的固定 slot；
- 不残留会在 One-Shot 时隐式产生 SRAM allocation 的 `tensor.empty` 或非 DPS result。

---

## 4. 两轮 tiling 如何提前区分 L2 和 L1

### 4.1 Tiling 决定颗粒度，不改变类型

L2 tiling 和 L1 tiling 各自负责发现本层 materialization boundary：

```text
DRAM whole tensor
  └─ L2 outer tile
       └─ L1 inner tile
            └─ hardware compute
```

- **L2 tiling** 决定 DRAM ↔ L2 的 tile shape、outer loop 和可能的 L2 residency。
- **L1 tiling** 在 L2 tile 内继续分块，决定 L2 ↔ L1 的 subtile shape、inner loop、accumulator 和 scratch。

层级信息不需要附着到 tile type。一个 allocation 的 `memory_space` 表示层级，它的 result shape 就表示
该层的颗粒度：

```mlir
// L2 颗粒度：4096x64。
%weight_l2_storage = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l2>}> : tensor<4096x64xf16>

// L1 颗粒度：4096x32。
%weight_l1_storage = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l1>}> : tensor<4096x32xf16>
```

### 4.2 标注只表达 intent，并且必须被消费

推荐 tiling transform 返回“哪些新 tile value 要 promote 到哪一层”的 handle/list，随后立即调用统一的
`MaterializeTensorStorage`。如果 pass pipeline 无法直接传递 handle，可以临时给 tile producer 加一个
`spm.materialize` dictionary attribute，至少包含：

```text
space = l2 | l1
role  = resident | stream | accumulator | scratch | writeback
pipeline_candidate = true | false
```

这只是 tiling 与 materialization 之间的短命协议，不是第二套 placement 真相。
`MaterializeTensorStorage` 必须消费并删除标注，插入：

1. 一个带相应 `memory_space` 的 `bufferization.alloc_tensor`；
2. 当原内容会被读取时，一个 `bufferization.materialize_in_destination`；
3. 对纯 output/accumulator destination 只插 allocation，不做无意义的初始 copy。

仓库已有 `transform.structured.promote_tensor to <memory-space>`，其实现正是
`alloc_tensor(memory_space)` 加必要的 `materialize_in_destination`，可以直接复用或抽取为公共 utility。

### 4.3 谁常驻、谁流式进入，不再由 canonical form 一刀切

本版删除“分配前所有数据一律完整落在 L1”的 canonical form。它无法表达完整 weight 留在 DRAM、
L2 只接收 outer tile、L1 再接收 subtile 的正常数据流。

placement 的初始选择由 tiling/use pattern 给出：

- 小而循环不变、复用高的 input 可以在 L2，甚至 L1，建立 loop-external resident allocation；
- 大 weight 不完整进入 L2，只在 outer loop 中产生 L2 tile；
- L1 只物化当前 inner tile 和 compute 必需的 accumulator/scratch；
- output 由 L1 写到 L2 staging，再写到 DRAM destination slice。

之后的容量 planner 可以撤销某个“preferred resident”选择并做 spill/demotion，但不会改变 tile 边界的
基本语义。

### 4.4 固定形状与尾块

静态 SRAM 要求最终 slot 数和每个 slot 的最大字节数在编译期已知。动态尾块采用：

- 最大静态 tile allocation；
- `tensor.extract_slice` 的有效子范围；
- padding、mask 或 predication 保证越界部分不参与计算。

不要为最后一次迭代生成一个动态大小的 L1/L2 allocation。若 shape 只有编译期上界，slot 按上界计费。
现有 `transform.tensor.make_loop_independent` 的做法可以直接参考：先用 value-bounds 推导循环无关的最大
shape，在循环外保留最大 storage，再在循环内提取实际大小的 slice。

---

## 5. Software pipeline / double buffer 是 storage graph 的一部分

### 5.1 为什么必须在 Bufferize 前完成

Tiling 后最自然的 IR 会在循环体里出现逻辑 allocation：

```mlir
scf.for %i = ... {
  %tile_storage = bufferization.alloc_tensor()
      <{memory_space = #spm.memory_space<l1>}> : tensor<4096x32xf16>
  %tile = bufferization.materialize_in_destination %src_tile in %tile_storage
      : (tensor<4096x32xf16>, tensor<4096x32xf16>)
        -> tensor<4096x32xf16>
  // compute(%tile)
}
```

这里的 alloc 不能直接交给 Bufferize。software pipeline 必须把它转换成固定数量的 slot，并将 slot
allocation 提到所有会重复执行它的循环之外。只有这样，Bufferize 后才得到固定数量的 `memref.alloc`，
静态 planner 才能对它们分配地址并跨迭代复用。

### 5.2 Pipeline 分成 plan 和 materialize 两步

pipeline depth 会乘大 SRAM 占用，而容量又可能迫使 depth 从 2 降到 1。为避免先后顺序形成循环依赖，
software pipeline 明确拆成两步：

1. **PlanPipeline**：只计算 stage、initiation interval、每个逻辑 allocation 的 `slotCount` 和 transfer
   reservation，不改写循环结构。
2. **MaterializePipeline**：在 storage/spill 计划冻结后，生成 prologue、steady state、epilogue，
   建立固定 slot，提升 allocation，并正确传递 tensor SSA 版本。

中间的 storage planner 按 `slotCount × alignedTileBytes` 计费。它插入的新 fill/writeback 也会回到
PlanPipeline 重新排 stage，直到 placement、slot count 和 pipeline depth 一起收敛。决策在一次编译中只允许：

- residency 从快层向慢层移动；
- pipeline depth 不增只减；
- tile size 由外层重试机制不增只减。

这个单调策略避免 placement 和 pipeline 之间来回振荡。

### 5.3 v0 的双缓冲表示：一个 root，K 个静态 disjoint slice

推荐把 K 个 slot 打包成一个带前导 slot 维度的 allocation：

```mlir
%ring = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l1>}> : tensor<2x4096x32xf16>

%ping = tensor.extract_slice %ring[0, 0, 0] [1, 4096, 32] [1, 1, 1]
    : tensor<2x4096x32xf16> to tensor<4096x32xf16>
%pong = tensor.extract_slice %ring[1, 0, 0] [1, 4096, 32] [1, 1, 1]
    : tensor<2x4096x32xf16> to tensor<4096x32xf16>
```

`%ring` 位于相关循环之前，`%ping` 和 `%pong` 是静态可证不相交的两个 subset。这样静态 planner 只看到
一个大小明确的 allocation root，One-Shot 则把两个 slice 降成同一个 memref 的两个 disjoint subview。

v0 对 K 很小的流水做 K-way partial unroll，使每个 residue 静态绑定一个 slot。不要在循环中用
`arith.select` 从两个独立 allocation root 中动态挑 ping/pong；那会把两个 root 合并成 may-alias 集合。
也不要只用 `%iv mod K` 动态切 ring 后依赖编译器证明跨迭代 disjoint；这个证明和 tensor SSA 版本传递
都比 K-way unroll 更复杂。

### 5.4 提升 allocation 不等于回到可变 tensor

allocation 提升后，每次 copy/compute 仍会产生新的 tensor SSA 内容版本。pipeline 必须通过
`scf.for iter_args` 或展开后的直连 SSA edge 传递这些版本：

```mlir
%r:2 = scf.for %i = ... step %c2
    iter_args(%ping_state = %ping, %pong_state = %pong)
    -> (tensor<4096x32xf16>, tensor<4096x32xf16>) {
  %ping_next = bufferization.materialize_in_destination
      %src0 in %ping_state : (...) -> tensor<4096x32xf16>
  // consume %ping_next; schedule the other stage on %pong_state
  %pong_next = bufferization.materialize_in_destination
      %src1 in %pong_state : (...) -> tensor<4096x32xf16>
  // consume %pong_next
  scf.yield %ping_next, %pong_next
      : tensor<4096x32xf16>, tensor<4096x32xf16>
}
```

不能每轮都从循环外最初的 `%ping` 重新开始，并假设上轮的写“自然改变了它”。那是 memref 思维，不是
tensor value semantics；One-Shot 会因此发现 RaW 冲突或插入额外 allocation。

ping/pong 在 `iter_args` 中的位置也必须稳定：ping version 始终 yield 回 ping 对应的位置，pong 同理。
不要通过交换两个 iter_arg 的位置实现轮转；K-way unroll 已经提供了静态 stage-to-slot 映射。

### 5.5 提升位置

slot allocation 应放在**最接近但位于所有重复执行 scope 之外**的位置，通常紧邻 owning top-level loop
之前，而不是无条件移动到函数入口开头：

- inner-loop tile 若跨 outer-loop 迭代复用，必须提升到 outer loop 外；
- 两个顺序执行的 top-level loop 各自在自己前面 allocation、结束后释放，才能复用同一 arena offset；
- v0 static planner 只接受 function entry block 中的 alloc 时，pipeline 必须将 allocation 放进该 block，
  但仍应尽量靠近 owning loop。

### 5.6 Pipeline 后的硬性后置条件

进入最终 One-Shot Bufferize 前必须满足：

1. 所有 L1/L2 `alloc_tensor` 都不在任何 repetitive region 内；
2. 每个逻辑迭代 allocation 已变成固定 K-slot storage；
3. K 与重叠 stage 的最大并发实例数一致；
4. 同一时刻可能被 DMA/compute 使用的 slot 必须物理不相交；
5. copy 后的 tensor result 通过 SSA/iter_args 传到消费者；
6. prologue/epilogue 覆盖少于 K 次迭代和非整除尾部；
7. 对 final scheduled IR 重做一次 capacity verification。

---

## 6. 完整 pass pipeline

推荐的主流程如下：

```text
Canonical DPS tensor IR
  ↓
L2 tiling + L2 materialization intent
  ↓
MaterializeTensorStorage(L2)
  ↓
L1 tiling + L1 materialization intent
  ↓
MaterializeTensorStorage(L1)
  ↓
ResolveTensorConflicts
  - One-Shot analysis + insertTensorCopies
  - 规范化 alloc_tensor copy(...) 为 alloc + materialize
  ↓
PlanPipeline ↔ PlanStorageAndSpills（单调迭代）
  ↓
MaterializePipeline（slotize + hoist + prologue/kernel/epilogue）
  ↓
VerifyTensorPlacementAndCapacity
  ↓
One-Shot Bufferize
  ↓
buffer-deallocation-pipeline + optimize-allocation-liveness
  ↓
StaticMemoryPlan：按 memory space 独立 arena/offset
  ↓
跨空间 memref.copy → target DMA；其余后端 lowering
```

这里 `ResolveTensorConflicts` 的目的，是在仍能被 pipeline 看见的时候把 One-Shot 所需的 RaW copy 显式化。
它产生的 allocation 同样必须有确定的 memory space，并参加 slotization 和容量规划。

pipeline 改写之后再做一次 One-Shot analysis-only 检查。如果分析表明仍需新增 L1/L2 allocation，说明
SSA version threading、placement 或 pipeline transformation 有错误；应在这里硬失败，不允许最终
One-Shot 静默生成循环内 `memref.alloc`。

---

## 7. Tensor 层的容量模型与 spilling

### 7.1 分析单位是 allocation root，不是 tensor type

对每个 `alloc_tensor` root 建立 allocation record：

```text
root
space
shape / bytes / alignment / bank constraints
alias/subset closure
reservation intervals
slotCount
role: required | resident | stream | scratch | spill-home | transit
```

DPS result、slice、cast、loop-carried value 都通过 provenance/One-Shot equivalence 归到相应 root。
view 默认按整个底层 root 计费；只有静态证明 disjoint 的 subset（例如 ping/pong slice）才能分别建
reservation。没有任何分析再按 `!spm.l1mem` 或 `!spm.l2mem` type 扫描。

### 7.2 Lifetime 是内容 reservation，不只是 alloc SSA 区间

alloc 被提升到循环外后，其 SSA handle 可能覆盖整个循环，但某个 slot 的内容只在 DMA start 到最后一次
消费/对应 wait 之间必须保留。容量 planner 应使用 pipeline schedule 上的 reservation interval：

- source 和 destination 都保持到异步 copy 完成；
- compute input 保持到最后一次 read；
- output 从第一次 write 保持到 writeback 完成；
- K 个重叠迭代映射到 modulo schedule 中的 K 个独立 slot。

v0 的物理 arena planner 可以保守地把整个 ring root 视为 owning loop 全程活跃。这仍然能实现固定 SRAM
和跨顺序 loop 的地址复用，只是不能让 ring 内暂时空闲的地址与同一 loop 的其他 allocation 交错复用。
更精确的 modulo reservation packing 是后续质量优化，不能混同为正确性的前置条件。

### 7.3 可行性与打包质量仍然分开

对每个 scheduled program point `p`：

```text
maxLive(p) = p 处所有 reservation 的对齐后字节数之和
packedHighWater = 当前 offset 打包方案使用到的最高地址
```

决策顺序：

```text
if maxLive > capacity:
    必须改变 residency、pipeline depth 或 tile size
else if packedHighWater > capacity:
    先重新打包；仍失败才是碎片问题
else:
    当前层可行
```

不能因为一个贪心 offset 布局有碎片就立刻增加 DMA；也不能只看 `maxLive` 而不做最终真实打包校验。

### 7.4 先区分 mandatory working set 和可驱逐 residency

某个 compute/pipeline stage 中同时需要的：

```text
operands + outs + scratch + 必须 in-flight 的 transfer slots
```

是 mandatory working set。它本身超过容量时，spilling 不能修复，因为候选在同一 stage 内正在被使用。
处理顺序是：

1. 降低 pipeline depth，例如 2 → 1；
2. 回退 L1/L2 tile size；
3. 仍无法满足则报告硬错误。

只有“跨过超容量窗口但窗口内不被使用”的 resident segment 才是合法 spill victim。

### 7.5 Spill/fill 仍然只生成标准 tensor op

dirty L1 内容写回 L2：

```mlir
%home_storage = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l2>}> : tensor<64x64xf16>
%home = bufferization.materialize_in_destination %value in %home_storage
    : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
```

后续重新装入 L1：

```mlir
%reload_storage = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l1>}> : tensor<64x64xf16>
%reload = bufferization.materialize_in_destination %home in %reload_storage
    : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
```

这些新 allocation/copy 和正常 tiling promotion 完全同构，必须重新参加 PlanPipeline、slotization 和
alloc hoisting；不允许 spill pass 在最终 pipeline 之后留下循环内 allocation。

### 7.6 用 content version 取代全局 dirty 位

旧版把 `dirty` 当作一个 buffer 上线性扫描得出的 bool，这在分支、循环和多次 DPS update 下不成立。
Tensor SSA 更适合直接追踪内容版本：

- function input、constant 或 compute result 创建一个 content version；
- view/slice 表示该 version 的 subset；
- `materialize_in_destination` 使 source version 在另一个 allocation root 中拥有一份有效副本；
- DPS write 产生新 version，旧 home 对新内容不再有效；
- 控制流 join 只有在所有 incoming path 都有相同 content version 时才能保留 home 事实。

对当前 version 维护 `validHomes` 集合，而不是单个“最深 home”：

```text
validHomes(v) = 当前保存 v 内容的所有慢层 SSA value/root
```

策略通常优先选择最近的有效 home，例如已经常驻 L2 的 input 从 L2 reload；当 L2 压力更重要时，也可放弃
长寿命 L2 home，选择 DRAM 加短命 L2 transit。这个选择属于 cost model，不能写死成“永远最深”或
“永远最近”。

由此得到：

- 有有效慢层副本的 clean eviction 不需要 store，但若之后还用，仍要支付 reload；
- 无有效 home 的新计算结果必须先 writeback；
- refill 后只读、再次 eviction 时可沿用原 home，消除重复 store；
- L1 的有效 home 只有 DRAM 时，reload 必须合法化为 DRAM → 临时 L2 slot → L1。

### 7.7 Victim 选择是带执行次数的覆盖问题

对最早的超容量窗口：

```text
deficit = max_p(maxLive(p) - capacity)
```

选择一组不在窗口内使用的 resident segments，使释放字节数覆盖 deficit，并最小化估算代价：

```text
dynamic store bytes
+ dynamic reload bytes
+ hop latency / bandwidth cost
- 可被 compute 隐藏的 overlap credit
+ 对下一层造成的 residency pressure
```

这里 size 不一定约掉，因为不同候选的：

- 路径可能是一跳或两跳；
- loop trip count / 执行频率不同；
- slotCount 不同；
- 有的已有 home，有的需要 writeback；
- refill 可能被流水隐藏，也可能位于 recurrence critical path。

实现上可先按 `cost / freedBytes` 贪心，再用小规模 knapsack 修正“选一块过大的 victim”问题。
next-use distance 只作为相近代价下的 tie-break。

### 7.8 控制流和循环的改写规则

spill 操作的是一个 content version 的 residency segment，不是简单地“从某个 op 之后替换所有 use”。

- 对分支后的多个 use cluster，在各自最近合法支配点插 fill；允许一个 home 对应多个 fill。
- 对静态 disjoint slice 可以独立 spill；否则 spill 整个 allocation root。
- loop-invariant resident value 可降到 L2，并在每次需要的 stage 中建立 L1 工作 slot。
- loop-local streamed tile 已由 pipeline slotization 管理；若 mandatory slots 超容量，应减 depth/retile，
  而不是把正在使用的 slot 再 spill。
- loop-carried recurrence 若加入 store/fill 会破坏 initiation interval，应标记为 non-spillable，或由专门的
  recurrence transformation 重写 iter_args；不能用普通 dominance-based split 硬改。

### 7.9 分层顺序与收敛

逻辑上复用同一个 `PlanTier`：

```text
PlanTier(L1, next = L2)
PlanTier(L2, next = DRAM)
```

L1 spilling 新增的 L2 home/transit 是普通 `alloc_tensor(memory_space=l2)`，自然进入 L2 的容量问题；
L2 spilling 同理生成 DRAM home。没有 spill-slot type，也不需要给下一层分配器开后门。

外层以单调决策保证收敛：allocation/content residency 只向更慢层移动，pipeline depth 只减不增，已经拆分的
live segment 不重新合并。若下一层仍不可行则继续向下；若 mandatory working set 不可行则退回 tiling，
而不是无限插 copy。

---

## 8. 与 One-Shot Bufferize 的精确契约

### 8.1 复用哪些现有能力

本设计直接依赖标准 op 已有的 BufferizableOpInterface：

- `bufferization.alloc_tensor`：bufferize 成新 `memref.alloc`，并把 `memory_space` 放入 memref type；
- `bufferization.materialize_in_destination`：强制写入 destination buffer，通常降为 `memref.copy`；
- `tensor.extract_slice`：降为 `memref.subview`；
- Linalg DPS op：复用 `outs` buffer；
- SCF region/iter_args：传播 equivalent buffer；
- `bufferization.to_tensor`：保留 DRAM ABI buffer identity。

因此无需为三种 memory type 注册外部模型，也无需让所有计算 op 接受自定义 memdesc。

### 8.2 推荐选项与边界

最终运行建议遵守：

- `allowUnknownOps = false`；
- kernel 使用 memref DRAM ABI 时保持 `bufferizeFunctionBoundaries = false`；
- 不使用 `use-encoding-for-memory-space`；
- 内部 unresolved memory space 返回 failure，等价于启用严格的 `must-infer-memory-space`；
- kernel ABI 已经是显式 DRAM memref，因此不依赖 tensor function-boundary 默认推断；
- 所有需要新存储的 tensor 都已由显式 `alloc_tensor` 表达；
- 所有跨层内容迁移都已由 `materialize_in_destination` 表达。

One-Shot 可以先用 `insertTensorCopies` 在 tensor 层消解 RaW conflict；其产生的 copy-form allocation 立即
规范化并重新参加 placement/pipeline 规划。最终 Bufferize 前再运行一次分析检查，要求：

```text
number of implicit SRAM allocation decisions == 0
```

测试中可以比对计划好的 live allocation 与最终 `memref.alloc`（忽略 dead allocation 消除），但正确性契约
不是脆弱的“op 个数必须逐字相等”，而是任何最终 One-Shot 想新增的 SRAM storage 都属于编译错误。
DRAM runtime 临时量是否允许隐式生成可单独配置；v0 建议同样禁止，以保持可预测性。

### 8.3 一个重要的回归测试

至少要有一个双缓冲测试证明：

```text
2-slot alloc_tensor（循环外）
+ 两个静态 disjoint tensor.extract_slice
+ scf.for iter_args 传递更新版本
```

经过 One-Shot 后只产生一个对应 ring 的 `memref.alloc` 和两个 `memref.subview`，循环体内不新增
`memref.alloc`。这是“显式 storage graph 真正被 One-Shot 保留”的最小可执行定义。

---

## 9. Bufferize 后的静态 SRAM 分配

### 9.1 按 memory space 独立规划

One-Shot 后的正常 IR 只需要标准形式：

```mlir
%a = memref.alloc() : memref<..., #spm.memory_space<l1>>
%b = memref.alloc() : memref<..., #spm.memory_space<l2>>
memref.copy %b, %a : ...
```

静态 planner 必须先按 memory space 分桶：

```text
L1 allocs → L1 arena / L1 capacity / L1 alignment & bank policy
L2 allocs → L2 arena / L2 capacity / L2 alignment & bank policy
DRAM      → ABI/runtime/global planner，不和 SRAM 混装
```

绝不能把不同 memory space 的 alloc 放进一个默认-space arena。`memref.view` 的 base/result 需要兼容的
memory space，而且两个物理存储本来就不是同一地址域。

当前 `StaticMemoryPlannerAnalysis` 若仍是“收集所有 `memref.alloc`，建立一个默认-space arena”的实现，
必须先改成按 space 分桶，再接入本设计。pipeline 的 allocation hoisting 则负责满足它“SRAM alloc 位于
可静态分析 block、shape 静态”的 eligibility 条件。

### 9.2 Arena lowering

对每个 space 独立执行：

1. 收集 allocation root、所有 alias/view 和真实 dealloc；
2. 计算字节数、alignment、bank/stripe 约束和 lifetime；
3. 先算 `maxLive`，再用 best-fit/图着色等算法分配 offset；
4. 验证 `packedHighWater <= capacity`；
5. 建立该 space 的 arena base；
6. 把每个 `memref.alloc` 替换成 arena 上的 `memref.view`/`reinterpret_cast`；
7. 删除被 arena 吸收的 alloc/dealloc。

概念结果：

```mlir
%l1_arena = ... : memref<1048576xi8, #spm.memory_space<l1>>
%c32768 = arith.constant 32768 : index
%tile = memref.view %l1_arena[%c32768][]
    : memref<1048576xi8, #spm.memory_space<l1>>
      to memref<4096x32xf16, #spm.memory_space<l1>>
```

offset 不必再作为 tensor alloc 上的属性穿过 One-Shot；arena view 本身就是最终地址决定。

### 9.3 Deallocation 与 liveness

最终形态应由 static planner 直接在 alias/view closure 上计算第一次真实 reservation 和最后一次真实 use，
而不是要求一个手写 dealloc 决定 lifetime。对异步 DMA，最后 use 必须延长到 completion/wait；在异步 op
尚未落地的 v0 中，可保守地把整个 pipeline ring 保留到 owning loop 结束。

当前实现若仍以 alloc/dealloc 位置为输入，可以先用下面的过渡 pipeline：

```text
One-Shot Bufferize
→ buffer-deallocation-pipeline
→ optimize-allocation-liveness
→ StaticMemoryPlan
```

`optimize-allocation-liveness` 必须显式加入；标准 deallocation pipeline 不会自动替你运行它。长期实现应让
SRAM planner 直接使用真实 alias/use lifetime 并消除静态 alloc；ownership deallocation 只处理未进入
arena 的 DRAM/runtime allocation。

静态 planner 不应把 tensor 阶段手写的“释放提示”当权威，也不应只因为 allocation 被提升就把内容的
出生点错误设成函数入口。v0 可用优化后的 alloc-to-dealloc 保守 root lifetime；更精确版本使用 §7.2 的
reservation interval 做 modulo packing。

### 9.4 DMA lowering 放在 One-Shot 之后

跨空间 `memref.copy` 最终统一降成一个通用后端 DMA 家族，例如：

```mlir
%token = spm.dma.start %src, %dst : (...) -> !async.token
spm.dma.wait %token
```

DMA verifier 从两个 builtin memref 的 memory space 推导方向并检查相邻层级、shape/layout、对齐和大小。
不再设计 `l1_load`、`l1_store`、`l2_load`、`l2_store` 四套名字。这个后端 op 出现在 One-Shot 之后，
所以完全不需要 BufferizableOpInterface。

---

## 10. 完整例子：LLM decode GEMV

场景：

```text
input  : tensor<1x4096xf16>        = 8 KiB
weight : tensor<4096x4096xf16>     = 32 MiB
output : tensor<1x4096xf16>        = 8 KiB

L2 capacity = 4 MiB
L1 capacity = 1 MiB
L2 N tile   = 64  → weight tile    = 512 KiB
L1 N tile   = 32  → weight subtile = 256 KiB
pipeline depth = 2
```

这里故意让 L2 和 L1 使用不同颗粒度。若把 512 KiB 的 L2 tile 原样放进 L1 并做双缓冲，单是两个 weight
slot 就已经占满 1 MiB，再加 input/output 必然超容量。这个失败不能靠 spilling 修复，因为两个 weight
slot 是 pipeline 同时需要的 mandatory working set；正确动作是把 L1 tile 降到 N=32，或把 L1 depth
降到 1。

### 10.1 两轮 tiling 和显式 materialization 之后

下面只展示关键 storage edge，所有 SSA 数据类型仍然是 builtin tensor：

```mlir
func.func @decode_gemv(
    %input_mem: memref<1x4096xf16, #spm.memory_space<dram>>,
    %weight_mem: memref<4096x4096xf16, #spm.memory_space<dram>>,
    %output_mem: memref<1x4096xf16, #spm.memory_space<dram>>) {
  %input = bufferization.to_tensor %input_mem restrict
      : memref<1x4096xf16, #spm.memory_space<dram>>
        to tensor<1x4096xf16>
  %weight = bufferization.to_tensor %weight_mem restrict
      : memref<4096x4096xf16, #spm.memory_space<dram>>
        to tensor<4096x4096xf16>

  // 小 input 在 L2/L1 各常驻一份，只搬一次。
  %input_l2_storage = bufferization.alloc_tensor()
      <{memory_space = #spm.memory_space<l2>}> : tensor<1x4096xf16>
  %input_l2 = bufferization.materialize_in_destination
      %input in %input_l2_storage : (...) -> tensor<1x4096xf16>

  %input_l1_storage = bufferization.alloc_tensor()
      <{memory_space = #spm.memory_space<l1>}> : tensor<1x4096xf16>
  %input_l1 = bufferization.materialize_in_destination
      %input_l2 in %input_l1_storage : (...) -> tensor<1x4096xf16>

  scf.for %n = %c0 to %c4096 step %c64 {
    // L2 tiling：完整 weight 从不进入 L2，只搬当前 4096x64 outer tile。
    %weight_dram_tile = tensor.extract_slice %weight[0, %n]
        [4096, 64] [1, 1]
        : tensor<4096x4096xf16> to tensor<4096x64xf16>
    %weight_l2_storage = bufferization.alloc_tensor()
        <{memory_space = #spm.memory_space<l2>}> : tensor<4096x64xf16>
    %weight_l2 = bufferization.materialize_in_destination
        %weight_dram_tile in %weight_l2_storage
        : (tensor<4096x64xf16>, tensor<4096x64xf16>)
          -> tensor<4096x64xf16>

    scf.for %nn = %c0 to %c64 step %c32 {
      // L1 tiling：从 L2 tile 中再取 4096x32 subtile。
      %weight_l2_subtile = tensor.extract_slice %weight_l2[0, %nn]
          [4096, 32] [1, 1]
          : tensor<4096x64xf16> to tensor<4096x32xf16>
      %weight_l1_storage = bufferization.alloc_tensor()
          <{memory_space = #spm.memory_space<l1>}> : tensor<4096x32xf16>
      %weight_l1 = bufferization.materialize_in_destination
          %weight_l2_subtile in %weight_l1_storage
          : (tensor<4096x32xf16>, tensor<4096x32xf16>)
            -> tensor<4096x32xf16>

      %zero = arith.constant 0.0 : f16
      %output_l1_storage = bufferization.alloc_tensor()
          <{memory_space = #spm.memory_space<l1>}> : tensor<1x32xf16>
      %output_l1_init = linalg.fill ins(%zero : f16)
          outs(%output_l1_storage : tensor<1x32xf16>)
          -> tensor<1x32xf16>
      %computed = linalg.matmul
          ins(%input_l1, %weight_l1
              : tensor<1x4096xf16>, tensor<4096x32xf16>)
          outs(%output_l1_init : tensor<1x32xf16>)
          -> tensor<1x32xf16>

      // L1 不能直达 DRAM，先写入短命 L2 output staging。
      %output_l2_storage = bufferization.alloc_tensor()
          <{memory_space = #spm.memory_space<l2>}> : tensor<1x32xf16>
      %output_l2 = bufferization.materialize_in_destination
          %computed in %output_l2_storage
          : (tensor<1x32xf16>, tensor<1x32xf16>) -> tensor<1x32xf16>

      %out_col = arith.addi %n, %nn : index
      %output_dram_tile = memref.subview %output_mem[0, %out_col]
          [1, 32] [1, 1]
          : memref<1x4096xf16, #spm.memory_space<dram>>
            to memref<1x32xf16,
                strided<[4096, 1], offset: ?>, #spm.memory_space<dram>>
      bufferization.materialize_in_destination
          %output_l2 in writable %output_dram_tile
          : (tensor<1x32xf16>,
             memref<1x32xf16,
               strided<[4096, 1], offset: ?>, #spm.memory_space<dram>>) -> ()
    }
  }
  return
}
```

此时循环内的 `alloc_tensor` 仍是“每迭代一个逻辑 tile”的表示，不能直接 Bufferize。

### 10.2 Pipeline materialization 之后

PlanPipeline 计算出 outer/inner transfer 各需要两个 slot。MaterializePipeline 在 outer loop 之前建立：

```mlir
%weight_l2_ring = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l2>}>
    : tensor<2x4096x64xf16>       // 2 × 512 KiB

%weight_l1_ring = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l1>}>
    : tensor<2x4096x32xf16>       // 2 × 256 KiB

%output_l1_ring = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l1>}>
    : tensor<2x1x32xf16>

%output_l2_ring = bufferization.alloc_tensor()
    <{memory_space = #spm.memory_space<l2>}>
    : tensor<2x1x32xf16>
```

然后为每个 ring 取静态 ping/pong slice，生成 prologue、K-way unrolled steady state 和 epilogue，并通过
iter_args 传递每次 materialize 后的 tensor version。原来的四个 loop-local allocation 全部消失。

容量至少计入：

```text
L2: resident input 8 KiB
    + weight ring 2 × 512 KiB
    + output ring 2 × 64 B
    + alignment/bank padding
    < 4 MiB

L1: resident input 8 KiB
    + weight ring 2 × 256 KiB
    + output ring 2 × 64 B
    + accumulator/scratch/alignment
    < 1 MiB
```

这个例子固定以下语义：

- 完整 weight 始终只在 DRAM；
- L2/L1 tile shape 由各自 tiling pass 决定，不能靠 type 名字区分；
- input residency 由 loop-external allocation 表达；
- weight/output 的动态迭代实例由固定 ring 表达；
- 双缓冲的两份容量必须在 Bufferize 前就计入；
- 若 L1 planner 后来驱逐 `%input_l1`，它应优先复用有效的 `%input_l2` home，而不是无条件重走 DRAM；
- One-Shot 后 L1/L2 循环体内不得出现新 `memref.alloc`。

---

## 11. 关键不变量

1. Bufferize 前只有 builtin tensor，不用 type/encoding 表示 L1/L2/DRAM。
2. placement 只属于显式 allocation root；`memory_space` 是唯一稳定真相。
3. 每个实体化 tensor 都能唯一追溯到 `alloc_tensor` 或显式 DRAM ABI root。
4. allocation 与 copy 分离；稳定形态不含 `alloc_tensor copy(...)`。
5. 所有跨 placement edge 都经过 `materialize_in_destination`。
6. hardware compute 的 operand/outs 由 L1 root 支撑。
7. 所有跨层 copy 只连接相邻层；禁止 DRAM ↔ L1 直达。
8. 不同 memory space 的 root 不直接在控制流 join 合并。
9. 两轮 tiling 产生不同颗粒度的 allocation；不靠后续 pass 猜 tile 属于哪一层。
10. 所有 SRAM allocation 都有静态大小或编译期上界。
11. software pipeline 的 slotCount 在容量规划中显式计费。
12. 最终 Bufferize 前，所有 L1/L2 allocation 已 slotize 并位于重复执行 scope 外。
13. 提升后仍通过 SSA/iter_args 传递更新的 tensor version，不依赖隐式 mutation。
14. DMA source/destination 的 reservation 至少延长到 transfer completion/wait。
15. clean/dirty 是 content-version 属性，不是 allocation root 的全局 bool。
16. dirty 且无有效 home 的内容在 eviction 前必须 writeback。
17. spill/fill 新增的 allocation 也必须经过 pipeline/slotization/hoisting。
18. mandatory working set 超容量时降低 depth 或 retile，不把正在使用的 buffer 当 victim。
19. 最终 One-Shot 不得引入未规划的 L1/L2 allocation。
20. L1/L2 分别建 arena；最终每层 `packedHighWater <= capacity`。

---

## 12. 测试矩阵

| 类别 | 必测内容 |
|---|---|
| 类型统一 | 整个 tensor pipeline 中不存在 `!spm.l1mem` / `!spm.l2mem` / `!spm.dram` 和 tier encoding |
| 显式 placement | `alloc_tensor(memory_space)` 经 One-Shot 变成相同 space 的 `memref.alloc` |
| 显式 copy | `materialize_in_destination` 写入指定 destination，source/result 内容关系正确 |
| 硬件路径 | DRAM↔L1 直接 materialize 报错；DRAM↔L2↔L1 合法 |
| 两级 tiling | L2/L1 allocation shape 分别等于 outer tile/subtile shape |
| Output-only | 纯 output promotion 只有 alloc，没有初始 copy |
| Tail | 尾块仍使用静态最大 slot，动态有效范围不改变 allocation 大小 |
| 双缓冲 | 两个静态 disjoint slot，prologue/kernel/epilogue 和 iter_args 正确 |
| Alloc hoist | 最终 tensor IR 和 memref IR 的 repetitive region 内均无 L1/L2 alloc |
| 无隐藏分配 | final One-Shot 的 SRAM alloc 数与计划一致 |
| 容量 | depth=2 按两份 slot 计费；GEMV 的 512 KiB L1 tile 正确失败、256 KiB 正确通过 |
| Clean eviction | 已有 L2 home 时不产生 L1→L2 store，只产生需要的 reload |
| Dirty eviction | 新建 L2 home 并 writeback；之后只读的重复 eviction 不重复 store |
| 分支 | 多 use cluster 产生各自受支配的 fill，不做错误的线性“之后全替换” |
| Arena 隔离 | L1/L2 alloc 分别进入各自 arena，绝不跨 space 复用 offset/base |
| Arena 复用 | 不重叠的 allocation 在同一 space 可得到相同 offset |
| 终态验证 | 每层 high-water、alignment、bank 约束和容量全部满足 |

---

## 13. 建议实现顺序

1. 定义唯一的 `#spm.memory_space<dram|l2|l1>` attr，并从 target description 提供容量、对齐和硬件邻接关系。
2. 实现 placement provenance/verifier，先覆盖 `alloc_tensor`、`to_tensor`、slice/cast、Linalg DPS 和 SCF。
3. 复用 `transform.structured.promote_tensor` 的实现，落地 `MaterializeTensorStorage`；稳定输出只有标准 op。
4. 让 L2 tiling 返回/标注 L2 materialization boundary，并立即消费为 L2 alloc/copy。
5. 让 L1 tiling 对 inner tile、accumulator、output 和 scratch 做同样处理。
6. 接入 One-Shot `insertTensorCopies`，并把 copy-form alloc 规范化为独立 alloc + materialize。
7. 实现 PlanPipeline 的 stage、II、slotCount 和 reservation 计算；先支持 depth=1/2、静态 trip count/尾部。
8. 实现 K-way partial unroll、静态 ping/pong slice、tensor iter_args threading 和 alloc hoist。
9. 写“pipeline 后 One-Shot 不新增 alloc”的硬性回归测试，再扩大 pipeline 覆盖面。
10. 实现 stage-aware `maxLive` 和 mandatory/resident 分类；先只做容量诊断和 depth fallback。
11. 实现 content-version/validHomes 数据流，再做 clean/dirty spill、use-cluster split 和分层 `PlanTier`。
12. 把 pipeline/storage 连接成单调迭代，确保 spill 新增的 copy 也被 slotize/hoist。
13. 修改 static memory planner：按 memory space 分桶、每层独立 arena/capacity，并保持 arena/view space 一致。
14. 接标准 deallocation/liveness 优化，完成 best-fit/图着色 offset 分配和 high-water 校验。
15. 为 `#spm.memory_space` 注册到后端整数 address space 的 attribute conversion；若 MVP 直接用整数，
    也必须由同一 target description 提供映射。
16. 最后把跨空间 `memref.copy` 降到通用 async DMA op，并让 wait 进入 reservation/liveness。

每一步都应保持一个明确 gate：一旦进入下一阶段，上一阶段禁止的隐式 allocation、未消费标注或循环内
SRAM alloc 必须立即报错，不能留给后续 pass 猜测和修补。

---

## 14. v0 边界与后续优化

v0 明确支持：

- builtin tensor + 显式 `alloc_tensor(memory_space)`；
- 两级静态/上界静态 tile；
- 相邻层同步 materialize；
- depth 1/2 的结构化 `scf.for` pipeline；
- K-way partial unroll 和静态 ping/pong subset；
- loop-invariant residency、loop-local stream tile、output staging；
- whole-root 或静态 disjoint subset 的 spilling；
- 每 memory space 独立的保守 arena packing。

以下内容后续再做：

- 用动态 `%iv mod K` 选择 ring slot，同时证明跨迭代 subset disjoint；
- 任意 CFG、不可规约循环和复杂 loop-carried recurrence 的 spilling；
- 同一 allocation 内更细的部分写 dirty tracking；
- stage-aware modulo arena packing，让同一 loop 内不同 root 的空闲 reservation 交错复用；
- bank-aware tile/layout 联合搜索；
- DMA/compute 时间模型和精确 overlap cost；
- 跨函数的 SRAM residency 与 caller/callee arena 协调。

这些扩展都不要求重新引入 tier-specific tensor type。它们只会增强 placement analysis、pipeline schedule
或最终 arena planner；本版确定的 IR 主轴保持不变。
