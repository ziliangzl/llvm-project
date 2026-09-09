# 尾块规范化（NormalizeTileShapes）：期望 IR 与实测

> 配套文档：`SPM_PRE_BUFFERIZE_DESIGN.md` §3.3（尾块规范化）、§11 第 3 步。
> 本文只落实 §3.3 的 **S1（peel）** 一条路径，而且只考虑**输入 IR 全静态 shape**的情形
> （用户约束：tiling 之前的输入一定是静态 shape）。
> 可复现的测试文件：`spm-mlir/test/pre-bufferize/tail-normalize-*.mlir`。
> 全部 IR 都是 `build/bin/mlir-opt` 的真实输出，不是手写的期望值。

---

## 0. 这一步在管什么

`TileForTier(T)` 用 tile size oracle 给出的 tile size 切块。当维度长度不被 tile size 整除时，
**即使输入 shape 全静态**，上游 tiling 也会产生**动态类型**的 tile：extent 变成 `affine.min`，
tile 类型退化成 `tensor<?x?xf32>`。这会连带两个致命后果（§3.3）：

1. `MaterializeTensorStorage` 在这种 tile 上插 `bufferization.alloc_tensor`
   → **循环内动态大小的 SRAM 分配**，违反 §7.2 的 V9（`alloc_tensor` result type 必须全静态）；
2. 内层 loop 的上界变成动态值 → 2-way unroll 的静态残差绑定无从谈起。

所以 §3.2 的步骤顺序是硬的：**第 4 步 NormalizeTileShapes 必须在第 5/6 步 materialize 之前**。
本文给出这一步的 before/after 期望 IR，并实测证明「上游已有的东西够用」。

---

## 1. 结论速查

| 问题                                        | 结论                                                                                      | 依据                          |
| ------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------- |
| 尾块规范化要自己写变换吗                    | **不用。** 变换 100% 是上游的：`transform.loop.peel` / `linalg::peelLoops` / `-scf-for-loop-peeling` | §5.1，实测 N1/N2/N6       |
| 连驱动 pass 都有现成的吗                    | **有。** `-scf-for-loop-peeling=skip-partial=false -canonicalize -cse` 的输出与手写 peel 路线**逐字一致** | 实测 N6（本文 §5.1）  |
| 那还要自己写什么                            | 策略（哪些 loop / 哪些维度）、`skip-partial=false`、出口 gate、尾块标注——不是变换，是驱动层 | 本文 §5.7             |
| `transform.loop.peel` 一步就够吗          | **不够。** 它只把 `affine.min` 换成 `affine.apply`，tile 类型仍是 `tensor<?x64xf32>` | 实测 N2a（本文 §4.1）        |
| 什么才把类型变静态                          | **canonicalize**（+ CSE）。NormalizeTileShapes = peel + canonicalize，两半都必须跑    | 实测 N2b（本文 §4.2）        |
| peel 的顺序                                 | **先内层，再外层**。外层 peel 会整体复制已经 peel 好的内层，只需 2 次调用           | 实测 N2（本文 §5.2）         |
| 已经整除时 peel 会怎样                      | **报 silenceable error** `failed to peel the last iteration`；`fail_if_already_divisible` 属性没被实现读取 | 实测 N4，`SCFTransformOps.cpp:274-282` |
| 两个维度都不整除的代价                      | 2 个维度 → **4 个区域、4 组 tile shape**（本例 64x64 / 64x8 / 2x64 / 2x8）        | 实测 N2（本文 §4.2）         |
| 全静态输入省掉了什么                        | peel 走 constant fast path，尾块 shape 编译期已知，**不需要 value-bounds 推断**     | `LoopSpecialization.cpp:135-138` |
| 规范化之后 `alloc_tensor` 都静态了吗      | **是。** 12 个 `alloc_tensor` 全静态，bufferize 后 0 个动态 memref                | 实测 N3（本文 §4.3）         |
| 尾块要流水吗                                | 不要。尾块 `pipeline_candidate = false`，`slotCount = 1`，走设计文档 §5.7 的 hoist-only    | 设计文档 §3.3 S1              |

---

## 2. 例子

取上游 `mlir/test/Dialect/Linalg/tile-tensors.mlir` 的骨架（`linalg.matmul` + `tile_using_for`），
但按用户约束改成**全静态 shape**，并且**两个 parallel 维都不被 tile size 整除**：

```mlir
func.func @matmul_130x64x200(%A: tensor<130x64xf32>, %B: tensor<64x200xf32>,
                             %C: tensor<130x200xf32>) -> tensor<130x200xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<130x64xf32>, tensor<64x200xf32>)
                     outs(%C : tensor<130x200xf32>) -> tensor<130x200xf32>
  return %0 : tensor<130x200xf32>
}
```

tile size `[64, 64]`（K 不切）：

```text
M = 130 = 2 * 64 + 2     →  主循环 2 趟 tile 64，尾块 tile 2
N = 200 = 3 * 64 + 8     →  主循环 3 趟 tile 64，尾块 tile 8
```

> 尺寸是刻意选的：`130 x 96`（设计文档 §3.3 用的 T10 尺寸）里 N 方向主循环只有 1 趟，
> canonicalize 会把这个单趟 loop 直接折叠掉，看不出「主循环还是循环」。`N = 200` 保证主循环仍是循环。

---

## 3. Before：不做规范化（禁止的形态）

### 3.1 只 tiling 之后

`transform.structured.tile_using_for %mm tile_sizes [64, 64]` + `-canonicalize -cse`：

```mlir
#map  = affine_map<(d0) -> (-d0 + 130, 64)>
#map1 = affine_map<(d0) -> (-d0 + 200, 64)>
%0 = scf.for %arg3 = %c0 to %c130 step %c64 iter_args(%arg4 = %arg2) -> (tensor<130x200xf32>) {
  %1 = scf.for %arg5 = %c0 to %c200 step %c64 iter_args(%arg6 = %arg4) -> (tensor<130x200xf32>) {
    %2 = affine.min #map(%arg3)                       // <== M 方向 extent：64 或 2
    %3 = affine.min #map1(%arg5)                      // <== N 方向 extent：64 或 8
    %extracted_slice   = tensor.extract_slice %arg0[%arg3, 0]     [%2, 64] [1, 1] : tensor<130x64xf32>  to tensor<?x64xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5]     [64, %3] [1, 1] : tensor<64x200xf32>  to tensor<64x?xf32>
    %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<130x200xf32> to tensor<?x?xf32>
    %4 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>)
                       outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>
    ...
```

这就是设计文档 §3.3 记录的 T10 现象，在 `N = 200` 上完全一致。

### 3.2 接着 materialize（这才是真正的问题）

再跑 `transform.structured.promote_tensor to 1`（= `MaterializeTensorStorage`，把 3 个 tile 提到 L1）：

```mlir
%extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%2, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
%4 = bufferization.alloc_tensor(%2) <{memory_space = 1 : i64}> : tensor<?x64xf32>          // <== 动态 L1 分配
%5 = bufferization.materialize_in_destination %extracted_slice in %4 : ...
%6 = bufferization.alloc_tensor(%3) <{memory_space = 1 : i64}> : tensor<64x?xf32>          // <== 动态 L1 分配
...
%8 = bufferization.alloc_tensor(%2, %3) <{memory_space = 1 : i64}> : tensor<?x?xf32>       // <== 动态 L1 分配
```

`-one-shot-bufferize` 之后（**实测 N1**）：

```mlir
%alloc_0 = memref.alloc(%6) alignment = 64 : memref<?x64xf32, 1>      // 循环内
%alloc_2 = memref.alloc(%7) alignment = 64 : memref<64x?xf32, 1>      // 循环内
%alloc_4 = memref.alloc(%6, %7) alignment = 64 : memref<?x?xf32, 1>   // 循环内
```

3 个**循环内、动态大小、space 1** 的分配。static arena planner 对这个形态无能为力
（下游文档 §11 不变量 10），必须在 materialize 之前拦掉 —— 这就是本步骤存在的理由。

测试：`spm-mlir/test/pre-bufferize/tail-normalize-negative.mlir`（这个文件是**反例基线**，
它的 CHECK 就是在固定「不规范化会得到什么」，实现之后不要去改它）。

---

## 4. After：规范化之后的期望 IR

### 4.1 只 peel（中间态，还不够）

`transform.loop.peel` 之后、canonicalize 之前，主循环里是这样：

```mlir
#map  = affine_map<() -> (64)>
#map2 = affine_map<(d0) -> (-d0 + 200)>
#map3 = affine_map<(d0) -> (-d0 + 130)>

%0 = scf.for %arg3 = %c0 to %c128 step %c64 ... {        // 主 M 循环：ub 128，整除
  %2 = scf.for %arg5 = %c0_0 to %c192 step %c64_1 ... {  // 主 N 循环：ub 192，整除
    %4 = affine.apply #map()                             // <== affine.min 变成了 apply(常量 64)
    %5 = affine.apply #map()
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%4, 64] [1, 1]
                     : tensor<130x64xf32> to tensor<?x64xf32>   // <== 类型仍然是动态的！
```

**关键实测（N2a）**：`peelForLoopAndSimplifyBounds` 内部的 `rewriteAffineOpAfterPeeling`
（`LoopSpecialization.cpp:183-207`）只负责把 `affine.min/max` 换成
主循环里的常量 `affine.apply` / 尾循环里的精确余量 `affine.apply`。
**它不改类型**。此时如果直接 materialize，`alloc_tensor` 仍然是 `tensor<?x64xf32>`。

### 4.2 peel + canonicalize（完整形态）

加上 canonicalize + CSE 之后，常量折叠进 `extract_slice`，slice 的 folder 把 result type 变静态，
`linalg.matmul` 的类型跟着静态化，单趟的尾循环被 single-iteration promotion 折叠掉：

```mlir
%0 = scf.for %arg3 = %c0 to %c128 step %c64 iter_args(%arg4 = %arg2) -> (tensor<130x200xf32>) {
  // ---- 区域 A：main M x main N，唯一的两层循环 ----
  %9 = scf.for %arg5 = %c0 to %c192 step %c64 iter_args(%arg6 = %arg4) -> (tensor<130x200xf32>) {
    %es_6 = tensor.extract_slice %arg0[%arg3, 0]     [64, 64] [1, 1] : tensor<130x64xf32>  to tensor<64x64xf32>
    %es_7 = tensor.extract_slice %arg1[0, %arg5]     [64, 64] [1, 1] : tensor<64x200xf32>  to tensor<64x64xf32>
    %es_8 = tensor.extract_slice %arg6[%arg3, %arg5] [64, 64] [1, 1] : tensor<130x200xf32> to tensor<64x64xf32>
    %23 = linalg.matmul ins(%es_6, %es_7 : tensor<64x64xf32>, tensor<64x64xf32>)
                        outs(%es_8 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %is_9 = tensor.insert_slice %23 into %arg6[%arg3, %arg5] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<130x200xf32>
    scf.yield %is_9 : tensor<130x200xf32>
  }
  // ---- 区域 B：main M x tail N（trip count 1，循环已被折叠成直线代码）----
  %es_2 = tensor.extract_slice %arg1[0, 192]     [64, 8] [1, 1] : tensor<64x200xf32>  to tensor<64x8xf32>
  %es_3 = tensor.extract_slice %arg0[%arg3, 0]   [64, 64][1, 1] : tensor<130x64xf32>  to tensor<64x64xf32>
  %es_4 = tensor.extract_slice %9[%arg3, 192]    [64, 8] [1, 1] : tensor<130x200xf32> to tensor<64x8xf32>
  %16 = linalg.matmul ins(%es_3, %es_2 : tensor<64x64xf32>, tensor<64x8xf32>)
                      outs(%es_4 : tensor<64x8xf32>) -> tensor<64x8xf32>
  scf.yield ... : tensor<130x200xf32>
}
// ---- 区域 C：tail M x main N（trip count 1 的 M 尾块 + 3 趟 N 主循环）----
%1 = scf.for %arg3 = %c0 to %c192 step %c64 iter_args(%arg4 = %0) -> (tensor<130x200xf32>) {
  %es_2 = tensor.extract_slice %arg0[128, 0]     [2, 64] [1, 1] : tensor<130x64xf32>  to tensor<2x64xf32>
  %es_3 = tensor.extract_slice %arg1[0, %arg3]   [64, 64][1, 1] : tensor<64x200xf32>  to tensor<64x64xf32>
  %es_4 = tensor.extract_slice %arg4[128, %arg3] [2, 64] [1, 1] : tensor<130x200xf32> to tensor<2x64xf32>
  %15 = linalg.matmul ins(%es_2, %es_3 : tensor<2x64xf32>, tensor<64x64xf32>)
                      outs(%es_4 : tensor<2x64xf32>) -> tensor<2x64xf32>
  scf.yield ... : tensor<130x200xf32>
}
// ---- 区域 D：tail M x tail N（直线代码）----
%es   = tensor.extract_slice %arg0[128, 0]   [2, 64] [1, 1] : tensor<130x64xf32>  to tensor<2x64xf32>
%es_0 = tensor.extract_slice %arg1[0, 192]   [64, 8] [1, 1] : tensor<64x200xf32>  to tensor<64x8xf32>
%es_1 = tensor.extract_slice %1[128, 192]    [2, 8]  [1, 1] : tensor<130x200xf32> to tensor<2x8xf32>
%8 = linalg.matmul ins(%es, %es_0 : tensor<2x64xf32>, tensor<64x8xf32>)
                   outs(%es_1 : tensor<2x8xf32>) -> tensor<2x8xf32>
return ...
```

**期望效果，逐条**：

| 期望                          | 实测结果                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------ |
| 所有 tile 类型静态            | ✅ 整个函数里 `tensor<?` 出现 **0** 次（FileCheck `--implicit-check-not`） |
| 主循环 trip count 静态且整除  | ✅ `0..128 step 64`、`0..192 step 64`                                    |
| 尾块 shape 编译期已知         | ✅ M 尾块 2，N 尾块 8                                                          |
| 区域数                        | 4（= 2^不整除维数）                                                            |
| tile shape 组数               | 4：`64x64` / `64x8` / `2x64` / `2x8`                                     |
| 单趟循环                      | 被 canonicalize 折叠成直线代码（区域 B、D 的 N 维；区域 C、D 的 M 维）         |

### 4.3 规范化之后再 materialize（出口 gate）

同一个脚本里接 `promote_tensor to 1`（**实测 N3**）：

```mlir
// 区域 A（两层循环内）
%17 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
%19 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
%21 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
// 区域 B
%10 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
%12 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
%14 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
// 区域 C
%9  = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
%11 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>
%13 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
// 区域 D
%2 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x64xf32>
%4 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x8xf32>
%6 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<2x8xf32>
```

12 个 `alloc_tensor`，**全部静态**。`-one-shot-bufferize` 之后 13 个 `memref.alloc`
（12 个 space 1 + 1 个 space 0 的 DRAM 输出 staging），`memref<?` 出现 **0** 次。

V9 在这个例子上成立。剩下的问题是「区域 A 的 3 个 alloc 还在循环内」——那是
`SPM_HOIST_ONLY_EXAMPLE.md`（§5.7 HoistOnly）的活，不是本步骤的活。

测试：`spm-mlir/test/pre-bufferize/tail-normalize-peel.mlir`。

---

## 5. 实现要点（都有实测依据）

### 5.1 用哪个上游入口

```text
C++ 直接可用的四个层次（越靠下越省事）：
  -scf-for-loop-peeling[=skip-partial=false]        <== 现成的 pass，见下方 N5/N6
      mlir/include/mlir/Dialect/SCF/Transforms/Passes.td:24-36
      对 func 内所有 scf.for 无脑 peel；结束时自己清掉 __peeled_loop__ 标记
      （LoopSpecialization.cpp:356-370），不留残渣
  scf::peelForLoopAndSimplifyBounds(rewriter, forOp, partialIteration)
      mlir/include/mlir/Dialect/SCF/Transforms/Transforms.h:101
      = peelForLoop + rewriteAffineOpAfterPeeling，已整除时返回 failure
  mlir::linalg::peelLoop / peelLoops(rewriter, loops)
      mlir/include/mlir/Dialect/Linalg/Transforms/Transforms.h:682
      mlir/lib/Dialect/Linalg/Transforms/Transforms.cpp:57-76
      包了一层，**把 failure 咽掉**（"assert(!partialIteration)" 之后原样返回）
  transform.loop.peel
      mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td:171
      调试/写 lit 用；已整除时报 silenceable error

推荐：v0 直接在 pipeline 里排 `-scf-for-loop-peeling=skip-partial=false -canonicalize -cse`
      即可跑通（实测 N6）；等需要「只 peel 某些维度」的策略时，再换成自己驱动
      mlir::linalg::peelLoops + canonicalize（§5.7）。
```

**实测 N5（默认选项是个坑）**：`-scf-for-loop-peeling` 的 `skip-partial` **默认 true**，
它会**故意不 peel** 位于别人尾块迭代里的内层循环（`LoopSpecialization.cpp:329-337`，
理由是「partial iteration 通常不是性能关键」）。于是 M 尾块那一支的 N 循环没被 peel：

```mlir
%1 = scf.for %arg3 = %c0 to %c200 step %c64 iter_args(%arg4 = %0) -> (tensor<130x200xf32>) {
  %2 = affine.min #map(%arg3)
  %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg3] [64, %2] [1, 1] : tensor<64x200xf32> to tensor<64x?xf32>
  %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<2x64xf32>, tensor<64x?xf32>)
                     outs(%extracted_slice_1 : tensor<2x?xf32>) -> tensor<2x?xf32>
```

对普通 CPU 后端这个默认值是对的（省代码体积），对我们是**致命的**：这一支照样会 materialize
出动态大小的 L1 分配。所以 `skip-partial=false` 必须显式给。

**实测 N6**：`-scf-for-loop-peeling=skip-partial=false -canonicalize -cse` 的输出与
§4.2 手写 peel（内→外两次 `transform.loop.peel`）+ canonicalize 的输出**逐字相同**
（`diff` 只差函数名）。测试：`tail-normalize-stock-pass.mlir`。

### 5.2 顺序：内层先 peel

```text
for each tiled loop nest:
    for L in loops(innermost -> outermost):
        if staticTripCount(L) 且 (ub - lb) % step == 0: continue     // 见 5.4
        peel(L)
```

**实测 N2**：先 peel 内层再 peel 外层，只需要 2 次调用 —— 外层 peel 时会把「已经 peel 好的内层结构」
整体克隆进主/尾两份，于是自动得到 4 个区域。反过来（先外层）则每 peel 一次就要重新 match 一次内层，
handle 数翻倍，没有任何好处。

### 5.3 canonicalize 是这一步的组成部分，不是「顺便跑的清理」

**实测 N2a/N2b**：peel 之后类型仍是动态的；把类型变静态的是 canonicalize
（`affine.apply` 折成常量 → `tensor.extract_slice` 的 folder 生成静态 result type → `linalg` op 类型静态化）。
所以 pass 的实现必须在 peel 之后**自己**跑一遍 canonicalize（`applyPatternsGreedily` +
`tensor`/`affine`/`scf` 的 canonicalization pattern，或直接在 pipeline 里紧跟一个 `-canonicalize -cse`），
并在**结束时**检查「本 nest 内不存在动态 tile 类型」——不要把这个检查留给下游。

顺带一个副作用要写进设计假设：canonicalize 会把 trip count = 1 的循环折叠掉
（本例的 N 尾块、M 尾块），**循环嵌套的层数和 loop handle 都会变**。
因此 `PlanPipeline` 必须在规范化**之后**重新发现循环，不能复用 `TileForTier` 拿到的 loop handle。

### 5.4 已整除时必须自己先判断

**实测 N4**（`tail-normalize-already-divisible.mlir`）：`128 x 64 * 64 x 192`、tile `[64, 64]`
全部整除，此时 `transform.loop.peel` 报

```text
error: failed to peel the last iteration
```

根因在 `LoopSpecialization.cpp:135-138`：

```cpp
// No specialization necessary if step already divides upper bound evenly.
// Fast path: lb, ub and step are constants.
if (lbInt && ubInt && stepInt && (*ubInt - *lbInt) % *stepInt == 0)
  return failure();
```

而且 `LoopPeelOp` 的 `fail_if_already_divisible` 属性**在实现里根本没被读**
（`SCFTransformOps.cpp:274-282` 只把 `LogicalResult` 无条件翻译成 error）。
两个可用做法：pass 自己先算 `(ub - lb) % step`（全静态输入下这一定可判定，也可以用
`LoopLikeOpInterface::getStaticTripCount`，`LoopLikeInterface.td:239-249`），
或者用 `linalg::peelLoops`（它咽掉 failure）。另外 `peelForLoop` 对 `step <= 1` 也直接返回 failure，
所以 tile size = 1 的维度天然跳过。

### 5.5 全静态输入让这一步简单在哪

- peel 走 `lbInt && ubInt && stepInt` 的 constant fast path，不需要 `affine.apply` 的动态余量计算；
- 尾块 shape 编译期已知（本例 2 和 8）→ 尾块的 L1/L2 分配大小是常量，容量核算可以直接做；
- 不需要 `transform.tensor.make_loop_independent` / value-bounds 那一套（§3.3 的 S2 才需要）；
- V9 是可以**结构性保证**的，不是「尽力而为」：本 pass 出口处不允许残留任何动态 tile 类型。

### 5.6 什么时候仍然需要 S2/S3

S1 的代价是 **2^d 个区域**（d = 不整除的维数）。本例 d=2 已经是 4 个区域、4 组 tile shape；
d=3 就是 8 个。两个后果：

- 代码体积：每个区域是一份独立的 compute 代码；
- allocation 组数：4 组 tile shape = 4 组 allocation root。arena 层面它们生命周期互不重叠，
  可以复用同一段 offset（下游文档 §9），但 planner 要处理 4 组而不是 1 组。

所以 v0 的策略建议是：**只对「尾块占比不可忽略」的维度用 S1**，其余维度用 S2（按上界静态分配 +
`extract_slice` 取有效子范围）。判据放在 tile size oracle 旁边即可，本文不再展开——S2/S3 的
IR 形态留到需要 mask op 时再定（设计文档 §10 的 backlog）。

---

### 5.7 「全用上游」到底剩下什么要自己写

变换本体一行都不用写。需要自己写的是**驱动层**，都很薄：

| 要自己写的                                    | 为什么上游给不了                                                                 |
| --------------------------------------------- | -------------------------------------------------------------------------------- |
| 传 `skip-partial=false`                     | 上游默认值为普通后端优化代码体积，对 SRAM 分配是错的（N5）                        |
| 挑 loop：只 peel tiling 产生的循环            | stock pass 对 func 内**所有** `scf.for` 生效，包括用户原有的、不该动的循环 |
| 挑维度：S1 还是 S2（§5.6）                   | 这是容量/代码体积的 trade-off，上游没有这个概念                                   |
| 出口 gate：本 nest 内无动态 tile 类型         | 上游不检查；不 gate 的话错误会漂到 bufferize 之后才炸                             |
| 尾块标注 `pipeline_candidate = false`       | 我们自己的 pipeline 语义（§3.3 硬规则）                                          |
| 记录「4 组 tile shape」给容量 planner         | 我们自己的 §3.1.1 容量核算                                                       |

所以结论是：**这一步没有算法风险，只有集成工作量**。v0 甚至可以先只做前两条
（排 stock pass + 出口 gate），策略部分等 tile size oracle 接上再补。

---

## 6. 与设计文档的接口

| 设计文档条目                | 本文的落实                                                                        |
| --------------------------- | --------------------------------------------------------------------------------- |
| §3.2 步骤 4                | NormalizeTileShapes = `linalg::peelLoops`（内→外，跳过已整除）+ canonicalize/CSE |
| §3.3 S1                    | 本文全部内容；S2/S3 未实现，判据见 §5.6                                          |
| §3.5 P2（tile 类型静态）   | 本 pass 的出口检查                                                                |
| §7.2 V9                    | 实测 N3 在本例上成立                                                              |
| §3.3 硬规则「尾块不流水」  | 尾块区域标 `pipeline_candidate = false`；其循环内 alloc 走设计文档 §5.7 hoist-only     |
| §11 第 3 步的 gate         | `tail-normalize-peel.mlir` 就是这个 gate；`tail-normalize-negative.mlir` 是反例基线 |

---

## 7. 复现

```sh
cd spm-mlir/test/pre-bufferize
B=../../../build/bin

# N1 反例：不规范化 -> 循环内动态 L1 分配
$B/mlir-opt tail-normalize-negative.mlir --transform-interpreter -canonicalize -cse \
  | $B/FileCheck tail-normalize-negative.mlir
$B/mlir-opt tail-normalize-negative.mlir --transform-interpreter -canonicalize -cse -one-shot-bufferize \
  | $B/FileCheck tail-normalize-negative.mlir --check-prefix=BUF

# N2/N3 正例：peel + canonicalize -> 全静态；再 materialize -> 12 个静态 alloc_tensor
$B/mlir-opt tail-normalize-peel.mlir --transform-interpreter \
  | $B/FileCheck tail-normalize-peel.mlir --implicit-check-not="tensor<?"
$B/mlir-opt tail-normalize-peel.mlir --transform-interpreter -one-shot-bufferize \
  | $B/FileCheck tail-normalize-peel.mlir --check-prefix=BUF --implicit-check-not="memref<?"

# N4 已整除时 peel 报错
$B/mlir-opt tail-normalize-already-divisible.mlir --transform-interpreter -verify-diagnostics

# N5/N6 现成的上游 pass：默认选项留下动态尾块，skip-partial=false 才全静态
$B/mlir-opt tail-normalize-stock-pass.mlir -scf-for-loop-peeling -canonicalize -cse \
  | $B/FileCheck tail-normalize-stock-pass.mlir --check-prefix=SKIPPARTIAL
$B/mlir-opt tail-normalize-stock-pass.mlir -scf-for-loop-peeling=skip-partial=false -canonicalize -cse \
  | $B/FileCheck tail-normalize-stock-pass.mlir --implicit-check-not="tensor<?"
```

### 实验记录

| #   | 内容                                                   | 结果                                                                          |
| --- | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| N1  | 全静态不整除，只 tiling + promote，再 bufferize        | 循环内 3 个 `memref.alloc(%..) : memref<?x?xf32, 1>`（禁止形态）            |
| N2a | tile + peel（内→外），**不** canonicalize        | `affine.min` → `affine.apply(64)`，但 tile 类型仍是 `tensor<?x64xf32>` |
| N2b | tile + peel + canonicalize/cse                         | 4 个区域全静态，`tensor<?` 0 次；单趟尾循环被折叠                          |
| N3  | N2b 之后 promote 到 L1                                 | 12 个 `alloc_tensor` 全静态；bufferize 后 13 个 alloc，`memref<?` 0 次    |
| N4  | 已整除（128x64 * 64x192）上 `transform.loop.peel`    | silenceable error `failed to peel the last iteration`                       |
| N5  | 已 tile 的 IR 上跑 `-scf-for-loop-peeling`（默认 skip-partial=true） | M 尾块那一支**没被 peel**，残留 `affine.min` + `tensor<64x?xf32>`  |
| N6  | 同上但 `skip-partial=false` + canonicalize/cse       | 与 §4.2 的手写 peel 路线输出**逐字一致**（diff 只差函数名）            |

工具：`build/bin/mlir-opt`、`build/bin/FileCheck`（LLVM 24.0.0git，assertions on）。
`memory_space` 用整数 `1` = L1（`#spm.memory_space` 尚未实现，对 bufferize 行为无差别）。

---

## 附录 A：peel 逐步拆解

给不熟悉 peel 的读者。每一步都有一个**单独的文件**，文件里的 `func.func` 就是这一步的 **before**，
跑文件头的 RUN 行得到的就是 **after**（也正好是下一个文件的 body）：

```text
spm-mlir/test/pre-bufferize/peel-steps/
  peel-101.mlir            peel 本身在做什么（1-D 最小例子，不含 tiling）
  step1-tile.mlir          before: 未 tile 的 matmul        after: 带 affine.min 的两层循环
  step2-peel-inner.mlir    before: step1 的输出              after: 内层 N 循环被劈成 main + remainder
  step3-peel-outer.mlir    before: step2 的输出              after: 外层 M 循环也被劈开 -> 4 个区域
  step4-canonicalize.mlir  before: step3 的输出              after: 类型全静态（这一步没有 transform，纯 pass）
  step5-promote.mlir       before: step4 的输出              after: 12 个静态 L1 alloc_tensor
```

### A.1 peel 到底做了什么

机械定义（`LoopSpecialization.cpp:123-181` 的 `peelForLoop`）：

```text
输入：L = scf.for %i = lb to ub step s { body }

splitBound = ub - (ub - lb) mod s                     // :157-166，全静态时直接折成常量

改写成两个串联的循环：
  L_main      = scf.for %i = lb         to splitBound step s { body }   // :176-179 只改 ub
  L_remainder = scf.for %i = splitBound to ub         step s { body }   // :169-174 克隆一份
  L_remainder 的 iter_args init = L_main 的 results，L 的所有 user 改用 L_remainder 的 results

然后 rewriteAffineOpAfterPeeling（:183-207）在两份 body 里分别化简 affine.min/max：
  L_main      里：extent 恒等于整个 step        -> affine.apply () -> (s)
  L_remainder 里：extent 恒等于 ub - %i         -> affine.apply (d0) -> (-d0 + ub)
```

**注意它没做的事**：它不改任何 op 的类型。`tensor.extract_slice` 的 size 从 `affine.min`
换成了 `affine.apply`，但 result type 还是 `tensor<?x...>`。把类型变静态的是 canonicalize（A.5）。

`peel-101.mlir` 是这段话的 1-D 版本，去掉了 tiling 的干扰：

**before**（一个 `[0, 130) step 64` 的循环，`%n` 是本次迭代真正处理多少个元素：64, 64, 2）

```mlir
%r = scf.for %i = %c0 to %c130 step %c64 iter_args(%acc = %t) -> (tensor<130xf32>) {
  %n = affine.min affine_map<(d0) -> (-d0 + 130, 64)>(%i)
  %s = tensor.extract_slice %acc[%i] [%n] [1] : tensor<130xf32> to tensor<?xf32>
  %f = linalg.fill ins(%cst : f32) outs(%s : tensor<?xf32>) -> tensor<?xf32>
  %ins = tensor.insert_slice %f into %acc[%i] [%n] [1] : tensor<?xf32> into tensor<130xf32>
  scf.yield %ins : tensor<130xf32>
}
```

**after（只 peel）**：splitBound = 130 - 130 mod 64 = 128，一个循环变两个，`affine.min` 各自化简

```mlir
%0 = scf.for %arg1 = %c0 to %c128 step %c64 iter_args(%arg2 = %arg0) -> (tensor<130xf32>) {
  %2 = affine.apply affine_map<() -> (64)>()                 // <== 主循环：恒为 64
  %extracted_slice = tensor.extract_slice %arg2[%arg1] [%2] [1] : tensor<130xf32> to tensor<?xf32>
  ...
}
%1 = scf.for %arg1 = %c128 to %c130 step %c64 iter_args(%arg2 = %0) -> (tensor<130xf32>) {
  %2 = affine.apply affine_map<(d0) -> (-d0 + 130)>(%arg1)   // <== 尾循环：恒为 130 - %i
  %extracted_slice = tensor.extract_slice %arg2[%arg1] [%2] [1] : tensor<130xf32> to tensor<?xf32>
  ...
}
return %1
```

**after（peel + canonicalize + cse）**：常量折叠 → 类型静态；尾循环只有 1 趟，被整体内联

```mlir
%0 = scf.for %arg1 = %c0 to %c128 step %c64 iter_args(%arg2 = %arg0) -> (tensor<130xf32>) {
  %extracted_slice_0 = tensor.extract_slice %arg2[%arg1] [64] [1] : tensor<130xf32> to tensor<64xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%extracted_slice_0 : tensor<64xf32>) -> tensor<64xf32>
  ...
}
%extracted_slice = tensor.extract_slice %0[128] [2] [1] : tensor<130xf32> to tensor<2xf32>   // 尾块直线代码
%1 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<2xf32>) -> tensor<2xf32>
```

一句话：**peel 把「一个循环 + 一个 min」换成「两个循环 + 两个常量表达式」**，用代码体积换编译期已知的 extent。

### A.2 Step 1：tile（还没 peel）

before 是未 tile 的 `linalg.matmul`；after：

```mlir
#map  = affine_map<(d0) -> (-d0 + 130, 64)>       // M 方向 extent
#map1 = affine_map<(d0) -> (-d0 + 200, 64)>       // N 方向 extent
%0 = scf.for %arg3 = %c0 to %c130 step %c64 ... {          // M
  %1 = scf.for %arg5 = %c0 to %c200 step %c64 ... {        // N
    %2 = affine.min #map(%arg3)                            // 64 或 2
    %3 = affine.min #map1(%arg5)                           // 64 或 8
    ... tensor<?x64xf32> / tensor<64x?xf32> / tensor<?x?xf32> ...
```

`130 % 64 != 0`、`200 % 64 != 0`，所以 tile 的两个 extent 都不是常量 —— 这是全部问题的根。

### A.3 Step 2：peel 内层（N）循环

**只**动 N 循环。M 循环原样不动，所以结果是「1 个外层循环里装着 2 个内层循环」：

```mlir
scf.for %arg3 = %c0 to %c130 step %c64 {                   // M：没动，affine.min 还在
  %c192 = arith.constant 192 : index                       // 200 - 200 mod 64
  %1 = scf.for %arg5 = %c0   to %c192 step %c64 {          // N main
    %3 = affine.min #map(%arg3)                            // M extent：还是 min
    %4 = affine.apply #map1()                              // N extent：常量 64
    ...
  }
  %2 = scf.for %arg5 = %c192 to %c200 step %c64 iter_args(%arg6 = %1) {   // N remainder
    %3 = affine.min #map(%arg3)
    %4 = affine.apply #map2(%arg5)                         // N extent：200 - %arg5 = 8
    ...
  }
  scf.yield %2
}
```

注意 `iter_args(%arg6 = %1)`：remainder 接着 main 的结果往下算，这就是 A.1 里那条
「remainder 的 init = main 的 results」。

### A.4 Step 3：peel 外层（M）循环

外层 peel 会**克隆整个 body**，于是 step 2 造出来的两个内层循环各被复制一份：

```text
scf.for %arg3 = %c0   to %c128 step %c64 {     // M main
  scf.for %arg5 = %c0   to %c192 step %c64 { ... }   // 区域 A：main M x main N
  scf.for %arg5 = %c192 to %c200 step %c64 { ... }   // 区域 B：main M x tail N
}
scf.for %arg3 = %c128 to %c130 step %c64 {     // M remainder
  scf.for %arg5 = %c0   to %c192 step %c64 { ... }   // 区域 C：tail M x main N
  scf.for %arg5 = %c192 to %c200 step %c64 { ... }   // 区域 D：tail M x tail N
}
```

这一步之后 `affine.min` **全部消失**（4 个区域各自知道自己的 extent，全是 `affine.apply`），
但 tile 类型仍然是 `tensor<?x?xf32>`。

「2 个不整除的维度 → 4 个区域」就是在这里发生的，也是 §5.6 里 `2^d` 代价的来源。
顺序也是在这里体现的：**先内后外，两次调用就够**；反过来先 peel 外层，就要在两份拷贝里
各自重新找内层循环再 peel。

### A.5 Step 4：canonicalize（把类型变静态的那一步）

这一步没有 transform 脚本，就是 `-canonicalize -cse`。三件事同时发生：

```text
1. affine.apply 的操作数都是常量  ->  折成 arith.constant
2. extract_slice / insert_slice 的 size 变常量  ->  result type 折成静态，
   静态类型再传播进 linalg.matmul
3. trip count = 1 的循环被 promote（body 内联）  ->  区域 B/D 的 N 循环、区域 C/D 的 M 循环消失
```

前后对比（区域 A）：

```mlir
// before
%4 = affine.apply #map()
%5 = affine.apply #map()
%extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [%4, 64] [1, 1] : tensor<130x64xf32> to tensor<?x64xf32>
%6 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<?x64xf32>, tensor<64x?xf32>)
                   outs(%extracted_slice_1 : tensor<?x?xf32>) -> tensor<?x?xf32>

// after
%extracted_slice_6 = tensor.extract_slice %arg0[%arg3, 0] [64, 64] [1, 1] : tensor<130x64xf32> to tensor<64x64xf32>
%5 = linalg.matmul ins(%extracted_slice_6, %extracted_slice_7 : tensor<64x64xf32>, tensor<64x64xf32>)
                   outs(%extracted_slice_8 : tensor<64x64xf32>) -> tensor<64x64xf32>
```

### A.6 Step 5：promote（这一步才用到静态类型）

`promote_tensor` 用 **tile 的类型**当分配大小：

```mlir
// before
%es = tensor.extract_slice %arg0[%arg3, 0] [64, 64] [1, 1] : tensor<130x64xf32> to tensor<64x64xf32>

// after
%es = tensor.extract_slice %arg0[%arg3, 0] [64, 64] [1, 1] : tensor<130x64xf32> to tensor<64x64xf32>
%17 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<64x64xf32>          // <== L1 分配
%18 = bufferization.materialize_in_destination %es in %17 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
```

如果跳过 step 2–4 直接做这一步，同一行会变成
`bufferization.alloc_tensor(%2, %3) <{memory_space = 1}> : tensor<?x?xf32>`
（见 `tail-normalize-negative.mlir`）——**这就是尾块规范化存在的唯一理由**。

### A.7 每步的量化对照

| 步骤                  | `scf.for` 个数 | `affine.min` | `affine.apply` | 动态 tensor 类型出现次数 | `alloc_tensor` |
| --------------------- | ---------------- | -------------- | ---------------- | ------------------------ | ---------------- |
| 输入                  | 0                | 0              | 0                | 0                        | 0                |
| 1 tile                | 2                | 2              | 0                | 6                        | 0                |
| 2 peel 内层           | 3                | 2              | 2                | 12                       | 0                |
| 3 peel 外层           | 6                | **0**    | 8                | 24                       | 0                |
| 4 canonicalize        | 3                | 0              | **0**      | **0**              | 0                |
| 5 promote             | 3                | 0              | 0                | 0                        | **12（全静态）** |

读法：peel 阶段（2、3 行）IR 是在**变大**的——循环数从 2 涨到 6，动态类型出现次数从 6 涨到 24。
真正的收敛发生在第 4 行：canonicalize 把 6 个循环压回 3 个（1 趟的循环被内联），动态类型清零。
所以「peel 之后不跑 canonicalize 就去看结果」会得出「peel 把事情搞糟了」的错误结论。

### A.8 复现

```sh
cd spm-mlir/test/pre-bufferize/peel-steps
B=../../../../build/bin

$B/mlir-opt peel-101.mlir --transform-interpreter                    # peel 做了什么
$B/mlir-opt peel-101.mlir --transform-interpreter -canonicalize -cse # 再加 canonicalize

$B/mlir-opt step1-tile.mlir       --transform-interpreter -canonicalize -cse
$B/mlir-opt step2-peel-inner.mlir --transform-interpreter
$B/mlir-opt step3-peel-outer.mlir --transform-interpreter
$B/mlir-opt step4-canonicalize.mlir -canonicalize -cse
$B/mlir-opt step5-promote.mlir    --transform-interpreter

# 每个文件也都自带 FileCheck 断言：
$B/mlir-opt step3-peel-outer.mlir --transform-interpreter | $B/FileCheck step3-peel-outer.mlir
```

把 step N 的输出和 step N+1 的文件 body 对比，应该逐字相同（`step1` 的输出经
`-canonicalize -cse`，其余直接对比）。
