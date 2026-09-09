# HoistOnly（slotCount = 1）最小实例：期望 IR 与实现方案

> 配套文档：`SPM_PRE_BUFFERIZE_DESIGN.md` §5.7（slotCount = 1 的 hoist-only 路径）、
> §5.2（slot 放哪）、§3.6（循环内 allocation 来源全表 A2/A4/A6）、§11 第 8 步。
> 可复现的测试文件：`spm-mlir/test/pre-bufferize/hoist-only-*.mlir`。
> **本文的 before / after IR 都已经用 `build/bin/mlir-opt` 跑过 `-one-shot-bufferize` 验证**，
> 不是纸上形态；实现之前请先 review §5 的 after IR。

---

## 0. 这份文档要解决的问题

设计文档 §5.7 只有 4 行伪码。它是「最简单也最常用」的一条路径（§3.6 表里 A2/A3/A4/A6/A7
五种来源都走它），所以先把它做出来。本文做三件事：

1. 用上游 linalg 测试构造一个**最小但真实**的例子：一个 `scf.for` 嵌套，body 内有**多个** linalg op，
   每个都有自己的 L1 allocation（本例 5 个）；
2. 给出 **before / after 的期望 IR**，并用 bufferize 实测两者的差别；
3. 给出实现方案（pass 名、算法、上游 API、出口 gate、lit 清单）。

「hoist-only」的定义（§5.7）：把循环内的 allocation root 提到 `hoistPoint`（§5.2：最接近但位于
所有重复执行 scope 之外），每层 loop 加一个 iter_arg 承载它的版本，**不做 unroll、不做 prologue、
不做谓词**，代码体积不变。

---

## 1. 结论速查

| 问题                                                            | 结论                                                                                            | 依据                        |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------- |
| 本 pass 是否创建 `alloc_tensor`                                 | **不创建**：allocation root 由 `MaterializeTensorStorage`（设计文档 §3.2 第 6 步）产出，本 pass 只 move + 穿 iter_args，进出数量 1:1 | §7.0                        |
| 要不要完整的 read-after-write 冲突分析                          | 不要：只沿 DPS destination chain 做一次线性性检查（O(链长)）；漏判由 G4 兜底，只有 G7/G8 必须自己保证 | §7.0、§7.3.1                |
| 例子从哪来                                                      | 上游 `transform-tile-and-fuse.mlir` 的 fill→matmul→generic 结构，改成全静态 shape + `scf.for` | §2                          |
| before 里有几个循环内 allocation                                | 5 个 `alloc_tensor`（bias / A tile / B tile / acc / out tile），bufferize 后 5 个循环内 `memref.alloc` | 实测 H1                     |
| after 期望形态                                                  | 5 个 `alloc_tensor` 提到最外层 loop 之前，穿过**两层** `iter_args`，body 末尾 yield 各自最新版本 | §5                          |
| after 的实测结果                                                | 函数顶层 5 个 space 1 alloc，**循环内 0 alloc、0 额外 copy**（只剩 3 条 load + 1 条 writeback） | 实测 H2                     |
| 不加 iter_args 只把 alloc 挪出去行不行                          | **本例也能 bufferize 干净**（0 循环内 alloc）——但仍然采用 iter_args 形态，理由见本文 §6.1     | 实测 H3                     |
| 上游有没有现成的替代品                                          | `-buffer-loop-hoisting` 在 **memref 层**能做同样的事（实测结果一致）；我们仍在 tensor 层做，理由见本文 §6.2 | 实测 H4                     |
| 5 个 slot 里有几个真的只需要 1 份                               | 全部 5 个：每个都在迭代开始时被**整块覆写**（materialize 或 fill），满足 §5.7 第 4 条          | 本文 §4 表                       |
| 什么时候不能只用 1 份                                           | 内容需要跨迭代 in-flight（DMA/compute 重叠）→ 走 §5.3 rotation；跨迭代累加 → 是 accumulator，走 §3.4 | 设计文档 §4.6               |

---

## 2. 例子的构造

### 2.1 出处与改动

骨架取自 `mlir/test/Dialect/Linalg/transform-tile-and-fuse.mlir` 的第一个用例
（`linalg.fill` → `linalg.matmul` → `linalg.generic`，一个 fusion group）。三处按本项目的约束改：

| 上游                        | 本例                          | 原因                                       |
| --------------------------- | ----------------------------- | ------------------------------------------ |
| `tensor<?x?xf32>` 动态    | 全静态                        | 用户约束：输入一定全静态                   |
| `tile_using_forall`（并行）| `structured.fuse`（`scf.for`）| 设计文档 §3.7：必须 `scf.for`          |
| `generic` 是 relu-with-select | `generic` 是 bias-add + relu | 让 `ins` 里多一个真正需要搬进 L1 的 tile |

尺寸选成整除（128x256 * 256x64，tile `[32, 32]`），这样本文完全不涉及尾块——
尾块是另一份文档（`SPM_TAIL_NORMALIZATION_EXAMPLE.md`）的事。

### 2.2 输入 IR（第 0 步）

```mlir
func.func @fused_matmul_relu(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>,
                             %C: tensor<128xf32>, %D: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill {__producer__} ins(%cst : f32) outs(%D : tensor<128x64xf32>) -> tensor<128x64xf32>
  %1 = linalg.matmul {__producer__} ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                     outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
  %2 = linalg.generic {__root__, indexing_maps = [...], iterator_types = ["parallel", "parallel"]}
       ins(%C, %1 : tensor<128xf32>, tensor<128x64xf32>) outs(%D : tensor<128x64xf32>) {
  ^bb0(%b: f32, %x: f32, %o: f32):
    %m = arith.addf %x, %b : f32
    %r = arith.maximumf %m, %cst : f32
    linalg.yield %r : f32
  } -> tensor<128x64xf32>
  return %2 : tensor<128x64xf32>
}
```

### 2.3 怎么走到 before

用上游 transform op 模拟设计文档 §3.2 的第 3 步和第 6 步：

```mlir
%fused, %lm, %ln = transform.structured.fuse %root tile_sizes [32, 32] {apply_cleanup}   // TileForTier
...
transform.structured.promote_tensor to 1 %tile : !transform.any_value                     // MaterializeTensorStorage
```

`promote_tensor` 的实现（`LinalgTransformOps.cpp:443-495`）正是设计文档 §3.2 第 6 步要的语义：
`alloc_tensor` + 按 `mayBeRead` 决定是否插 `materialize_in_destination`。本例 5 个 promote：

```text
matmul operand 0（A tile）  -> alloc + materialize
matmul operand 1（B tile）  -> alloc + materialize
fill   outs   （acc）        -> alloc only（纯 destination，不读）
generic operand 0（bias tile）-> alloc + materialize
generic outs  （out tile）    -> alloc only
```

> **实测陷阱（必须写进实现）**：`transform.structured.match ops{["linalg.matmul"]}` 如果 scope 到整个
> payload，会**同时匹配到 tile-and-fuse 留下的那个已死的原始 matmul**（它要等 canonicalize 才被删）。
> 于是 promote 会作用在**原始的整块 DRAM tensor** 上，插出 `alloc_tensor : tensor<128x256xf32>` 并把
> 循环内 `extract_slice` 的源头改成它——静默多出 3 份整块 DRAM 大小的 L1 分配。
> 正确做法：把 match scope 到生成的 loop（本例 `%ln`），或者在 materialize 之前先 DCE。
> `MaterializeTensorStorage` 的实现里必须有等价的保护（只处理 tiling 产出的 tile value）。

---

## 3. Before IR（期望形态 = 实测输出）

`hoist-only-input.mlir` 经 `--transform-interpreter -canonicalize -cse` 之后：

```mlir
%0 = scf.for %arg4 = %c0 to %c128 step %c32 iter_args(%arg5 = %arg3) -> (tensor<128x64xf32>) {
  %1 = scf.for %arg6 = %c0 to %c64 step %c32 iter_args(%arg7 = %arg5) -> (tensor<128x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg2[%arg4] [32] [1] : tensor<128xf32> to tensor<32xf32>
    %2 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32xf32>              // slot 1
    %3 = bufferization.materialize_in_destination %extracted_slice in %2 : ...
    %extracted_slice_0 = tensor.extract_slice %arg0[%arg4, 0] [32, 256] [1, 1] : ... to tensor<32x256xf32>
    %4 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x256xf32>          // slot 2
    %5 = bufferization.materialize_in_destination %extracted_slice_0 in %4 : ...
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg6] [256, 32] [1, 1] : ... to tensor<256x32xf32>
    %6 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<256x32xf32>          // slot 3
    %7 = bufferization.materialize_in_destination %extracted_slice_1 in %6 : ...
    %8 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>           // slot 4
    %9  = linalg.fill ins(%cst : f32) outs(%8 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %10 = linalg.matmul ins(%5, %7 : tensor<32x256xf32>, tensor<256x32xf32>)
                        outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %11 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>          // slot 5
    %12 = linalg.generic {...} ins(%3, %10 : tensor<32xf32>, tensor<32x32xf32>)
                          outs(%11 : tensor<32x32xf32>) { ... } -> tensor<32x32xf32>
    %inserted_slice = tensor.insert_slice %12 into %arg7[%arg4, %arg6] [32, 32] [1, 1]
                    : tensor<32x32xf32> into tensor<128x64xf32>
    scf.yield %inserted_slice : tensor<128x64xf32>
  }
  scf.yield %1 : tensor<128x64xf32>
}
```

5 个 `alloc_tensor` 在**内层循环体内**，直接违反设计文档 §5.8 的 Q1 / §7.2 的 V10。

**实测 H1**（`-one-shot-bufferize`）：

```mlir
%0 = scf.for ... {
  %6 = scf.for ... {
    %alloc_0 = memref.alloc() alignment = 64 : memref<32xf32, 1>       // 循环内
    memref.copy %subview, %alloc_0
    %alloc_2 = memref.alloc() alignment = 64 : memref<32x256xf32, 1>   // 循环内
    memref.copy %subview_1, %alloc_2
    %alloc_4 = memref.alloc() alignment = 64 : memref<256x32xf32, 1>   // 循环内
    memref.copy %subview_3, %alloc_4
    %alloc_5 = memref.alloc() alignment = 64 : memref<32x32xf32, 1>    // 循环内
    linalg.fill ins(%cst) outs(%alloc_5)
    linalg.matmul ins(%alloc_2, %alloc_4) outs(%alloc_5)
    %alloc_6 = memref.alloc() alignment = 64 : memref<32x32xf32, 1>    // 循环内
    linalg.generic ins(%alloc_0, %alloc_5) outs(%alloc_6)
    memref.copy %alloc_6, %subview_7
    scf.yield %arg7
  }
}
```

循环内 **5 个 `memref.alloc`（space 1）**。这就是要消掉的东西。
（另外还有 1 个函数级 `memref.alloc() : memref<128x64xf32>` + copy，那是没开
`bufferize-function-boundaries` 时对被写的 func 参数做的 staging，与本文无关。）

---

## 4. 5 个 slot 的归类（决定它们各自要几份）

| slot | 类型                 | §3.6 归类                | 每迭代是否整块覆写           | slotCount | 路径      |
| ---- | -------------------- | ------------------------ | ---------------------------- | --------- | --------- |
| 1    | `tensor<32xf32>`   | A1 stream（bias tile）   | 是（`materialize`）        | 1（v0）   | HoistOnly |
| 2    | `tensor<32x256xf32>` | A1 stream（A tile）      | 是（`materialize`）        | 1（v0）   | HoistOnly |
| 3    | `tensor<256x32xf32>` | A1 stream（B tile）      | 是（`materialize`）        | 1（v0）   | HoistOnly |
| 4    | `tensor<32x32xf32>` | A4 scratch（matmul 中间结果）| 是（`fill` 打头）        | 1         | HoistOnly |
| 5    | `tensor<32x32xf32>` | A2 输出 staging          | 是（`generic` 全写 outs）  | 1         | HoistOnly |

三点说明：

- slot 1–3 是 stream，**真要做 DMA/compute 重叠时它们会升级到 slotCount = 2 走 §5.3**。
  本文刻意让 `PlanPipeline` 判定为不流水（或还没跑），这样 5 个都落在 HoistOnly 路径上——
  这正是设计文档 §11 第 8 步的 gate 想要的最小场景。
- slot 4 不是 accumulator：K 维（256）没有被切，`fill` 在每次迭代内部，acc 的生命周期完全在一次迭代内。
  如果 K 被切了，它就变成 §3.4 的 A3，`fill` 必须提到循环外，那时 iter_arg 承载的是**真实内容**而不只是版本。
- slot 5 的下游是 `tensor.insert_slice` 写回 DRAM。按设计文档 §7.1 的 V4，跨 placement 的搬运应该是
  `materialize_in_destination`；上游 tile-and-fuse 给的是 `insert_slice`。本文保留上游原样，
  这条规范化属于 `ResolveTensorConflicts` / `MaterializeTensorStorage` 的职责，不在 HoistOnly 范围内。

---

## 5. After IR（期望形态，**这是要 review 的部分**）

`hoist-only-after.mlir` 全文（已实测可解析、可 bufferize）：

```mlir
#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @fused_matmul_relu(%arg0: tensor<128x256xf32>, %arg1: tensor<256x64xf32>,
                             %arg2: tensor<128xf32>, %arg3: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32

  // ---- 1. 5 个 slot 提到 hoistPoint（= 最外层 loop 之前），各 1 份 ----
  %bias_l1 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32xf32>
  %a_l1    = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x256xf32>
  %b_l1    = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<256x32xf32>
  %acc_l1  = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>
  %out_l1  = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<32x32xf32>

  // ---- 2. 每层 loop 各加 5 个 iter_arg，位置固定（§5.4 的 R2）----
  %r:6 = scf.for %i = %c0 to %c128 step %c32
      iter_args(%out = %arg3, %sb = %bias_l1, %sa = %a_l1, %sw = %b_l1, %sacc = %acc_l1, %so = %out_l1)
      -> (tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>,
          tensor<32x32xf32>, tensor<32x32xf32>) {
    %rr:6 = scf.for %j = %c0 to %c64 step %c32
        iter_args(%out1 = %out, %sb1 = %sb, %sa1 = %sa, %sw1 = %sw, %sacc1 = %sacc, %so1 = %so)
        -> (tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>,
            tensor<32x32xf32>, tensor<32x32xf32>) {

      // ---- 3. body 内对 slot 的第一次写改成写 iter_arg（bbArg）----
      %cs = tensor.extract_slice %arg2[%i] [32] [1] : tensor<128xf32> to tensor<32xf32>
      %bias_v = bufferization.materialize_in_destination %cs in %sb1
              : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
      %as = tensor.extract_slice %arg0[%i, 0] [32, 256] [1, 1] : tensor<128x256xf32> to tensor<32x256xf32>
      %a_v = bufferization.materialize_in_destination %as in %sa1
           : (tensor<32x256xf32>, tensor<32x256xf32>) -> tensor<32x256xf32>
      %bs = tensor.extract_slice %arg1[0, %j] [256, 32] [1, 1] : tensor<256x64xf32> to tensor<256x32xf32>
      %b_v = bufferization.materialize_in_destination %bs in %sw1
           : (tensor<256x32xf32>, tensor<256x32xf32>) -> tensor<256x32xf32>

      %f  = linalg.fill ins(%cst : f32) outs(%sacc1 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %mm = linalg.matmul ins(%a_v, %b_v : tensor<32x256xf32>, tensor<256x32xf32>)
                          outs(%f : tensor<32x32xf32>) -> tensor<32x32xf32>
      %g  = linalg.generic {indexing_maps = [#map, #map1, #map1],
                            iterator_types = ["parallel", "parallel"]}
            ins(%bias_v, %mm : tensor<32xf32>, tensor<32x32xf32>)
            outs(%so1 : tensor<32x32xf32>) {
      ^bb0(%in: f32, %in_2: f32, %o: f32):
        %x = arith.addf %in_2, %in : f32
        %y = arith.maximumf %x, %cst : f32
        linalg.yield %y : f32
      } -> tensor<32x32xf32>

      %ins = tensor.insert_slice %g into %out1[%i, %j] [32, 32] [1, 1]
           : tensor<32x32xf32> into tensor<128x64xf32>

      // ---- 4. 每个 slot 的“最后一个版本”yield 回同一个位置 ----
      //      slot4 的最后版本是 %mm（fill -> matmul 链的末端），不是 %f
      scf.yield %ins, %bias_v, %a_v, %b_v, %mm, %g
        : tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>,
          tensor<32x32xf32>, tensor<32x32xf32>
    }
    scf.yield %rr#0, %rr#1, %rr#2, %rr#3, %rr#4, %rr#5
      : tensor<128x64xf32>, tensor<32xf32>, tensor<32x256xf32>, tensor<256x32xf32>,
        tensor<32x32xf32>, tensor<32x32xf32>
  }
  return %r#0 : tensor<128x64xf32>
}
```

四条改写规则，逐条对应 §5.7 的伪码：

| §5.7 步骤                       | 本例的落实                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------ |
| 1. alloc 移到 `hoistPoint(a)` | 5 个 `alloc_tensor` 移到最外层 `scf.for` 之前（两层 loop 都会重复执行它们） |
| 2. 每层 loop 加 iter_arg        | 外层 + 内层各加 5 个，顺序与 slot 编号一致                                     |
| 3. body 内第一次写改写 iter_arg | 3 个 `materialize` 的 dest 换成 `%sb1/%sa1/%sw1`；`fill`/`generic` 的 outs 换成 `%sacc1/%so1` |
| 3b. 终端版本 yield 回同一位置   | `%bias_v, %a_v, %b_v, %mm, %g`；注意 slot4 走的是 `fill → matmul` 两跳     |

**实测 H2**（`-one-shot-bufferize`）：

```mlir
%alloc   = memref.alloc() alignment = 64 : memref<32xf32, 1>       // 函数顶层
%alloc_0 = memref.alloc() alignment = 64 : memref<32x256xf32, 1>   // 函数顶层
%alloc_1 = memref.alloc() alignment = 64 : memref<256x32xf32, 1>   // 函数顶层
%alloc_2 = memref.alloc() alignment = 64 : memref<32x32xf32, 1>    // 函数顶层
%alloc_3 = memref.alloc() alignment = 64 : memref<32x32xf32, 1>    // 函数顶层
%4:6 = scf.for ... iter_args(%arg5 = %alloc_4, %arg6 = %alloc, %arg7 = %alloc_0, ...) {
  %6:6 = scf.for ... iter_args(%arg12 = %arg5, %arg13 = %arg6, %arg14 = %arg7, ...) {
    memref.copy %subview,   %arg13                 // DRAM -> L1 bias
    memref.copy %subview_5, %arg14                 // DRAM -> L1 A tile
    memref.copy %subview_6, %arg15                 // DRAM -> L1 B tile
    linalg.fill    ins(%cst)          outs(%arg16)
    linalg.matmul  ins(%arg14, %arg15) outs(%arg16)
    linalg.generic ins(%arg13, %arg16) outs(%arg17)
    memref.copy %arg17, %subview_7                 // L1 -> DRAM writeback
    scf.yield %arg12, %arg13, %arg14, %arg15, %arg16, %arg17
  }
  scf.yield %6#0, %6#1, %6#2, %6#3, %6#4, %6#5
}
```

| 判据                                     | 结果                                            |
| ---------------------------------------- | ----------------------------------------------- |
| 循环内 `memref.alloc`                  | **0**                                     |
| 函数级 space 1 `memref.alloc`          | 5（= slot 数，每个 1 份）                       |
| 循环内 `memref.copy`                   | 4，全部是预期搬运（3 条 load + 1 条 writeback） |
| 额外 copy（bufferize 为消解冲突插的）    | **0**                                     |
| yield 的 memref 与 iter bbArg 是否等价   | 是（否则会报 `not equivalent to ... iter bbArg`）|

对应设计文档 §5.8 的出口 gate：Q1 ✅、Q3 ✅（K = 1）、Q6 ✅、Q7 ✅。

---

## 6. 期望之外的三个实测发现

### 6.1 不加 iter_args 也能 bufferize 干净——但仍然采用 iter_args 形态

**实测 H3**（`hoist-only-after-no-iterargs.mlir`）：把 5 个 slot 提出去但**不**穿 iter_args，
body 直接用循环外的 `%bias_l1` 等作为 destination。结果与 H2 **完全一样**：
5 个函数级 alloc、循环内 0 alloc、0 额外 copy。

原因：`alloc_tensor` 的内容是未定义的，本例每个 slot 在同一次迭代里都是「先整块写、后读」，
没有任何人读写前的版本，所以 One-Shot 的分析不需要保留旧值，可以原地复用同一个 buffer。

**为什么还是要 iter_args**：

1. **语义**：一旦 slot 的内容需要跨迭代存活（accumulator，§3.4）或者循环之后还要读它的最后版本，
   不穿 iter_args 在 tensor 层就无法表达「同一份内容」——那时不是「分析不给力」，是程序意思变了。
2. **一致性**：§5.3 的 rotation（slotCount = 2）**必须**把 slot 放在 iter_args 里。
   HoistOnly 与 rotation 用同一套 slot 表示，`PlanPipeline` 把某个 slot 从 1 升到 2 时不需要重建结构。
3. **可验证性**：§7.3 的 V15（yield 到位置 j 的值与 iter_arg j 的 init 同 root）是机械可检查的
   「一个 slot 一份物理 buffer」证据。不穿 iter_args 时这个性质只能靠「bufferize 之后数 alloc」来事后确认。
4. **代价为零**：H2 与 H3 的 bufferize 输出等价，iter_args 只是多几个 SSA 值，不产生指令。

所以：**实现只产出 iter_args 形态**；H3 作为「即使 threading 有 bug 也不会静默多分配」的旁证保留。

### 6.2 上游 `-buffer-loop-hoisting` 在 memref 层能做同样的事

**实测 H4**：直接把 before IR bufferize（循环内 5 个 alloc），再跑上游
`-buffer-loop-hoisting`（`Bufferization/Transforms/Passes.td:302`），5 个 alloc 全部被提到函数入口块，
结果与 H2 一致。

那为什么不省事用它？三条，都不是风格问题：

1. **决策时机**：本项目的容量规划、`slotCount` 决策、spill 决策全部发生在 **bufferize 之前**
   （设计文档 §4.8 的单调迭代、下游文档 §5.2）。等到 memref 层再 hoist，容量核算已经基于错误的形态做完了。
2. **Q1/V10 是 pre-bufferize 契约**：`VerifyPreBufferizeForm` 必须能在进入 One-Shot 之前判定
   「repetitive region 内没有 L1/L2 allocation」。把这件事推给 bufferize 之后的 pass，
   等于放弃这条不变量（下游文档 §11 不变量 19：不允许静默生成循环内 `memref.alloc`）。
3. **rotation 无法在 memref 层表达**：slotCount = 2 的 2-way unroll + 静态残差绑定（§5.3）
   是 tensor 层的结构改写。HoistOnly 是它的退化情形，两条路必须在同一层做，否则 slot 表示会分裂。

另外 `-buffer-loop-hoisting` 只处理实现 `AllocationOpInterface` 且 `HoistingKind::Loop` 的 op
（`BufferOptimizations.cpp:79-83`），依赖也必须在循环外 dominate——动态大小的 alloc 提不出去。
这也说明它救不了「尾块没规范化」的情形（见 `SPM_TAIL_NORMALIZATION_EXAMPLE.md`）。

### 6.3 一个要避开的 mlir-opt 组合

`-one-shot-bufferize="bufferize-function-boundaries=true must-infer-memory-space=true"` 在本例上
**直接 abort**（栈顶在 `bufferization::getBufferType` 之下，Release build 没有打印消息）。
原因是 func 参数没有 memory space 可推断。

实现里的做法：`must-infer-memory-space=true` 只用于**不开** function boundaries 的自检
（设计文档 §7.1 的 V3 就是这么实测出来的），或者先给 func 参数标好 space 再开 boundaries。
本文的所有实测都用不带选项的 `-one-shot-bufferize`。

---

## 7. 实现方案

### 7.0 前置结论：本 pass 不创建 allocation

**HoistOnly 不负责创建 `alloc_tensor`。** allocation root 由 `MaterializeTensorStorage(T)` 产出
（设计文档 §3.2 第 6 步；参照实现 `LinalgTransformOps.cpp:460-491`：一个被 promote 的 value 对应一个
`alloc_tensor`，再按 `mayBeRead` 决定是否插 `materialize_in_destination`，最后 `replaceAllUsesExcept`）。
连设计文档 §3.6 表里 A5 的 `tensor.empty` 也是先由 `-empty-tensor-to-alloc-tensor` 转成 `alloc_tensor`、
**之后**才走 §5.7 hoist。本文 §3 的 before IR 里 5 个 `alloc_tensor` 已经就位。

所以本 pass 只做两件事：**把已有的 alloc 搬位置** + **穿 iter_args**。写成硬不变量：

```text
G6. #alloc_tensor(输出) == #alloc_tensor(输入)，且是同一批 op
    （只 move，不 create / erase / merge / split）
```

这条不变量决定了本 pass **不需要**完整的 read-after-write conflict 分析——「几份 alloc」的两个决定
都不在这里：

| 决定                                | 归属                         | 依据                                              |
| ----------------------------------- | ---------------------------- | ------------------------------------------------- |
| 一共几个 allocation root            | `MaterializeTensorStorage` | 按被 promote 的 destination 数，1:1               |
| 每个 root 几份（`slotCount` 1 / 2）  | `PlanPipeline`             | §4.6，依据是 stage 重叠（调度问题），不是冲突分析 |

本 pass 自己需要的分析只有一条，而且是退化版：**沿 DPS destination chain 走一遍，确认它线性、并取终端值**
（因为要知道 yield 谁）。复杂度 O(链长)，不需要 alias、不需要全局 liveness、不需要跨 root 的干涉图。

### 7.1 pass 的位置与名字

```text
spm-mlir/lib/SPM/Transforms/：  spm-hoist-loop-allocs         （= §5.7 HoistOnly，本文范围）
                               后续：spm-materialize-pipeline （= §5.3 rotation，同一个文件）
```

v0 先做成独立 pass 便于单测；`MaterializePipeline` 落地时把它作为 `slotCount == 1` 的分支调用
（设计文档 §5.6 的由内向外遍历里，每个 loop 先处理 rotation 组，再处理 hoist-only 组）。

### 7.2 算法

```text
HoistLoopAllocs(func):
  // 由内向外遍历（与 §5.6 一致），保证内层 slot 的 hoistPoint 计算时外层 loop 还是完整的 op
  for L in loops(func, innermost -> outermost):
      A = { a : a is bufferization.alloc_tensor in L.body,
                a.memory_space in {L1, L2},
                slotCount(a) == 1 }                      // v0：没有 PlanPipeline 时全部当 1
      if A empty: continue

      for a in A:
          // 1. 前置检查（任一条不满足就报错退出，清单见 §7.2.2）
          require a.getType().hasStaticShape() 且 a 无 dynamic operand      // V9
          P = hoistPoint(a)                                                // §5.2 + §7.2.1 的白名单
          // 2. 版本链：从 a 出发，沿 DPS destination 一路走到最后一个写
          chain = versionChain(a)      // a -> materialize/fill/matmul/... -> last
          require chain 每一跳都是 DestinationStyleOpInterface 且 dest 是上一跳的 result
          require a 与 chain 上每个中间 version 都只有一个 use，且该 use 是 DPS-init use
          if 首次写不是整块覆写(chain): warn "疑似 A3 accumulator"          // 不报错，见 §7.2.3
      // 3. 一次性把 A 里所有 alloc 移到 P 之前，然后自内向外给每层 loop 加 iter_arg
      move all a in A to just before P
      for M in loops(from L up to P, inner -> outer):
          M = M.replaceWithAdditionalYields(
                 rewriter,
                 /*newInitOperands=*/ A 的当前版本值,
                 /*replaceInitOperandUsesInLoop=*/ true,      // 关键：body 内对 a 的 use 自动换成 bbArg
                 /*newYieldValuesFn=*/ [](bbArgs) { return versionChain(bbArg).back() for each })
```

#### 7.2.1 `hoistPoint` 必须走白名单，不能直接反复外推（安全条件）

`bufferization::getEnclosingRepetitiveRegion`（`BufferizableOpInterface.h:564`）**把 `scf.forall`
也算成 repetitive region**（`SCF/Transforms/BufferizableOpInterfaceImpl.cpp:106-113`，
`ForallOpInterface::isRepetitiveRegion` 只看 step 数）。盲目拿它反复外推 → alloc 被提到 `forall`
之外 → **所有并发迭代共享同一份 L1 buffer = data race**。而且这个错误是**静默**的：

- bufferize 不报错（形态完全合法）；
- §7.3 的 G4 也数不出来（循环内确实 0 alloc）。

设计文档 §3.7 只禁止了「本层 tiling 产出 `scf.forall`」，并不能假设 hoist 路径上没有别人产的
`scf.forall`。所以实现必须显式白名单：

```text
hoistPoint(a):
  last = null; cur = a 的 parent op
  while cur 是 scf.for 或 scf.if:        // 只跨这两种
      last = cur; cur = cur 的 parent op
  若 cur 是 scf.forall / scf.while / 其他 repetitive region owner：停在它里面，
    此时 a 仍在 repetitive region 内（G1 不满足）→ 报错，绝不静默外推
  return last 之前的插入点
```

`outermostRepetitiveAncestor` 的语义仍然是 §5.2 那一条，只是实现不能偷懒。v0 若采用 static planner
的简化版（一路提到 func entry block）也一样要过白名单检查，并保留 `hoistPoint` 的接口，
别把 entry block 写死。

#### 7.2.2 两种 def-use 分叉，性质完全不同

| 分叉形式                                                            | 例子                                                              | 处理                                                                                                                                                              |
| ------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **读分叉**：同一个 version 被多个 op 当 `ins`                     | `%v` 同时喂 `matmul` 和 `generic`                             | **无害**：一份 alloc、一条链、终端就是 `%v`。这正是 `promote_tensor` 的 `replaceAllUsesExcept` 天然产出的形态                                                 |
| **destination 分叉**：同一个 version 被两个 DPS op 当 `outs`      | `%f = fill outs(%a)` 与 `%g = generic outs(%a)`                | **tensor 程序本身就需要两份 buffer**，不是「hoist 变复杂了」。本 pass 报错退出；真正的修法在上游：`MaterializeTensorStorage` 应按 **destination use** promote，而不是按 value promote |
| **覆写后的迟到 read**：中间 version 被下一跳覆写之后仍被读          | `%f = fill outs(%a)`、`%mm = matmul outs(%f)`，之后再读 `%f` | 与 destination 分叉一起，按「中间 version 的 use 数不等于 1」挡掉。保守（顺序上位于写之前的 read 其实合法），但覆盖 tiling 能产出的所有形态                       |

> 输入例子（§2.2）里 `%D` 本来就同时是 `linalg.fill` 和 `linalg.generic` 的 `outs`——只是
> tile-and-fuse 恰好给了两个不同的 `extract_slice`，才没有在 tile 层面撞上 destination 分叉。
> 换一个 fusion 决策就会撞上，所以这条检查不是理论洁癖。

#### 7.2.3 「整块覆写」是诊断，不是正确性 gate

设计文档 §5.7 第 4 条要求「a 在每次迭代开始时都被完整覆写」。**这条不影响正确性**：`alloc_tensor`
的内容是 undefined，读它是 UB；hoist + iter_args threading 把「上一迭代的残留」穿进来，是对 UB 的
合法 refine——即使 slot 不是整块覆写，程序含义也没变。

它的真实作用是**探测上游的分类错误**：非整块覆写的 slot 大概率是设计文档 §3.4 的 A3 accumulator，
`fill` 应该被提到循环外、iter_arg 承载的应该是真实内容。所以实现里它是 **warning（或 opt-in error）**，
不是 `assert`——否则会在完全合法的 IR 上硬失败。

#### 7.2.4 其余要点

- **`replaceInitOperandUsesInLoop = true`** 正好实现 §5.7 的第 3 步：把 body 内对 alloc 的所有 use
  换成新加的 region iter_arg，不需要自己做 RAUW
  （`LoopLikeInterface.td:229-238`，`scf::ForOp` 的实现在 `SCF.cpp:627`）。
- `newYieldValuesFn` 里返回的是 `versionChain` 的**终端**值（本例 slot4 是 `%mm` 而不是 `%f`）。
  版本链的走法：`用户是 DestinationStyleOpInterface 且该 operand 属于 getDpsInits()` → 取对应 result → 重复。
  `bufferization.materialize_in_destination` 也实现 `DestinationStyleOpInterface`
  （`BufferizationOps.td`），所以 copy 和 compute 用同一套代码走链。
- 多层时**先给内层加、再给外层加**：外层 `newInitOperands` 用的是外层 bbArg，内层 loop 的 init 换成它。
  实现上更简单的写法是自内向外逐层调用 `replaceWithAdditionalYields`，每层的 init 用上一层暴露出来的值。

### 7.3 出口 gate（写成 pass 的 verifier + lit）

```text
G1. func 内不存在位于 repetitive region 里的、memory_space 为 L1/L2 的 alloc_tensor     （Q1 / V10）
G2. 每个被提升的 slot：每层 loop 的 iter_args 里位置固定，yield 值与 init 同 root      （Q7 / V15）
G3. 每个 alloc_tensor 的类型静态                                                        （V9）
G4. bufferize 之后：循环内 memref.alloc 数 = 0，函数级 space 1 alloc 数 = slot 数        （实测 H2）
G5. bufferize 之后额外 copy 数 = 0（只剩预期的 load/writeback）                          （实测 H2）
G6. #alloc_tensor(输出) == #alloc_tensor(输入)，且是同一批 op（只 move）                （§7.0）
G7. yield 回位置 j 的值是该 root 版本链的**终端** DPS result（不只是「同 root」）        （§7.2 第 2 步）
G8. 没有任何 alloc 被提到 scf.forall / scf.while 之外                                   （§7.2.1）
```

G4/G5 直接用现成的 lit 文件当回归：实现完成后，`hoist-only-before.mlir` 经
`spm-opt -spm-hoist-loop-allocs` 应当逐字产出 `hoist-only-after.mlir` 的形态，
再经 `mlir-opt -one-shot-bufferize` 得到 H2 的输出。

#### 7.3.1 分析漏判的失败模式（为什么可以只做最干净的链）

只实现「最简单干净的 def-use chain」之所以安全，是因为漏判基本都是**响的**：

| 漏判                                                                             | 后果                                                                                                             | 谁抓住                        |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| destination 分叉 / 覆写后的迟到 read，且冲突那一跳是 linalg DPS op               | One-Shot 的 `insertTensorCopies` 插 copy + alloc，循环内 alloc 回来                                            | **G4**                  |
| 同上，但冲突那一跳是 `materialize_in_destination`                              | **bufferize 直接失败**——op 的语义就是「原内容之后还要读就报错」（`BufferizationOps.td:212-216`）        | bufferize 自己，更早          |
| alloc 被提到 `scf.forall` 之外                                                 | 静默 data race（bufferize 合法，「循环内 0 alloc」也照样成立）                                                   | 只有 **G8**，不能省     |
| yield 了非终端 version                                                           | 静默：`%f` 与 `%mm` 是同一 buffer 的 alias，G2 的「同 root」挡不住                                           | 只有 **G7**，不能省     |

所以：G4/G5 是「保守分析退化」的兜底网（配合设计文档 §7.4 的 analysis-only 自检），
G7/G8 是必须由 pass 自己保证的两条，其余复杂场景可以先不做。

### 7.4 lit 清单（实现时补齐）

| 测试                          | 内容                                                                    |
| ----------------------------- | ----------------------------------------------------------------------- |
| 已有 `hoist-only-input.mlir`| 上游 linalg → tiling+fuse → promote，产出 before（含 match scope 陷阱的说明）|
| 已有 `hoist-only-before.mlir`| before 的 bufferize 基线（循环内 5 alloc）+ `-buffer-loop-hoisting` 对照 |
| 已有 `hoist-only-after.mlir`| after 的 bufferize 基线（循环内 0 alloc）                              |
| 已有 `hoist-only-after-no-iterargs.mlir` | 6.1 的对照形态                                             |
| 待加 `hoist-basic.mlir`     | 单层 loop、1 个 slot：pass 的最小正例                                   |
| 待加 `hoist-nested.mlir`    | 两层 loop：slot 必须提到**最外层**之外、穿两层 iter_args          |
| 待加 `hoist-multi-op.mlir`  | 本文的 5-slot 例子：`spm-opt` 输出与 after IR 逐字比对                |
| 待加 `hoist-partial-write.mlir` | slot 不是整块覆写 → 期望 **warning**（不是报错），提示按 §3.4 accumulator 处理 |
| 待加 `hoist-read-fork.mlir`   | 一个 slot 被多个 `ins` 读（读分叉）→ 正例，仍然只提出 1 份              |
| 待加 `hoist-dest-fork.mlir`   | 同一 version 被两个 DPS op 当 `outs` → 期望**报错**（§7.2.2）           |
| 待加 `hoist-late-read.mlir`   | 中间 version 被覆写后仍被读 → 期望**报错**（§7.2.2）                    |
| 待加 `hoist-forall.mlir`      | hoist 路径上有 `scf.forall` → 期望**报错**，绝不静默提出去（§7.2.1）    |
| 待加 `hoist-dynamic.mlir`   | 动态类型 alloc_tensor → 期望报错（V9 应该在更早的 pass 拦住）        |
| 待加 `hoist-space0.mlir`    | 没有 memory_space 或 space 0 的 alloc_tensor → 不动它                |

### 7.5 不在本 pass 范围内

- 尾块规范化（另一份文档），本 pass 假设输入已全静态；
- `slotCount = 2` 的 rotation（§5.3）；
- `tensor.empty` 的消除（§3.6.1，必须在更早的位置跑 `-eliminate-empty-tensors`）；
- 跨 placement 边从 `insert_slice` 规范化成 `materialize_in_destination`（§4 第 3 点）；
- alloc 数量的最小化（同 shape、生命周期不重叠的两个 slot 共用一块物理内存）——属于 arena planner
  （下游文档 §9），那里 memref liveness 是精确的；本 pass 一律 1:1，见 §7.0 的 G6；
- 冗余 promote 的消除（嵌套 `extract_slice`：唯一 consumer 是同层另一次 promote）——属于
  `MaterializeTensorStorage` 的 promote 决策 + §3.6.1 规定的 `-eliminate-empty-tensors` 次序；
- slot 的物理连续放置（`spm.slot_group`，属于 arena planner）。

### 7.6 已知保守性（记在这里，v0 不修）

hoist 把 slot 的 live range 从「一次迭代内」拉长到「整个循环嵌套」。于是两个同 shape、在一次迭代内
生命周期**不重叠**的 scratch slot，在 arena planner 眼里变成了全程并存，无法共用一块物理内存——
要恢复这个复用，planner 需要 per-iteration liveness ＋「读前整块覆写」的判定（正是 §7.2.3 那条事实）。

后果：v0 的容量核算会**高估**峰值，可能触发一次本不必要的 spill。这是保守，不是错误
（设计文档 §4.8 的单调迭代吃得下高估），但已记入设计文档 §10 的 v0 边界表。

---

## 8. 复现

```sh
cd spm-mlir/test/pre-bufferize
B=../../../build/bin

# 第 0 步：上游 linalg -> tiling+fuse -> promote，得到 before
$B/mlir-opt hoist-only-input.mlir --transform-interpreter -canonicalize -cse \
  | $B/FileCheck hoist-only-input.mlir

# H1：before 直接 bufferize -> 循环内 5 个 space 1 alloc
$B/mlir-opt hoist-only-before.mlir -one-shot-bufferize | $B/FileCheck hoist-only-before.mlir
# H4：memref 层的上游 pass 也能救（对照用）
$B/mlir-opt hoist-only-before.mlir -one-shot-bufferize -buffer-loop-hoisting \
  | $B/FileCheck hoist-only-before.mlir --check-prefix=HOISTED

# H2：after（iter_args 形态）-> 循环内 0 alloc、0 额外 copy
$B/mlir-opt hoist-only-after.mlir -one-shot-bufferize | $B/FileCheck hoist-only-after.mlir

# H3：after（不穿 iter_args）-> 本例同样干净
$B/mlir-opt hoist-only-after-no-iterargs.mlir -one-shot-bufferize \
  | $B/FileCheck hoist-only-after-no-iterargs.mlir
```

### 实验记录

| #   | 内容                                                     | 结果                                                                      |
| --- | -------------------------------------------------------- | ------------------------------------------------------------------------- |
| H0  | 上游 fill→matmul→generic + `structured.fuse [32,32]` + 5 个 `promote_tensor` | before IR：内层循环体内 5 个 `alloc_tensor`（space 1）           |
| H0b | 同上，但 match 未 scope 到 loop                          | 额外 promote 了 3 个**整块 DRAM tensor**（死掉的原始 producer）→ 必须 scope 或先 DCE |
| H1  | before + `-one-shot-bufferize`                         | 循环内 5 个 `memref.alloc(space 1)`；总 6 个 alloc / 5 条 copy         |
| H2  | after（iter_args）+ `-one-shot-bufferize`              | 函数级 5 个 space 1 alloc、**循环内 0 alloc**、0 额外 copy         |
| H3  | after（无 iter_args）+ `-one-shot-bufferize`           | 与 H2 等价（0 循环内 alloc）→ threading 不是 bufferize 的硬需求，但仍采用 |
| H4  | before bufferize 之后再 `-buffer-loop-hoisting`        | 5 个 alloc 被提到函数入口，与 H2 一致 → 但决策时机不对，见 §6.2         |
| H5  | `bufferize-function-boundaries=true must-infer-memory-space=true` | mlir-opt **abort**（func 参数无法推断 space）→ 别用这个组合   |

工具：`build/bin/mlir-opt`、`build/bin/FileCheck`（LLVM 24.0.0git，assertions on）。
`memory_space` 用整数 `1` = L1。
