# SPM SRAM 静态分配 + 三级溢出（Spilling）设计

本文承接 [NPU_SRAM_ALLOCATION_PORTING.md](NPU_SRAM_ALLOCATION_PORTING.md)（Triton 图着色分配算法的可移植部分）和
[NPU_LOAD_STORE_MEMDESC_PORTING.md](NPU_LOAD_STORE_MEMDESC_PORTING.md)（memdesc 类型 + load/store op 设计），
是原创设计部分——两篇 porting 文档都明确指出 "spill-to-DRAM 是 Triton 原设计里没有、需要自己另外设计的部分"。

设计目标：在 `spm-mlir` 已有的 `!spm.l1mem`/`!spm.l2mem` 两级 SRAM 描述符之上，加一套完整的生命周期 op
（alloc/dealloc/load/store/**spill/fill**），并设计一个把"图着色静态分配"和"容量不够时溢出到下一级存储"
结合起来的算法，覆盖 **L1 → L2 → DRAM** 三级级联溢出。

---

## 0. 与 one-shot-bufferize 的衔接：为什么别名分析可以很"薄"

one-shot-bufferize 阶段一的核心不变量（见 `one-shot-bufferize-outline.md`）是：分析阶段结束后，
"此后任何机械式的结构化改写都不会引入 RAW 冲突"。当阶段二把这份已经过冲突消解的 tensor IR
机械改写成 `spm.l1_alloc` + `l1_load`/`l1_store`（DPS 类 op 直接复用 `outs` 对应 buffer）时，
**每个 `alloc` 出发的 def-use chain 内部天然无冲突**——这正是用户观察到的现象，也是本设计能够
直接照搬 Triton `AliasInfo`（见 `NPU_SRAM_ALLOCATION_PORTING.md` §3）那套"链内无需判断，只需
追踪谁和谁共享物理存储"规则的前提。

于是职责划分很清楚：

- **bufferize 阶段一**：保证单条 def-use chain 内部（同一个 alloc 的所有 view/传递）不会有 RAW 冲突。
- **本文的 SRAM 分配器**：保证**不同 alloc 之间**（跨 chain）不会被分配到重叠的物理地址/时间段，
  以及当所有 chain 的总占用超过某一级存储容量时，如何把部分数据挪到下一级。

两者互不越界——分配器的 alias 分析不需要重新证明"in-place 安全"，只需要判断"这个 SSA 值是不是
某个 alloc/fill 的别名"。

---

## 1. Buffer 全生命周期 op 一览

| 阶段 | Op | 状态 | 语义 |
|---|---|---|---|
| 分配 | `spm.l1_alloc` / `spm.l2_alloc` / `spm.dram_alloc` | **新增** | 产生一块新的 L1/L2/DRAM buffer，此时不带地址；三层同构 |
| 释放提示 | `spm.l1_dealloc` / `spm.l2_dealloc` / `spm.dram_dealloc` | **新增** | 纯提示/校验用，**不参与生命周期计算**（照抄 Triton `local_dealloc` 的语义划分，见 porting doc §2） |
| 计算域搬运 | `spm.l1_load` / `spm.l1_store` / `spm.l2_load` / `spm.l2_store` | 已有 | buffer ↔ tensor（计算域） |
| **L1↔L2 溢出** | `spm.l1_spill` / `spm.l1_fill` | **新增** | 纯拷贝、无返回值，目标 buffer 必须由调用方预先 `alloc` 好（见 §1.2，修正版） |
| **L2↔DRAM 溢出** | `spm.l2_spill` / `spm.l2_fill` | **新增** | 同上，目标是 `!spm.dram` |

`spill`/`fill` 的目标层级在 op 里是**固定的**（`l1_spill` 恒定去 L2，`l2_spill` 恒定去 DRAM），
不做成一个通用的"任意层到任意层"op——理由和现有 L1/L2 各自独立成对 op 的既有设计一致
（见 `spm_mlir_scaffold` 里"保持 L1/L2 拆分一致"的既定原则），且 v0 只做**逐级级联**，
不支持跳级（比如 L1 满了直接丢 DRAM）。跳级留作后续优化，不在 v0 范围内。

### 1.1 `!spm.dram` 类型

```tablegen
def SPM_DRAMType : SPM_MemDescTypeBase<"DRAM", "dram"> {
  // 字段完全镜像 l1mem/l2mem：shape, elementType, mutableMemory, allocShape
}
```

和 `l1mem`/`l2mem` 保持同一套字段、同一个 `ShapedTypeInterface` 基类，是第三个独立类型而非
"memref + address space attr"——延续 `spm-mlir-scaffold` 里"三种独立类型而非一种类型+内存空间属性"
的既有选择，一致性优先。

### 1.2 spill/fill 的 op 签名：目标必须是显式预分配的 buffer

**这一节是对上一版的修正**。上一版里 `%spilled = spm.l1_spill %buf : ... -> !spm.l2mem<...>` 让 spill
隐式地在目标层"生出"一块新 buffer 作为返回值。这是一个真实的设计错误：每一层的分配算法都是靠"扫描
这一层所有 `*_alloc` op"来发现自己的分配对象的（porting doc §7 骨架第 2 步就是"扫 `local_alloc`"）。
如果 spill 的目标不是一个真正的 `spm.l2_alloc`/`spm.dram_alloc` 产生的值，L2/DRAM 自己那一遍
`AllocateTier`（§3）根本看不到它，也就没有任何 op 能把算出来的 offset 属性钉上去——L1 分配完、
轮到 L2/DRAM 处理这些"空降"buffer 时必然出问题。

修正：`spill`/`fill` 一律**不返回值**，签名和已有的 `l1_store`（写入一个**预先存在**的目标 memdesc，
而不是自己产生一个）完全同构——目标必须由调用方先用该层自己的 `*_alloc` 显式分配出来：

```mlir
// L1 -> L2：先在 L2 上显式 alloc，spill 只管拷贝，不生成新值
%l2tmp = spm.l2_alloc : !spm.l2mem<64x64xf32>
spm.l1_spill %buf, %l2tmp : !spm.l1mem<64x64xf32>, !spm.l2mem<64x64xf32>

// ...稍后需要用回来时，先在 L1 上显式 alloc，fill 只管拷贝
%buf2 = spm.l1_alloc : !spm.l1mem<64x64xf32>
spm.l1_fill %l2tmp, %buf2 : !spm.l2mem<64x64xf32>, !spm.l1mem<64x64xf32>
// 后续所有原本用 %buf 的地方，从这里开始改用 %buf2

// L2 -> DRAM 同构
%dramtmp = spm.dram_alloc : !spm.dram<64x64xf32>
spm.l2_spill %buf2, %dramtmp : !spm.l2mem<64x64xf32>, !spm.dram<64x64xf32>
%buf3 = spm.l2_alloc : !spm.l2mem<64x64xf32>
spm.l2_fill %dramtmp, %buf3 : !spm.dram<64x64xf32>, !spm.l2mem<64x64xf32>
```

（这也顺带补齐了一个 op：`spm.dram_alloc`/`spm.dram_dealloc`，镜像 `l1_alloc`/`l2_alloc`——既然目标
必须显式分配，DRAM 侧自然也需要自己的 alloc/dealloc，不再是"隐式打底"。上一版 §9 里"DRAM 不需要
alloc op"的说法一并撤销。）

关键设计点（相比上一版的修正）：

- **目标必须先显式 `alloc`**，spill/fill 本身是纯拷贝、无返回值的 void op——和 `l1_store` 同一个模子
  刻出来的（"写入一个已存在的目的 memdesc"），不是一个新发明的模式。
- **每一层的分配算法完全不需要知道"这个 alloc 是不是溢出产生的"**——它看到的就是一个普普通通的
  `*_alloc` op，用统一的"扫 alloc op"方式发现，用同一套图着色算法分配 offset。溢出驱动逻辑（§3）
  的职责被压缩成纯粹的"决定何时、把谁的 alloc+copy 插到哪里"，不需要给分配算法开任何后门。
- **`spill` 是其源操作数的终结使用**：spill 之后原来那个源 buffer 的物理存储被回收，后续使用非法
  （和 dealloc 之后使用非法是同一类规则）。**这一点甚至不需要专门的 alias 规则**——只要后续所有真实
  使用都已经被重写到新分配的 buffer 上，`mlir::Liveness` 会自动把源 buffer 的活跃区间算到 spill 这一点
  为止，不需要人为标注"这是终结点"（这正是 porting doc §2 强调的"生死由 use-def 决定，不是扫描某个
  特殊 op"这条原则，本来就该直接套用在 spill 上——上一版单独发明"sink/seed"规则是想多了，见 §2）。
- **同样的"预分配目标 + 拷贝"模式，不是溢出机制专属**——kernel 输入数据从 DRAM 经 L2 到 L1 的**正常**
  加载路径（完全不涉及容量溢出）本来就需要同一种"每一级显式 alloc + 显式拷贝"结构：DRAM 侧的值可能
  就是函数参数（不需要额外 `dram_alloc`），但 L2/L1 侧要把数据首次搬进来时，一样得先 `l2_alloc`/
  `l1_alloc` 再拷贝进去。区别只在于**谁负责插入这些 op**——正常加载路径的 alloc+copy 由更早的
  "tensor→SRAM lowering" pass 插入（类比 bufferize 阶段二），溢出场景的 alloc+copy 由 §3 的溢出驱动
  循环插入。**对分配算法本身而言，这两种来源完全无法区分，也不需要区分**——这正是这个设计想要的
  compositional 效果，也是本节要回应的"三层都要显式创建 alloc/load"这个问题的答案：不是只有溢出才
  显式建 alloc，是任何跨层数据搬运都必须显式建 alloc，溢出只是复用了同一条规则。

---

### 1.3 完整示例：分配前 → 分配后（够用）→ 分配后（触发 spill）

用一个贯穿始终的小例子把 §1-§4 串起来。所有 buffer 都是 `64x64xf32` = 16KB，记一个"单位" = 16KB。

**分配前（lowering 直接产出的 IR，所有 `l1_alloc` 都还没有 `offset`）：**

```mlir
func.func @accum_across_burst(%in_a: tensor<64x64xf32>, %in_b: tensor<64x64xf32>,
                               %in_d: tensor<64x64xf32>, %in_e: tensor<64x64xf32>)
    -> (tensor<64x64xf32>, tensor<64x64xf32>) {
  // ---- 阶段一：c = a @ b（a、b 用完即死；c 要活到阶段三）----
  %a = spm.l1_alloc : !spm.l1mem<64x64xf32>
  spm.l1_store %in_a, %a : tensor<64x64xf32> -> !spm.l1mem<64x64xf32>
  %b = spm.l1_alloc : !spm.l1mem<64x64xf32>
  spm.l1_store %in_b, %b : tensor<64x64xf32> -> !spm.l1mem<64x64xf32>
  %c = spm.l1_alloc : !spm.l1mem<64x64xf32>
  linalg.matmul ins(%a, %b : !spm.l1mem<64x64xf32>, !spm.l1mem<64x64xf32>)
                outs(%c : !spm.l1mem<64x64xf32>)
  spm.l1_dealloc %a : !spm.l1mem<64x64xf32>
  spm.l1_dealloc %b : !spm.l1mem<64x64xf32>

  // ---- 阶段二：一段完全不碰 c 的"突发"计算，f = d @ e ----
  %d = spm.l1_alloc : !spm.l1mem<64x64xf32>
  spm.l1_store %in_d, %d : tensor<64x64xf32> -> !spm.l1mem<64x64xf32>
  %e = spm.l1_alloc : !spm.l1mem<64x64xf32>
  spm.l1_store %in_e, %e : tensor<64x64xf32> -> !spm.l1mem<64x64xf32>
  %f = spm.l1_alloc : !spm.l1mem<64x64xf32>
  linalg.matmul ins(%d, %e : !spm.l1mem<64x64xf32>, !spm.l1mem<64x64xf32>)
                outs(%f : !spm.l1mem<64x64xf32>)
  spm.l1_dealloc %d : !spm.l1mem<64x64xf32>
  spm.l1_dealloc %e : !spm.l1mem<64x64xf32>
  %f_result = spm.l1_load %f : !spm.l1mem<64x64xf32> -> tensor<64x64xf32>
  spm.l1_dealloc %f : !spm.l1mem<64x64xf32>

  // ---- 阶段三：终于再用到 c ----
  %c_result = spm.l1_load %c : !spm.l1mem<64x64xf32> -> tensor<64x64xf32>
  spm.l1_dealloc %c : !spm.l1mem<64x64xf32>
  return %f_result, %c_result : tensor<64x64xf32>, tensor<64x64xf32>
}
```

真实（活跃区间意义上的，不看 dealloc hint）并发占用随程序推进的变化：

| 程序点 | 事件 | 此刻活着的 buffer | 占用 |
|---|---|---|---|
| t1 | `alloc a` | a | 16K |
| t3 | `alloc b` | a, b | 32K |
| t5 | `alloc c` | a, b, c | 48K |
| t6 | `matmul1`（读 a,b，写 c） | a, b, c | 48K —— **peak①** |
| t6 之后 | a、b 的最后一次使用已经发生 | c | 16K |
| t9 | `alloc d` | c, d | 32K |
| t11 | `alloc e` | c, d, e | 48K |
| t13 | `alloc f` | c, d, e, f | **64K** |
| t14 | `matmul2`（读 d,e，写 f；**c 全程没被用到，只是恰好还活着**） | c, d, e, f | 64K —— **peak②** |
| t14 之后 | d、e 的最后一次使用已经发生 | c, f | 32K |
| t17 | `load f`（f 最后一次使用） | c | 16K |
| t19 | `load c`（c 最后一次使用） | c | 16K |

注意 peak② 的关键点：`matmul2` 本身只需要 d/e/f 三个 buffer 同时在场（这本身就正好等于一个"单个 op
需要多少 buffer 同时在场"的下限），c 只是"活着但这段时间完全用不上"，纯粹因为它的活跃区间跨过了这段
突发计算——**这正是可以被安全 spill 掉的那种 buffer**，和"某个 op 自己的操作数加起来就超容量"（无法
通过 spill 解决，见 §7 硬错误情形）是完全不同的两回事。

#### 情况一：容量足够（capacity = 64K = 4 单位）——不需要 spill

peak② = 64K 恰好等于容量，图着色能找到一个可行方案，并且能复用 a/b 死后腾出来的位置：

```mlir
  %a = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
  ...
  %b = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>
  ...
  %c = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>
  ...
  %d = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // 复用 a 的位置：a 在 t6 已死
  ...
  %e = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>   // 复用 b 的位置
  ...
  %f = spm.l1_alloc {offset = 49152 : i64} : !spm.l1mem<64x64xf32>   // t13/t14 时 c,d,e 都还活着，只能开新槽
```

IR 结构完全没变，只是每个 `l1_alloc` 多了一个 `offset` 属性——这就是"分配后"最简单的样子：**没有 spill
时，分配算法只往 IR 上贴属性，不改变任何 op 的结构。**`a`/`d` 共享 offset 0、`b`/`e` 共享 offset
16384，是图着色"给互不冲突的 alloc 分配相同静态 buffer id"的直接体现；`f` 因为和 c/d/e 三者在 t13/t14
同时活着，拿不到任何一个空位，只能开在第 4 个槽位（总占用刚好 4×16K = 64K，压满整个容量）。

#### 情况二：容量不够（capacity = 48K = 3 单位）——触发 spill

peak② = 64K > 48K，超出 1 个单位（16K）。溢出窗口是 `[t13, t14]`（f 刚分配到 matmul2 执行完这一小段，
c/d/e/f 四者同时活着）。按 §4 candidate 规则，窗口内候选是"活跃区间跨过整个窗口、且窗口后还有真实使用"
的 buffer：

- `d`、`e` 的最后一次使用就在 t14（窗口末尾），窗口后没有使用了 —— **排除**（它们本来就要自然死亡）。
- `c`：跨过窗口，窗口后下一次使用在 t19（`load c`），distanceToNextUse = 19 − 14 = 5。
- `f`：t13 才出生，恰好等于窗口本身，窗口后下一次使用在 t17（`load f`），distanceToNextUse = 17 − 14 = 2。

打分：`score(c) = 16K × 5 = 80K`，`score(f) = 16K × 2 = 32K`。`c` 分数更高，选中 `c` 作为 spill 对象——
和直觉一致：`c` 之后要等很久才会再被用到，现在挪出去"不心疼"；`f` 挪出去马上又要用，等于白白多花一次
搬运。只需要腾出 1 个单位，spill 一个 `c` 就够了。

**插入 spill/fill 之后的 IR**（`insertPoint` = t6 之后 c 最后一次真正被使用的地方，也就是紧跟着
`matmul1`；`resumePoint` = t19 的 `load c` 之前）：

```mlir
  ...
  linalg.matmul ins(%a, %b : ...) outs(%c : !spm.l1mem<64x64xf32>)
  spm.l1_dealloc %a : !spm.l1mem<64x64xf32>
  spm.l1_dealloc %b : !spm.l1mem<64x64xf32>

  // >>> 溢出驱动循环新插入的部分：先在 L2 上显式 alloc，再拷贝，c 的 L1 物理存储到此为止 <<<
  %c_l2 = spm.l2_alloc : !spm.l2mem<64x64xf32>
  spm.l1_spill %c, %c_l2 : !spm.l1mem<64x64xf32>, !spm.l2mem<64x64xf32>

  %d = spm.l1_alloc : !spm.l1mem<64x64xf32>
  ... // 阶段二不变
  spm.l1_dealloc %f : !spm.l1mem<64x64xf32>

  // >>> 用之前先在 L1 上显式 alloc，再拷贝回来，得到一个全新的 SSA 值 %c2 <<<
  %c2 = spm.l1_alloc : !spm.l1mem<64x64xf32>
  spm.l1_fill %c_l2, %c2 : !spm.l2mem<64x64xf32>, !spm.l1mem<64x64xf32>
  spm.l2_dealloc %c_l2 : !spm.l2mem<64x64xf32>

  // 阶段三：所有原本引用 %c 的地方，从这里开始换成 %c2
  %c_result = spm.l1_load %c2 : !spm.l1mem<64x64xf32> -> tensor<64x64xf32>
  spm.l1_dealloc %c2 : !spm.l1mem<64x64xf32>
  return %f_result, %c_result : tensor<64x64xf32>, tensor<64x64xf32>
```

IR 变了之后回到 §3 的 `loop`：重新做一遍 alias+liveness+图着色，这次 `c` 的活跃区间被切成了很短的
`[t5, spill点]`，`c2` 是一个全新的、同样很短的 `[fill点, t19]`，两段都不再需要在 t13/t14 那个窗口
里占位——峰值自然回落到 48K，一次迭代就收敛（不需要再触发第二轮 spill）。最终贴上 offset 之后：

```mlir
  %a  = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
  %b  = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>
  %c  = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>
  ...
  %c_l2 = spm.l2_alloc {offset = 0 : i64}   : !spm.l2mem<64x64xf32>   // L2 这一遍分配算法独立跑，offset 从 0 开始
  spm.l1_spill %c, %c_l2 : !spm.l1mem<64x64xf32>, !spm.l2mem<64x64xf32>
  %d  = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // 复用 a 的位置
  %e  = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>   // 复用 b 的位置
  %f  = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>   // c 已经 spill 走了，复用 c 的位置
  ...
  %c2 = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // d/e/f 都已死，随便复用一个
  spm.l1_fill %c_l2, %c2 : !spm.l2mem<64x64xf32>, !spm.l1mem<64x64xf32>
```

总占用回到 3×16K = 48K，压满但不超过容量；代价是多了一次 L1→L2→L1 的往返拷贝，换来的是原本需要 64K
容量的程序现在 48K 也能跑。这就是"分配前 / 分配后（够用）/ 分配后（触发 spill）"三种形态的完整对照。

---

## 2. 每层的 Alias / Liveness：零新增规则

修正 §1.2 的 op 签名之后，这一节比上一版更简单：`NPU_SRAM_ALLOCATION_PORTING.md` §3 的四条别名规则
（seed / view 透传 / select 取并 / 其他一律断言）**原样保留、一条都不用加**，按内存类型分层独立跑
（L1 的分析只看 `!spm.l1mem` 值，L2 的分析只看 `!spm.l2mem` 值，互不干扰）。

原因：`spill`/`fill` 现在是没有返回值的纯拷贝 op（同构于 `l1_store`），根本不产生新的 SSA 值，
**不参与别名格（lattice）的传播**——它们只是对某个已有 alloc 值的一次"使用"，和 `l1_store` 的 `dst`
操作数完全同类。真正"开启新别名链"的永远只有 `*_alloc`（第一条规则已经覆盖，包括为溢出/回填新插入的
那些 alloc——它们和手写/lowering 产生的 alloc 在 IR 里毫无区别）。

§4 的 liveness 算法（后序编号 + `mlir::Liveness` + 别名扩展）也**完全不用改**：只要 §3 的驱动逻辑确实
把 spill 点之后所有真实 use 都重写到了新 alloc 出来的 buffer 上，源 buffer 在 IR 里就再也没有更晚的
use 了，`mlir::Liveness` 自动把它的活跃区间算到 spill 这一点为止——不需要专门标注"这是终结点"。这正是
porting doc §2 反复强调的"buffer 生死由标准 use-def 活跃区间决定，不要扫描某个特殊 op 位置来定生命周期"
这条原则，直接套用在 spill/fill 上就够了；上一版单独发明"sink/seed"规则是想多了，删掉。

§5 的图着色分配算法（`calculateStarts` → `buildInterferenceGraph` → `allocate` 不动点）也**完全不用改**，
原样按层独立跑一遍即可。**这是本设计最重要的一点**：溢出机制不是去改造图着色算法本身，而是在图着色算法
外面包一层"发现放不下就改写 IR、改写完再重新跑图着色"的驱动逻辑——和经典寄存器分配器
（Chaitin-Briggs：着色失败 → 插入 spill code → 重新构建干涉图 → 重新着色）是同一个结构。

---

## 3. 核心算法：单层 `AllocateTier`

```text
AllocateTier(IR, tier, capacity, nextTier):
  loop:
    aliasInfo   = AliasAnalysis(IR, tier)                # §2，只看该层 memdesc 类型
    bufferRange = Liveness(IR, aliasInfo)                 # porting doc §4，原样复用
    offsets     = GraphColorAllocate(bufferRange)         # porting doc §5，原样复用
    occupancy   = OccupancyProfile(offsets, bufferRange)  # 每个程序点的并发占用字节数
    overflow    = { p : occupancy(p) > capacity }
    if overflow is empty:
      stamp `offset` attribute onto every *_alloc/_fill op at this tier
      return
    if nextTier is null:
      error("buffer 在最后一级存储上仍然放不下，无法继续溢出")
    window  = earliest interval in overflow
    victims = SelectSpillVictims(window, bufferRange, capacity)   # §4
    for v in victims:
      insertPoint  = last use of v strictly before window
      resumePoint  = first use of v strictly after window
      spillDst  = insert <nextTier>_alloc right before the spill        # 目标层显式 alloc（§1.2 修正）
      insert <tier>_spill(v, spillDst) right after insertPoint          # 纯拷贝，无返回值
      refillDst = insert <tier>_alloc right before the fill             # 本层显式 alloc
      insert <nextTier>_fill(spillDst, refillDst) right before resumePoint  # 纯拷贝，无返回值
      rewrite all uses of v at/after resumePoint to use `refillDst` instead
    goto loop   # IR 变了，从头重新分析
```

顶层驱动，严格级联、不跳级：

```text
AllocateTier(IR, L1,   capacityL1,   nextTier = L2)
AllocateTier(IR, L2,   capacityL2,   nextTier = DRAM)   # 这一步会同时看到原生 l2_alloc
                                                          # 和上一步因 L1 溢出新产生的 l1_fill 目标
AllocateTier(IR, DRAM, capacityDRAM, nextTier = null)    # 见 §9，通常直接判定 overflow 恒为空
```

L2 这一遍跑的时候，"工作集"已经自动包含了 L1 溢出下来的那些 buffer——它们的物理形态就是 L1 那一轮
`AllocateTier` 为 spill/fill 目标插入的、普普通通的 `spm.l2_alloc` op（§1.2 修正后的版本），**不需要
专门的"外来溢出 buffer"特殊分支**：L2 的 `AllocateTier` 扫描 `l2_alloc` 时，根本分不出（也不需要分出）
哪些是原生的、哪些是上一层溢出插入的。层与层之间的耦合被彻底压缩成了"上一层跑完之后 IR 里多了几个新的
该层 `alloc` + 拷贝 op"这一件事。

---

## 4. Spill 候选打分：`SelectSpillVictims`

因为整条 use-def 链在编译期完全已知（静态 shape、静态调度、没有运行时不确定性），这是一个 **offline**
问题——不需要 LRU 这类只能猜测未来的在线启发式，可以直接用"未来最晚才用到的先踢"（Belady 最优置换的
直觉）再叠加 buffer 大小做加权，这也是 vDNN/Capuchin 这类 GPU 显存 offload 系统的标准做法：

```text
score(v, p) = size(v) * distanceToNextUse(v, p)
```

- `size(v)` 越大，踢出去省的空间越多；
- `distanceToNextUse` 越大（离下次真正被用到还很远），说明"现在不用它、之后也不用马上取回来"，
  踢出去更划算。

两者都应该**抬高**分数，所以是相乘，不是相除——早先一版写成 `size / distanceToNextUse` 是笔误：除法会让
"离下次使用越远"反而拉低分数，和想要的效果正好相反。§1.3 的算例里用相乘的版本验证过（`c` 因为
`distanceToNextUse` 更大而被优先选中，符合直觉）。

对 `window` 内每个仍然活跃的候选 buffer 算一次 score，从高到低贪心踢，直到 `occupancy(window) ≤ capacity`。
两个需要显式处理的边界：

- 若一个候选在 `window` 内**没有**"下次使用"（也就是它的活跃区间恰好在 window 内结束），
  说明它本来就要死了，`distanceToNextUse = ∞` 效果上等价于最高优先级踢除——但这种情况其实不需要真的
  插 spill/fill，直接等它自然死亡即可（图着色本身会处理），不应该被 `SelectSpillVictims` 选中。
  所以候选集合应该限定为"活跃区间**跨过**整个 window 的 buffer"，天然排除了这种情况。
- 若把 window 内所有候选都踢完仍然放不下（单个不可再拆分的 buffer 比 capacity 还大），
  这是 §3 里 `nextTier is null` 分支要报的硬错误的另一种前置形式——即使 `nextTier` 非空，
  也要检测"这个 buffer 自己的 size 就超过 capacity"并直接报错，而不是无限重试。

插入点选取（对应 §3 伪代码里的 `insertPoint`/`resumePoint`）：

- **spill 尽量早插**：紧跟在 `window` 之前那次真实使用后面，让空间尽快腾出来；
- **fill 尽量晚插**：紧贴 `window` 之后那次真实使用之前，让数据尽量晚一点占用回该层空间。

v0 不做"提前预取（prefetch）以和计算重叠"的调度优化——`fill` 就是这次使用前的同步阻塞搬运，
和现有 `l1_load`/`l1_store` 的同步语义保持一致（异步搬运本来就在 `NPU_LOAD_STORE_MEMDESC_PORTING.md`
§8 里被列为 v0 明确不做的范围）。

---

## 5. 收敛性

外层 `loop` 每一轮要么直接收敛（`overflow` 为空），要么至少把 `window` 内一个 buffer 的物理占用时间
严格缩短（原来跨越整个原始活跃区间，现在被切成 spill 前/fill 后两段更短的区间，中间那段完全不占用
该层空间）。由于 buffer 数量有限，"每轮至少永久性缩短一个 buffer 在某个冲突窗口内的占用"这件事不能
无限发生，所以外层循环必然终止（工程上和 porting doc §5 提到的内层不动点循环一样，没有给出严格的
收敛上界证明，但同一套论证方式在 Triton 那边已经用了很多年）。

---

## 6. 循环（`scf.for`）内 buffer 怎么办

如果溢出窗口落在循环体内部、且被选中的 victim 是一个通过 `iter_args` 跨迭代传递的 buffer
（porting doc §4 提到的"alias 扩展让循环里的 buffer 区间正确覆盖整个循环"那种情况），
本设计**不做特殊分支**：spill/fill 直接插在循环体内部对应位置，结构上会随循环体一起每次迭代都执行一遍。
这样得到的结果是**正确但不是最优**的（如果这个 buffer 其实在循环的每一轮里都会立刻被用到，
每轮都 spill 一次再 fill 回来就是纯浪费的 DRAM/L2 带宽）。把"循环不变的 spill/fill 提到循环外"
是一个后续可以再加的优化（类似 LICM），v0 先保证正确性、不做这一层优化。

---

## 7. v0 明确排除/简化的范围

| 内容 | 为什么先不做 |
|---|---|
| 跨级跳跃（L1 满了直接丢 DRAM，跳过 L2） | 级联顺序已经能覆盖正确性，跳级是纯粹的性能优化，等有实测数据再决定要不要加 |
| 异步 spill/fill（和计算重叠的 DMA prefetch） | 和 `l1_load`/`l1_store`/`async_copy_*` 的既有排除范围一致，等 NPU 的异步搬运原语设计出来后再接 |
| DRAM 侧真正的空间复用 | `spm.dram_alloc`/`spm.dram_dealloc` op 本身已经存在（§1.2 修正后），但 v0 里"DRAM 这一层的 `AllocateTier`"可以先是一个不做复用的 trivial bump 分配器（每个 `dram_alloc` 各占一块新地址，`dealloc` 提示忽略），而不是像 L1/L2 那样跑完整的图着色。DRAM 容量相对宽裕，值得先接受这个浪费；如果 DRAM 占用本身成为问题，直接把 §3 的完整 `AllocateTier`（含 `nextTier = null`，capacity 给一个有限值）套用到 DRAM 层即可，op/IR 层不用改 |
| 循环不变 spill/fill 外提（LICM 式优化） | 见 §6，先保证正确性 |
| Spill 决策感知"计算耗时"来判断能否被 DMA 隐藏 | v0 的 cost 模型只看 size/复用距离，不建模具体的计算-访存重叠时间线 |

---

## 8. 建议实现顺序

1. `!spm.dram` 类型（照抄 `l1mem`/`l2mem` 字段）。
2. `spm.l1_alloc`/`spm.l1_dealloc`/`spm.l2_alloc`/`spm.l2_dealloc`/`spm.dram_alloc`/`spm.dram_dealloc`
   （porting doc 已给出 L1/L2 的完整设计，dealloc 只做验证不参与生命周期计算；`dram_alloc`/`dram_dealloc`
   是 §1.2 修正后新补的，三层同构，照抄同一套设计）。
3. `SRAMAliasAnalysis`，按内存类型参数化，先只覆盖 L1（4 条规则 + alloc=seed）。
4. Liveness + 图着色分配（porting doc §4-§5），先跑通"L1 不考虑溢出、假设容量无限"的版本，
   验证和 Triton 版本行为一致。
5. 接入真实容量：`OccupancyProfile` + overflow 检测，先只做到"检测到超容量就报错"（porting doc
   §7 建议的第 6 步），确认这一步的判据是对的。
6. `spm.l1_spill`/`spm.l1_fill` + `SelectSpillVictims` + IR 重写（§3-§4），把上一步的"报错"换成
   "自动插入 spill/fill 再重试"，先只验证 L1 这一层单独工作。
7. 别名规则扩展到 spill=sink / fill=seed，用带真实溢出场景的 lit 测试验证活跃区间被正确切分。
8. 把 §3 的三层级联驱动接起来（L1→L2→DRAM），复用第 2-7 步已经跑通的单层逻辑，`spm.l2_spill`/
   `spm.l2_fill`/`!spm.dram` 到这一步才第一次真正被驱动到。
9. 循环场景的正确性测试（`scf.for` 里的 buffer 溢出），确认 §6 描述的"结构上随循环体重复"行为符合预期，
   不需要而不是"偶然"没触发。

每一步都能独立编译、独立写 lit 测试验证，出问题时容易定位是"分配算法本身"还是"溢出改写逻辑"的问题。
