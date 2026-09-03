# SPM SRAM 静态分配 + 三级溢出（Spilling）设计

本文承接 [NPU_SRAM_ALLOCATION_PORTING.md](NPU_SRAM_ALLOCATION_PORTING.md)（Triton 图着色分配算法的可移植部分）和
[NPU_LOAD_STORE_MEMDESC_PORTING.md](NPU_LOAD_STORE_MEMDESC_PORTING.md)（memdesc 类型 + load/store op 设计），
是原创设计部分——两篇 porting 文档都明确指出 "spill-to-DRAM 是 Triton 原设计里没有、需要自己另外设计的部分"。

设计目标：在 L1 / L2 / DRAM 三级存储之上，设计一个把"图着色静态分配"和"容量不够时溢出到下一级存储"
结合起来的算法，覆盖 **L1 → L2 → DRAM** 三级级联溢出。

---

## 0. 本轮修订概要

这一版相对上一版有两处结构性改动，都是为了让"溢出"不再是一套独立机制：

| 上一版 | 本版 | 理由 |
|---|---|---|
| 新增 `l1_spill`/`l1_fill`/`l2_spill`/`l2_fill` 四个 op | **全部删除**。spill 就是 `store`，fill 就是 `load` | 上一版已把 spill/fill 修正成"无返回值 + 目标必须预先 alloc"，此时它和 `store` 的签名、语义完全同构，保留两套名字是纯冗余（§3.3） |
| load/store 语义是 buffer ↔ tensor（"计算域"） | load/store 语义是 **tier ↔ tier**（`l1_load` = L2→L1，`l1_store` = L1→L2） | 计算 op 直接消费 memdesc，不需要 tensor 中间层；改成 tier-relative 之后 spill/fill 才能被 store/load 吸收（§3.1） |
| 隐含假设"数据可以停在任意一级" | 引入 **L1-resident canonical form**：分配前所有数据一律显式加载到 L1，中途不回落 | 把 placement 自由度彻底消灭，让 L2 的需求变成 spilling 的**派生量**，L2 pass 因此可以原样复用 L1 pass 的算法（§4） |

另外撤销了上一版一处说法（"alloc 和 dealloc 都只是为了可读性"）：这条对 `dealloc` 成立，对 `alloc` 不成立，见 §3.4。

---

## 1. 设计前提：现有的 op 和 type 都不是既定事实

`spm-mlir/include/SPM/SPMOps.td` 和 `SPMTypes.td` 里已经落地的东西
（`!spm.l1mem`/`!spm.l2mem`，以及四个 buffer ↔ tensor 语义的 `l1_load`/`l1_store`/`l2_load`/`l2_store`）
**不是本设计的权威参考，也不构成约束**。它们是在分配算法还没设计之前先搭出来的脚手架。

本文的立场是反过来的：**分配 + 溢出算法是主导，op 和 type 随算法需要随时重构**。具体到这一版，
现有四个 load/store op 的 `arguments`/`results` 会被整体改写（§3.1），并新增 `!spm.dram` 类型和
三层 alloc/dealloc。凡是本文和现有 `.td` 冲突的地方，一律以本文为准，`.td` 跟着改。

这一条单独列成一节，是因为它直接决定了后面很多取舍的方向：遇到"算法想要 A，但现有 op 是 B"时，
正确的反应是改 op，而不是给算法打补丁去适配 B。

---

## 2. 与 one-shot-bufferize 的衔接：为什么别名分析可以很"薄"

one-shot-bufferize 阶段一的核心不变量（见 `one-shot-bufferize-outline.md`）是：分析阶段结束后，
"此后任何机械式的结构化改写都不会引入 RAW 冲突"。当阶段二把这份已经过冲突消解的 tensor IR
机械改写成 `spm.l1_alloc` + 计算 op 直接消费 memdesc（DPS 类 op 直接复用 `outs` 对应 buffer）时，
**每个 `alloc` 出发的 def-use chain 内部天然无冲突**——这是本设计能够直接照搬 Triton `AliasInfo`
（见 `NPU_SRAM_ALLOCATION_PORTING.md` §3）那套"链内无需判断，只需追踪谁和谁共享物理存储"规则的前提。

于是职责划分很清楚：

- **bufferize 阶段一**：保证单条 def-use chain 内部（同一个 alloc 的所有 view/传递）不会有 RAW 冲突。
- **本文的 SRAM 分配器**：保证**不同 alloc 之间**（跨 chain）不会被分配到重叠的物理地址/时间段，
  以及当所有 chain 的总占用超过某一级存储容量时，如何把部分数据挪到下一级。

两者互不越界——分配器的 alias 分析不需要重新证明"in-place 安全"，只需要判断"这个 SSA 值是不是
某个 alloc 的别名"。

---

## 3. Op 家族：tier-relative 的 load/store

### 3.1 命名约定：load = 进入本层，store = 离开本层

这是本版最需要在文档里钉死的一条约定，否则 `l2_load` 这个名字必然被读错：

> `lN_load` 把数据**搬进** tier N（源在更慢的一层）；`lN_store` 把数据**搬出** tier N（目标在更慢的一层）。
> 名字里的层级永远指**本层**，也就是"离计算更近的那一端"。

| Op | 源 | 目标 |
|---|---|---|
| `spm.l1_load` | `!spm.l2mem`（若支持直达则也可以是 `!spm.dram`，见 §3.6） | `!spm.l1mem` |
| `spm.l1_store` | `!spm.l1mem` | `!spm.l2mem` |
| `spm.l2_load` | `!spm.dram` | `!spm.l2mem` |
| `spm.l2_store` | `!spm.l2mem` | `!spm.dram` |

注意这套约定的直接后果：**"把 L2 里的数据读到计算域"这件事没有对应的 op，也不需要有**。
计算 op（`linalg.*` 等）只接受 `!spm.l1mem` 操作数，所有计算都发生在 L1 上（这是 §4 canonical form
的一部分）。上一版里 `l2_load: l2mem -> tensor` 那个语义整个被删掉。

曾考虑过的替代命名是一个通用的 `spm.dma %src, %dst`，靠操作数类型区分方向，歧义为零。没有采用，
是因为每一层的分配器需要"扫描属于本层的 op"来发现工作集（`AllocateTier`，§7），按层命名的 op
让这一步是纯粹的 op 类型匹配；通用 `dma` 会把它变成"匹配 op 之后再看操作数类型"。这个理由不强，
如果实现时发现 tier-relative 命名反复被读错，换成 `spm.dma` 是可接受的重构（见 §1）。

### 3.2 完整 op 表

| 阶段 | Op | 状态 | 语义 |
|---|---|---|---|
| 分配 | `spm.l1_alloc` / `spm.l2_alloc` / `spm.dram_alloc` | **新增** | 产生一块新 buffer，此时不带地址；三层同构。**唯一开启别名链的 op，也是 `offset` 属性唯一的挂载点** |
| 释放提示 | `spm.l1_dealloc` / `spm.l2_dealloc` / `spm.dram_dealloc` | **新增** | 纯提示/校验用，**不参与生命周期计算**（照抄 Triton `local_dealloc` 的语义划分，见 porting doc §2） |
| 跨层搬运 | `spm.l1_load` / `spm.l1_store` / `spm.l2_load` / `spm.l2_store` | **语义改写** | 见 §3.1。一律**无返回值**，写入一个调用方预先 alloc 好的目标 memdesc |

三层严格只和相邻层对话（`l1_*` 只碰 L1↔L2，`l2_*` 只碰 L2↔DRAM），v0 不支持跳级
（比如 L1 满了直接丢 DRAM）。跳级留作后续优化，不在 v0 范围内；§3.6 是这条规则唯一的待定例外。

### 3.3 spill 就是 store，fill 就是 load

上一版为溢出专门设计了 `l1_spill`/`l1_fill` 等四个 op，签名是"无返回值 + 目标必须由调用方预先 alloc"。
一旦 load/store 改成 tier-relative（§3.1），这四个 op 和 store/load 的签名、语义、验证条件就**完全重合**了：
两者都是"把一块已有 buffer 的内容拷进另一块已有 buffer，不产生新值"。因此四个 spill/fill op 全部删除。

```mlir
// spill：L1 装不下 %c 了，把它挪到 L2 —— 这就是一次普通的 l1_store
%c_spill = spm.l2_alloc : !spm.l2mem<64x64xf32>
spm.l1_store %c, %c_spill : !spm.l1mem<64x64xf32> -> !spm.l2mem<64x64xf32>
// %c 的 L1 物理存储到此为止（由 liveness 自动算出，不需要标注）

// fill：要用回来了，先在 L1 显式 alloc，再拷回来 —— 这就是一次普通的 l1_load
%c2 = spm.l1_alloc : !spm.l1mem<64x64xf32>
spm.l1_load %c_spill, %c2 : !spm.l2mem<64x64xf32> -> !spm.l1mem<64x64xf32>
spm.l2_dealloc %c_spill : !spm.l2mem<64x64xf32>
// 后续所有原本用 %c 的地方，从这里开始改用 %c2
```

这么做换来的最重要的性质是：**溢出产生的 IR 和正常数据加载产生的 IR 在结构上完全不可区分**。
`%c_spill = spm.l2_alloc` 和一个原生的、为 DRAM→L1 transit 服务的 `l2_alloc`，在 L2 那一遍
`AllocateTier` 眼里就是同一个东西。层与层之间的耦合被压缩成了"上一层跑完之后 IR 里多了几个本层的
`alloc` + 搬运 op"这一件事，分配算法不需要任何"这是溢出 buffer"的后门。

同时它也解释了为什么目标必须显式预先 `alloc`（这是上一版就已经修正过的结论，这里重申理由）：
每一层的分配算法都是靠"扫描这一层所有 `*_alloc` op"来发现自己的分配对象的（porting doc §7 骨架
第 2 步），如果搬运 op 自己"生出"一块目标 buffer 作为返回值，那块 buffer 就没有任何 op 可以承载
算出来的 `offset` 属性，该层的 `AllocateTier` 根本看不到它。

### 3.4 `alloc` 和 `dealloc` 的地位并不对称

一个自然的想法是："既然 liveness 分析已经能从 use-def 算出每个物理 buffer 的活跃区间，
那 alloc/dealloc 就都不是必需的，只是 materialize 阶段的可读性糖。"

**这条对 `dealloc` 成立，对 `alloc` 不成立。**

- `dealloc` 确实是纯注释：buffer 的死亡点由"最后一次 use"决定，扫描 `dealloc` 的位置来定生死是
  porting doc §2 明确反对的做法。整个分配算法可以在完全没有 `dealloc` op 的 IR 上正确工作。
- `alloc` 是 load-bearing 的，有三个不可替代的职责：
  1. 它是那个 SSA 值的 **def**——"一块物理 buffer"这个概念在 IR 里就是靠这个值来命名和传递的，
     别名分析的第一条 seed 规则就是它；
  2. 它是 `offset` 属性**唯一的挂载点**（§5"分配后"的形态就是往每个 alloc 上贴一个 `offset`）；
  3. 它是每一层分配器的**扫描入口**。

更关键的是：**计算产出的 buffer 根本没有对应的 load**。`%c` 作为 `linalg.matmul` 的 `outs` 操作数，
数据不是从任何地方搬进来的，而 DPS 语义又要求 `outs` 操作数事先存在——所以即使把 `alloc` 融进搬运 op
（`load` 顺便产生目标 buffer），计算输出仍然必须有一个独立的 `alloc`。既然无法统一消掉，统一保留才一致。

准确的表述是：**liveness 让 `dealloc` 变成可选的注释；`alloc` 仍然是分析的基本单位。**
这也是既定原则"绝不把 alloc 和 data-move 融进一个 op，否则 tier 分配器扫不到"的直接推论。

### 3.5 `!spm.dram` 类型

```tablegen
def SPM_DRAMType : SPM_MemDescTypeBase<"DRAM", "dram"> {
  // 字段完全镜像 l1mem/l2mem：shape, elementType, mutableMemory, allocShape
}
```

和 `l1mem`/`l2mem` 保持同一套字段、同一个 `ShapedTypeInterface` 基类，是第三个独立类型而非
"memref + address space attr"——延续 scaffold 里"三种独立类型而非一种类型 + 内存空间属性"的既有选择。

### 3.6 待定项：DRAM → L1 是否直达

§4 的 canonical form 默认走**级联**：`dram_alloc` 之后紧跟 `l2_alloc` + `l2_load` + `l1_alloc` + `l1_load`，
数据经 L2 中转进 L1。这和"v0 不跳级"一致，但代价是本来要直奔 L1 的数据白吃一遍 L2 带宽和一块 L2 容量。

**如果硬件 DMA 支持 DRAM → L1 直达**，更干净的形态是让 `spm.l1_load` 的源类型同时接受
`!spm.l2mem` 和 `!spm.dram`：

```tablegen
def SPM_SlowerThanL1 : AnyTypeOf<[SPM_L2MemType, SPM_DRAMType]>;
```

这样 **L2 就不再有任何"原生"需求，它的全部内容都由 L1 的 spilling 产生**——§4 那条论证被推到最纯的形式，
canonical form 里根本不出现 `l2_alloc`，L2 的工作集从空集开始。返回路径同理（L1 直接 store 回 DRAM）。

| | 变体 A（级联，v0 默认） | 变体 B（直达） |
|---|---|---|
| `l1_load` 源类型 | 只有 `!spm.l2mem` | `!spm.l2mem` 或 `!spm.dram` |
| canonical form 里的 L2 | transit staging，每块只活一跳 | 完全不出现 |
| L2 工作集来源 | transit + L1 溢出 | **纯 L1 溢出** |
| 每个输入的额外 L2 流量 | 一次读 + 一次写 | 无 |

**这一项需要硬件答案才能定。** 在拿到答案之前，本文其余部分按变体 A 描述（它是两者中约束更强的，
从 A 改到 B 只是放宽 `l1_load` 的源类型并删掉 canonical form 里的 transit，不影响任何算法）。

---

## 4. Canonical form：分配前所有数据一律显式落在 L1

### 4.1 定义

分配算法的输入 IR 必须处于 **L1-resident canonical form**，即满足：

1. **所有计算发生在 L1 上**。计算 op 只接受 `!spm.l1mem` 操作数和 `outs`。
2. **每个 DRAM 侧的值（函数参数、常量、`dram_alloc`）后面紧跟到 L1 的完整搬运链**：
   `l2_alloc` + `l2_load` + `l1_alloc` + `l1_load`（变体 B 下是 `l1_alloc` + `l1_load`）。
3. **中间结果永不回落**。计算产出的 buffer 就留在 L1，不存在"这个中间结果先放 L2"的 IR。
4. **只有整个计算图 return 的时候才写回 DRAM**：`l1_store` + `l2_store` 级联下去。

也就是说，canonical form 是一个**假设 L1 容量无限**的 IR。它一定不满足真实容量约束，这是故意的。

### 4.2 为什么要这个假设

它把 **placement 这个自由度彻底消灭了**。在 canonical form 里不存在"这块数据放 L1 还是放 L2"的决策——
答案永远是 L1。所有的存储层级决策被推迟到唯一一个地方：spilling。

于是整套流程和经典寄存器分配严格同构，不只是"结构相似"：

| 寄存器分配 | 本设计 |
|---|---|
| 假设所有值都在寄存器里 | canonical form：所有 buffer 都在 L1 |
| 构建干涉图 + 着色 | alias + liveness + 图着色 |
| 着色失败 | occupancy > capacity |
| 插入 spill code（store/load 到栈） | 插入 `l1_store`/`l1_load`（到 L2） |
| 重建干涉图、重新着色 | `goto loop`，重新分析（§7） |
| 栈帧本身也要分配 | L2 层再跑一遍同一个 `AllocateTier` |

最后一行是这个 canonical form 真正的收益：**L2 的需求变成纯派生量**。L1 pass 跑完之后，L2 需要装的
东西（变体 A 下是 transit staging + L1 溢出物；变体 B 下只有 L1 溢出物）已经全部以普普通通的
`spm.l2_alloc` op 的形式显式存在于 IR 里了。L2 的 `AllocateTier` 因此可以**原样**是同一个算法跑在
一个更小的、派生出来的问题上，不需要任何"L1 会不会溢出下来"的预测或反馈。DRAM 层同理。

### 4.3 为什么这个假设不悲观

一个自然的担心是："把所有数据都强行落 L1，会不会人为放大 L1 压力、凭空造出本来不存在的 spill？"

**不会。** liveness 仍然在最后一个消费者处杀掉 buffer，所以 canonical form 的 L1 峰值就等于
**程序自然的工作集峰值**，没有任何额外膨胀。canonical form 唯一引入的额外占用是 transit buffer
（变体 A 下），而 transit 只影响 **L2** 占用，不影响 L1：`l1_load %a_l2, %a` 执行时 L2 侧多一块
`%a_l2`、L1 侧就是 `%a` 本身，L1 上并没有多出任何东西。

反过来看也成立：一个"产出很早、消费很晚"的 buffer，在 canonical form 里会一直占着 L1 从而触发 spill；
而一个"聪明的 placement"会一开始就把它放 L2、要用时再取。这两件事的最终 IR 是**同一个**——
spill 出来的就是"存 L2、要用时取回"。所以 canonical form + spilling 不会比手工 placement 差，
它只是把决策时机推迟到了信息最完整的时候（liveness 和 occupancy 都已知）。

### 4.4 常量策略

v0 规定：**所有常量一律先在 DRAM 落地**（`dram_alloc` + 初始化数据段），然后走和普通输入完全相同的
搬运链进 L1。不为常量开任何特殊路径。

理由：如果允许"小的标量 splat / 立即数直接在 L1 上就地物化"，就需要重新引入一个
tensor（或 scalar）→ `l1mem` 的物化 op，也就是 §3.1 刚刚删掉的那类 op，canonical form 的第 2 条
不变量随之破掉。常量走 DRAM 会浪费一点带宽，但保住了"L1 上的每一块 buffer 要么是 `l1_load` 搬进来的、
要么是计算 op 产出的"这条二分法。如果实测发现小常量的搬运开销显著，再单独设计一个
`spm.l1_materialize_constant` 并同步扩充别名规则，属于后续优化。

---

## 5. 完整示例：canonical form → 分配后（够用）→ 分配后（触发 spill）

所有 buffer 都是 `64x64xf32` = 16KB，记一个"单位" = 16KB。为了让示例聚焦在 L1，下面省略了
transit buffer 的 `l2_alloc`/`l2_dealloc`（用 `// DRAM->L2->L1` 注释代替），它们不影响 L1 占用（§4.3）。

**Canonical form（分配前，所有 `l1_alloc` 都还没有 `offset`）：**

```mlir
func.func @accum_across_burst(%in_a: !spm.dram<64x64xf32>, %in_b: !spm.dram<64x64xf32>,
                               %in_d: !spm.dram<64x64xf32>, %in_e: !spm.dram<64x64xf32>)
    -> (!spm.dram<64x64xf32>, !spm.dram<64x64xf32>) {
  // ---- 阶段一：c = a @ b（a、b 用完即死；c 要活到阶段三）----
  %a = spm.l1_alloc : !spm.l1mem<64x64xf32>
  // DRAM->L2->L1: %in_a ==> %a
  %b = spm.l1_alloc : !spm.l1mem<64x64xf32>
  // DRAM->L2->L1: %in_b ==> %b
  %c = spm.l1_alloc : !spm.l1mem<64x64xf32>
  linalg.matmul ins(%a, %b : !spm.l1mem<64x64xf32>, !spm.l1mem<64x64xf32>)
                outs(%c : !spm.l1mem<64x64xf32>)                            // t6

  // ---- 阶段二：一段完全不碰 c 的"突发"计算，f = d @ e ----
  %d = spm.l1_alloc : !spm.l1mem<64x64xf32>
  // DRAM->L2->L1: %in_d ==> %d
  %e = spm.l1_alloc : !spm.l1mem<64x64xf32>
  // DRAM->L2->L1: %in_e ==> %e
  %f = spm.l1_alloc : !spm.l1mem<64x64xf32>                                 // t11
  linalg.matmul ins(%d, %e : !spm.l1mem<64x64xf32>, !spm.l1mem<64x64xf32>)
                outs(%f : !spm.l1mem<64x64xf32>)                            // t12

  // f 立刻写回 DRAM（返回路径）
  %f_l2 = spm.l2_alloc : !spm.l2mem<64x64xf32>
  spm.l1_store %f, %f_l2 : !spm.l1mem<64x64xf32> -> !spm.l2mem<64x64xf32>   // t14
  %f_dram = spm.dram_alloc : !spm.dram<64x64xf32>
  spm.l2_store %f_l2, %f_dram : !spm.l2mem<64x64xf32> -> !spm.dram<64x64xf32>

  // ---- 阶段三：终于再用到 c ----
  %g = spm.l1_alloc : !spm.l1mem<64x64xf32>                                 // t17
  linalg.exp ins(%c : !spm.l1mem<64x64xf32>) outs(%g : !spm.l1mem<64x64xf32>) // t18
  %g_l2 = spm.l2_alloc : !spm.l2mem<64x64xf32>
  spm.l1_store %g, %g_l2 : !spm.l1mem<64x64xf32> -> !spm.l2mem<64x64xf32>
  %g_dram = spm.dram_alloc : !spm.dram<64x64xf32>
  spm.l2_store %g_l2, %g_dram : !spm.l2mem<64x64xf32> -> !spm.dram<64x64xf32>
  return %f_dram, %g_dram : !spm.dram<64x64xf32>, !spm.dram<64x64xf32>
}
```

L1 上真实（活跃区间意义上的，不看 `dealloc` 提示）并发占用随程序推进的变化：

| 程序点 | 事件 | 此刻活着的 L1 buffer | L1 占用 |
|---|---|---|---|
| t1 | `alloc a` | a | 16K |
| t3 | `alloc b` | a, b | 32K |
| t5 | `alloc c` | a, b, c | 48K |
| t6 | `matmul1`（读 a,b，写 c） | a, b, c | 48K —— **peak①** |
| t6 之后 | a、b 的最后一次使用已经发生 | c | 16K |
| t7 | `alloc d` | c, d | 32K |
| t9 | `alloc e` | c, d, e | 48K |
| t11 | `alloc f` | c, d, e, f | **64K** |
| t12 | `matmul2`（读 d,e，写 f；**c 全程没被用到，只是恰好还活着**） | c, d, e, f | 64K —— **peak②** |
| t12 之后 | d、e 的最后一次使用已经发生 | c, f | 32K |
| t14 | `l1_store f`（f 最后一次使用） | c | 16K |
| t17 | `alloc g` | c, g | 32K |
| t18 | `linalg.exp`（c 最后一次使用） | c, g | 32K |

peak② 的关键点：`matmul2` 本身只需要 d/e/f 三个 buffer 同时在场（这正好等于"单个 op 需要多少 buffer
同时在场"的下限），c 只是"活着但这段时间完全用不上"，纯粹因为它的活跃区间跨过了这段突发计算——
**这正是可以被安全 spill 掉的那种 buffer**，和"某个 op 自己的操作数加起来就超容量"（无法通过 spill
解决，见 §12）是完全不同的两回事。

### 5.1 情况一：容量足够（capacity = 64K = 4 单位）

peak② = 64K 恰好等于容量，图着色能找到可行方案，并且能复用 a/b 死后腾出来的位置：

```mlir
  %a = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
  %b = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>
  %c = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>
  %d = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // 复用 a 的位置：a 在 t6 已死
  %e = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>   // 复用 b 的位置
  %f = spm.l1_alloc {offset = 49152 : i64} : !spm.l1mem<64x64xf32>   // t11/t12 时 c,d,e 都活着，只能开新槽
  %g = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // d/e/f 都已死
```

IR 结构完全没变，只是每个 `l1_alloc` 多了一个 `offset` 属性——这就是"分配后"最简单的样子：
**没有 spill 时，分配算法只往 IR 上贴属性，不改变任何 op 的结构。**

### 5.2 情况二：容量不够（capacity = 48K = 3 单位）——触发 spill

peak② = 64K > 48K，超出 1 个单位（16K）。溢出窗口是 `[t11, t12]`（f 刚分配到 matmul2 执行完，
c/d/e/f 四者同时活着）。按 §8 candidate 规则，窗口内候选是"活跃区间跨过整个窗口、且窗口后还有真实使用"
的 buffer：

- `d`、`e` 的最后一次使用就在 t12（窗口末尾），窗口后没有使用了 —— **排除**（本来就要自然死亡）。
- `c`：跨过窗口，窗口后下一次使用在 t18（`linalg.exp`），distanceToNextUse = 18 − 12 = 6。
- `f`：t11 才出生，恰好等于窗口本身，窗口后下一次使用在 t14（`l1_store f`），distanceToNextUse = 14 − 12 = 2。

打分（§8）：`score(c) = 16K × 6 = 96K`，`score(f) = 16K × 2 = 32K`。选中 `c`——和直觉一致：
`c` 之后要等很久才会再被用到，现在挪出去"不心疼"；`f` 挪出去马上又要用，等于白白多花一次搬运。
只需要腾出 1 个单位，spill 一个 `c` 就够了。

**插入 spill/fill 之后的 IR**（`insertPoint` = t6 之后，c 最后一次真正被使用的地方；
`resumePoint` = t18 之前）：

```mlir
  linalg.matmul ins(%a, %b : ...) outs(%c : !spm.l1mem<64x64xf32>)

  // >>> spill：就是一次普通的 l1_store，目标先在 L2 显式 alloc <<<
  %c_spill = spm.l2_alloc : !spm.l2mem<64x64xf32>
  spm.l1_store %c, %c_spill : !spm.l1mem<64x64xf32> -> !spm.l2mem<64x64xf32>

  %d = spm.l1_alloc : !spm.l1mem<64x64xf32>
  ... // 阶段二不变
  spm.l2_store %f_l2, %f_dram : ...

  // >>> fill：就是一次普通的 l1_load，目标先在 L1 显式 alloc，得到全新的 SSA 值 %c2 <<<
  %c2 = spm.l1_alloc : !spm.l1mem<64x64xf32>
  spm.l1_load %c_spill, %c2 : !spm.l2mem<64x64xf32> -> !spm.l1mem<64x64xf32>
  spm.l2_dealloc %c_spill : !spm.l2mem<64x64xf32>

  // 阶段三：所有原本引用 %c 的地方，从这里开始换成 %c2
  %g = spm.l1_alloc : !spm.l1mem<64x64xf32>
  linalg.exp ins(%c2 : !spm.l1mem<64x64xf32>) outs(%g : !spm.l1mem<64x64xf32>)
```

IR 变了之后回到 §7 的 `loop`：重新做一遍 alias + liveness + 图着色。这次 `c` 的活跃区间被切成了很短的
`[t5, spill点]`，`c2` 是一个全新的、同样很短的 `[fill点, t18]`，两段都不再需要在 t11/t12 那个窗口里占位——
峰值自然回落到 48K，一次迭代就收敛。最终贴上 offset：

```mlir
  %a       = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
  %b       = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>
  %c       = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>
  %c_spill = spm.l2_alloc {offset = 0 : i64}     : !spm.l2mem<64x64xf32>   // L2 那一遍独立跑，offset 从 0 起
  spm.l1_store %c, %c_spill : ...
  %d       = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // 复用 a 的位置
  %e       = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>   // 复用 b 的位置
  %f       = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>   // c 已 spill 走，复用 c 的位置
  %c2      = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // d/e/f 都已死
```

L1 总占用回到 3×16K = 48K，压满但不超容量；代价是多了一次 L1→L2→L1 往返拷贝，换来的是原本需要 64K
容量的程序现在 48K 也能跑。

---

## 6. 每层的 Alias / Liveness：零新增规则

`NPU_SRAM_ALLOCATION_PORTING.md` §3 的四条别名规则（seed / view 透传 / select 取并 / 其他一律断言）
**原样保留、一条都不用加**，按内存类型分层独立跑（L1 的分析只看 `!spm.l1mem` 值，L2 的分析只看
`!spm.l2mem` 值，互不干扰）。

原因：所有搬运 op（`l1_load`/`l1_store`/`l2_load`/`l2_store`）都是没有返回值的纯拷贝，根本不产生新的
SSA 值，**不参与别名格（lattice）的传播**——它们只是对某个已有 alloc 值的一次"使用"。真正"开启新别名链"
的永远只有 `*_alloc`（第一条 seed 规则已经覆盖），而且**为溢出新插入的 alloc 和 canonical form 里原生的
alloc 在 IR 里毫无区别**（§3.3）。

liveness 算法（porting doc §4：后序编号 + `mlir::Liveness` + 别名扩展）也**完全不用改**：只要 §7 的驱动
逻辑确实把 spill 点之后所有真实 use 都重写到了新 alloc 出来的 buffer 上，源 buffer 在 IR 里就再也没有更晚的
use，`mlir::Liveness` 自动把它的活跃区间算到 spill 这一点为止——不需要专门标注"这是终结点"。这正是
porting doc §2 强调的"buffer 生死由标准 use-def 活跃区间决定，不要扫描某个特殊 op 位置来定生命周期"
这条原则的直接套用。

图着色分配算法（porting doc §5：`calculateStarts` → `buildInterferenceGraph` → `allocate` 不动点）
同样**完全不用改**，原样按层独立跑一遍即可。**这是本设计最重要的一点**：溢出机制不去改造图着色算法本身，
而是在它外面包一层"发现放不下就改写 IR、改写完再重新跑图着色"的驱动逻辑。

---

## 7. 核心算法：单层 `AllocateTier`

```text
AllocateTier(IR, tier, capacity, nextTier):
  loop:
    aliasInfo   = AliasAnalysis(IR, tier)                 # §6，只看该层 memdesc 类型
    bufferRange = Liveness(IR, aliasInfo)                 # porting doc §4，原样复用
    offsets     = GraphColorAllocate(bufferRange)         # porting doc §5，原样复用
    occupancy   = OccupancyProfile(offsets, bufferRange)  # 每个程序点的并发占用字节数
    overflow    = { p : occupancy(p) > capacity }
    if overflow is empty:
      stamp `offset` attribute onto every *_alloc op at this tier
      return
    if nextTier is null:
      error("buffer 在最后一级存储上仍然放不下，无法继续溢出")
    window  = earliest interval in overflow
    victims = SelectSpillVictims(window, bufferRange, capacity)   # §8
    for v in victims:
      insertPoint = last use of v strictly before window
      resumePoint = first use of v strictly after window
      spillDst    = insert <nextTier>_alloc right before the spill    # 目标层显式 alloc（§3.3）
      insert <tier>_store(v, spillDst) right after insertPoint        # spill == store
      refillDst   = insert <tier>_alloc right before the fill         # 本层显式 alloc
      insert <tier>_load(spillDst, refillDst) right before resumePoint # fill == load
      rewrite all uses of v at/after resumePoint to use `refillDst` instead
    goto loop   # IR 变了，从头重新分析
```

顶层驱动，严格级联、不跳级：

```text
AllocateTier(IR, L1,   capacityL1,   nextTier = L2)
AllocateTier(IR, L2,   capacityL2,   nextTier = DRAM)   # 同时看到 transit 的 l2_alloc
                                                        # 和上一步因 L1 溢出新产生的 l2_alloc
AllocateTier(IR, DRAM, capacityDRAM, nextTier = null)   # 见 §12，v0 可以是 trivial bump 分配器
```

L2 这一遍跑的时候，"工作集"已经自动包含了 L1 溢出下来的那些 buffer——它们的物理形态就是普普通通的
`spm.l2_alloc` op，**不需要专门的"外来溢出 buffer"特殊分支**：L2 的 `AllocateTier` 扫描 `l2_alloc` 时
根本分不出（也不需要分出）哪些是 transit、哪些是上一层溢出插入的。这是 §4 canonical form 的直接收益。

---

## 8. Spill 候选打分：`SelectSpillVictims`

因为整条 use-def 链在编译期完全已知（静态 shape、静态调度、没有运行时不确定性），这是一个 **offline**
问题——不需要 LRU 这类只能猜测未来的在线启发式，可以直接用"未来最晚才用到的先踢"（Belady 最优置换的
直觉）再叠加 buffer 大小做加权，这也是 vDNN/Capuchin 这类 GPU 显存 offload 系统的标准做法：

```text
score(v, p) = size(v) * distanceToNextUse(v, p)
```

- `size(v)` 越大，踢出去省的空间越多；
- `distanceToNextUse` 越大（离下次真正被用到还很远），说明"现在不用它、之后也不用马上取回来"，
  踢出去更划算。

两者都**抬高**分数，所以是相乘，不是相除——写成 `size / distanceToNextUse` 会让"离下次使用越远"
反而拉低分数，和想要的效果正好相反。§5.2 的算例里验证过（`c` 因为 `distanceToNextUse` 更大而被优先选中）。

> **已知缺陷**：`size` 放在分子这件事本身是可疑的。见 §12 表格中"cost model 只看 size 和复用距离"一行，
> 以及 §12.1。这一版先按上式实现，把它当作待实测校准的基线。

对 `window` 内每个仍然活跃的候选 buffer 算一次 score，从高到低贪心踢，直到 `occupancy(window) ≤ capacity`。
两个需要显式处理的边界：

- 若一个候选在 `window` 内没有"下次使用"（活跃区间恰好在 window 内结束），说明它本来就要死了，
  不需要真的插 spill/fill，等它自然死亡即可（图着色本身会处理）。所以候选集合限定为
  "活跃区间**跨过**整个 window 的 buffer"，天然排除了这种情况。
- 若把 window 内所有候选都踢完仍然放不下（单个不可再拆分的 buffer 比 capacity 还大，或者某个 op
  自己的操作数加起来就超容量），这不是 spill 能解决的问题，必须**直接报错**而不是 `goto loop` 重试
  （否则 §7 的外层循环会在 `victims` 为空集时死循环）。这类情形需要 tiling 而不是 spilling。

插入点选取（对应 §7 伪代码里的 `insertPoint`/`resumePoint`）：

- **spill 尽量早插**：紧跟在 `window` 之前那次真实使用后面，让空间尽快腾出来；
- **fill 尽量晚插**：紧贴 `window` 之后那次真实使用之前，让数据尽量晚一点占用回该层空间。

v0 不做"提前预取（prefetch）以和计算重叠"的调度优化——`fill` 就是这次使用前的同步阻塞搬运，
和搬运 op 的同步语义保持一致（异步搬运在 `NPU_LOAD_STORE_MEMDESC_PORTING.md` §8 里已被列为
v0 明确不做的范围）。

---

## 9. Clean buffer 不需要 store：canonical form 白送的一个优化

§4 的 canonical form 有一个副产品：**原始的 DRAM 源一直留在 IR 里**（`dram_alloc` / 函数参数本身
没有被消耗掉）。于是对一个"载入 L1 之后只读、从未被写过"的 buffer——典型就是权重——被选中 spill 时，
**根本不需要 `l1_store`**：直接放弃这块 L1 存储，等要用时重新从原来的 DRAM 源走一遍搬运链就行。

这就是寄存器分配里的 **rematerialization / clean-vs-dirty page** 区分：

| victim 状态 | 判据 | spill 动作 | 搬运量 |
|---|---|---|---|
| **dirty**（被写过） | 该 memdesc 值曾作为某个 op 的写目标（`outs`、`store` 的 `dst`…）出现 | `l2_alloc` + `l1_store`，之后 `l1_load` 回来 | 一写一读 |
| **clean**（只读） | 从未作为写目标出现 | **什么都不做**，只是不再占 L1 | 只有重新加载的一读 |

省掉一半搬运量，而且实现代价极低——判据就是"这个 `l1mem` 值有没有作为任何 op 的写目标出现过"，
是一次线性扫 use。对权重占主体的推理负载，这个优化的收益可能比 §8 的打分公式调优大得多。

一个特别有说服力的场景：变体 A（§3.6）下，L2 的 `AllocateTier` 有可能选中一个 **transit buffer**
作为 victim，把它 spill 到 DRAM——而它一条指令之前才刚从 DRAM 读进来。有了 clean/dirty 区分，
这种荒唐的往返自动消失（transit buffer 永远是 clean 的）。所以这不完全是"可选优化"。

v0 是否实现：建议实现，它比 §8 的任何调优都更简单、收益更确定。

---

## 10. 收敛性

外层 `loop` 每一轮要么直接收敛（`overflow` 为空），要么至少把 `window` 内一个 buffer 的物理占用时间
严格缩短（原来跨越整个原始活跃区间，现在被切成 spill 前 / fill 后两段更短的区间，中间那段完全不占用
该层空间）。插入 fill 不会在别处抬高本层峰值：`refillDst` 的活跃区间是原 victim 活跃区间的一个后缀
子集，同尺寸、更短。由于 buffer 数量有限，"每轮至少永久性缩短一个 buffer 在某个冲突窗口内的占用"
不能无限发生，所以外层循环必然终止。

唯一的死循环风险是 `victims` 算出空集而 `overflow` 非空——§8 第二个边界条件必须显式检测并报错。

三层之间的级联同样必然终止，因为溢出方向严格向下（L1→L2→DRAM），不存在环：L2 那一遍永远不会
需要往 L1 插东西。

工程上和 porting doc §5 提到的内层不动点循环一样，没有给出严格的收敛上界证明。

---

## 11. 循环（`scf.for`）内 buffer 怎么办

如果溢出窗口落在循环体内部、且被选中的 victim 是一个通过 `iter_args` 跨迭代传递的 buffer
（porting doc §4 提到的"alias 扩展让循环里的 buffer 区间正确覆盖整个循环"那种情况），
本设计**不做特殊分支**：spill/fill 直接插在循环体内部对应位置，结构上随循环体每次迭代都执行一遍。
这样得到的结果是**正确但不是最优**的（如果这个 buffer 其实在循环每一轮里都会立刻被用到，
每轮都 spill 再 fill 回来就是纯浪费的带宽）。把"循环不变的 spill/fill 提到循环外"是后续可以再加的
优化（类似 LICM），v0 先保证正确性。

**注意**：§7 伪代码里 `rewrite all uses of v at/after resumePoint` 这一步，只在 `resumePoint`
支配（dominate）所有待重写的 use 时才是良定义的。循环体内、以及 `scf.if` 分支里的 use 不满足这个条件。
这是本设计一个尚未解决的问题，见 §12.1。

---

## 12. v0 明确排除 / 简化的范围

| 内容 | 为什么先不做 |
|---|---|
| 跨级跳跃（L1 满了直接丢 DRAM，跳过 L2） | 级联顺序已经能覆盖正确性，跳级是性能优化，等有实测数据再决定。但见下面 cost model 一行——不跳级不只是"慢一点" |
| 异步 spill/fill（和计算重叠的 DMA prefetch） | 和搬运 op 的既有同步语义一致，等 NPU 异步搬运原语设计出来后再接 |
| DRAM 侧真正的空间复用 | `dram_alloc`/`dram_dealloc` op 本身存在，但 v0 里 DRAM 层的 `AllocateTier` 可以先是不做复用的 trivial bump 分配器。DRAM 容量相对宽裕，先接受这个浪费；真成为问题时直接把 §7 完整算法套到 DRAM 层即可，op/IR 层不用改 |
| 循环不变 spill/fill 外提（LICM 式优化） | 见 §11 |
| Spill 决策感知"计算耗时"来判断能否被 DMA 隐藏 | v0 cost 模型不建模计算-访存重叠时间线 |
| 带宽/bank/对齐约束 | `!spm.l1mem` 目前没有 encoding/layout 字段，occupancy 按字节总量计。这意味着"字节数装得下"不等于"真的放得下"（bank 冲突、对齐空洞），等硬件约束明确后再补 |
| cost model 只看 size 和复用距离 | 见 §12.1 —— 这一条是已知的**设计缺陷**，不只是"简化" |

### 12.1 已知的设计缺陷（待下一轮修订）

下面这几条是本轮修订之后仍然存在的问题，记录在此以免被当成已解决：

1. **`score = size × distanceToNextUse` 的 `size` 方向可疑。** 搬运一块 buffer 的代价大致正比于
   `size`，而它腾出的空间也正比于 `size`——代价/收益比在 size 上是常数。真正的判别式应该只有
   Belady 距离（和 §9 的 clean/dirty）。把 `size` 放在分子会系统性地偏向踢掉最大的 buffer：
   为了填 16K 的缺口去搬一块 1MB 的 buffer，产生 2MB 流量解决一个 16K 的问题。更合理的形式是
   "在能填上缺口的候选里，选 Belady 距离最远、且尺寸刚够的那个"。

2. **overflow 判据混淆了两件事。** §7 的 `OccupancyProfile(offsets, ...)` 量的是**图着色打包之后的
   高水位**，而 §5 的叙述用的是**并发活跃字节总和**（max-live-bytes）。前者 ≥ 后者，差值就是碎片。
   用前者做判据会把"碎片导致装不下"误判成"容量不够"，从而触发本不必要的 spill——而碎片应该用更好的
   打包或压缩来解决，不是 spill。判据应该基于 max-live-bytes（真正的下界），碎片单独处理。

3. **victim 可能是 view 而不是 seed alloc。** §8 没有规定候选集合必须是 seed `alloc`。如果一个
   `allocShape` 宽于 `shape` 的 view 被选中，"spill 它"实际上要 spill 整个底层 allocation 并重写
   所有 view。候选集合应该限定为 seed alloc。

4. **`resumePoint` 的支配关系没有处理。** 见 §11 末尾。victim 在 window 之后的 use 分散在
   `scf.if` 的不同分支、或跨循环边界时，不存在单一 `resumePoint`。需要改成"在所有后续 use 的最近
   共同支配点插 fill"，或者允许插入多个 fill。§11 声称循环场景"正确但不最优"——在 `iter_args`
   参与的情况下，这个"正确"还没有被论证过。

5. **victim 的下一次使用本身可能就是一次 store。** §5.2 里如果选中的是 `f`（下一次使用是返回路径的
   `l1_store %f, %f_l2`），那么"spill 到 L2 再 fill 回 L1 再 store 到 L2"是纯粹的浪费——正确做法是
   把返回路径的 store 直接提前，fill 整个删掉。更一般地：当 victim 的下一次 use 是一个到本次 spill
   目标层的搬运 op 时，两者应该合并。

6. **L1 pass 假设 L2 无限，导致代价估计失真。** L1 决定 spill 时不知道 L2 有没有空间。如果 L2 也
   紧张，L1 往 L2 扔 10 块、L2 再转手把 9 块扔到 DRAM，实际路径是 L1→L2→DRAM→L2→L1（四次搬运），
   而直接 L1→DRAM→L1 只要两次。正确性不受影响（§10），但 §8 的打分隐含假设"每次 spill 代价相同"，
   而真实代价取决于 victim 最终落在哪一层——这在 L1 pass 跑完之前不可知。这也是"不跳级"这条
   简化的真实成本，比"慢一点"严重。

7. **每轮 `goto loop` 重跑全套分析，编译时间是 O(窗口数 × 着色开销)。** 着色本身还有内层不动点。
   如果窗口很多，编译时间可能不可接受。一次迭代处理多个窗口、或增量更新干涉图，是需要的优化。

---

## 13. 建议实现顺序

1. `!spm.dram` 类型（照抄 `l1mem`/`l2mem` 字段）。
2. 三层 `*_alloc`/`*_dealloc`（六个 op）。`dealloc` 只做验证，不参与生命周期计算（§3.4）。
3. **改写**现有四个 load/store op 的语义为 tier-relative、无返回值（§3.1）。这一步会破坏现有 lit 测试，
   一并更新。
4. canonical form 的产生者（"tensor → SPM lowering" pass）：把计算图降到 §4.1 的四条不变量上。
   这一步不涉及任何分配决策，纯结构改写。
5. `SRAMAliasAnalysis`，按内存类型参数化，先只覆盖 L1（4 条规则 + alloc = seed）。
6. Liveness + 图着色分配（porting doc §4-§5），先跑通"L1 假设容量无限"的版本，验证和 Triton 行为一致。
7. 接入真实容量：`OccupancyProfile` + overflow 检测，先只做到"检测到超容量就报错"，确认判据正确。
   **在这一步就把 §12.1 第 2 条（max-live-bytes vs 打包高水位）定下来**，因为它决定判据本身。
8. `SelectSpillVictims` + IR 重写（§7-§8），把上一步的"报错"换成"自动插入 spill/fill 再重试"，
   先只验证 L1 这一层单独工作。同时实现 §8 第二个边界条件（victims 为空则报错），避免死循环。
9. §9 的 clean/dirty 区分。它简单、收益确定，而且是变体 A 下避免 transit buffer 荒唐往返的前提。
10. 把 §7 的三层级联接起来（L1→L2→DRAM），复用第 5-9 步已经跑通的单层逻辑。`!spm.dram` 到这一步
    才第一次真正被驱动到。
11. 循环场景（`scf.for` 里的 buffer 溢出）：先写测试暴露 §12.1 第 4 条的支配关系问题，再决定怎么修。

每一步都能独立编译、独立写 lit 测试验证，出问题时容易定位是"分配算法本身"还是"溢出改写逻辑"的问题。
