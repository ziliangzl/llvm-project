# SPM SRAM 静态分配 + 三级溢出（Spilling）设计

本文承接 [NPU_SRAM_ALLOCATION_PORTING.md](NPU_SRAM_ALLOCATION_PORTING.md)（Triton 图着色分配算法的可移植部分）和
[NPU_LOAD_STORE_MEMDESC_PORTING.md](NPU_LOAD_STORE_MEMDESC_PORTING.md)（memdesc 类型 + load/store op 设计），
是原创设计部分——两篇 porting 文档都明确指出 "spill-to-DRAM 是 Triton 原设计里没有、需要自己另外设计的部分"。

设计目标：在 L1 / L2 / DRAM 三级存储之上，设计一个把"图着色静态分配"和"容量不够时溢出到下一级存储"
结合起来的算法，覆盖 **L1 → L2 → DRAM** 三级级联溢出。

---

## 0. 修订历史与本轮改动

### 第二轮（已并入）

| 上一版 | 本版 | 理由 |
|---|---|---|
| 新增 `l1_spill`/`l1_fill`/`l2_spill`/`l2_fill` 四个 op | **全部删除**。spill 就是 `store`，fill 就是 `load` | 签名和语义完全同构，保留两套名字是纯冗余（§3.3） |
| load/store 语义是 buffer ↔ tensor | load/store 语义是 **tier ↔ tier** | 计算 op 直接消费 memdesc；改成 tier-relative 之后 spill/fill 才能被 store/load 吸收（§3.1） |
| 隐含假设"数据可以停在任意一级" | 引入 **L1-resident canonical form** | 消灭 placement 自由度，让 L2 需求变成 spilling 的派生量（§4） |
| "alloc 和 dealloc 都只是可读性糖" | 只对 dealloc 成立 | §3.4 |

### 第三轮（本轮）

三个此前开放的问题现在有了结论：

- **`dealloc` 走"纯注释"路线**：liveness 计算必须显式排除 `*_dealloc` 的 operand，`dealloc` 因此
  **不具备提前释放能力**（§7.2）。
- **L1 不能直连 DRAM**（硬件约束）：所有 L1 的 load/store 必经 L2，§3.6 的"变体 B"取消，级联是唯一形态。
- **`maxLive` 本身要多精确**仍然开放，但"判据用打包高水位还是 maxLive"这一半已由 §9.4 的增量论证关闭。

本轮的实质内容是两处抽象层级的提升，它们各自回答一个具体问题：

| 问题 | 原设计的做法 | 本轮的做法 |
|---|---|---|
| spilling 到底是什么？ | 一次"插入 store + 插入 load"的 IR 改写 | **home location + dirty 位的写回式纪律**（§5）。§9 的 clean/dirty、spill slot 合并、重复 spill 消除全部是它的推论 |
| "放不下"怎么判定？ | 图着色打包之后的高水位 > capacity | **可行性（maxLive）和打包质量分开判**（§6）。这也是增量分配能成立的前提（§9.4） |

以及两个新增的算法章节：**增量分析**（§9，回答"L2 要不要从头重做 liveness"）和
**L2 侧的复用质量**（§10，回答"L1 的 spill slot 会不会让 L2 无法复用"）。

---

## 1. 设计前提：现有的 op 和 type 都不是既定事实

`spm-mlir/include/SPM/SPMOps.td` 和 `SPMTypes.td` 里已经落地的东西不是本设计的权威参考，
**也不构成约束**。它们是在分配算法还没设计之前先搭出来的脚手架。

本文的立场是反过来的：**分配 + 溢出算法是主导，op 和 type 随算法需要随时重构**。凡是本文和现有
`.td` 冲突的地方，一律以本文为准，`.td` 跟着改。遇到"算法想要 A，但现有 op 是 B"时，正确的反应是
改 op，而不是给算法打补丁去适配 B。

---

## 2. 与 one-shot-bufferize 的衔接：为什么别名分析可以很"薄"

one-shot-bufferize 阶段一的核心不变量是：分析阶段结束后，"此后任何机械式的结构化改写都不会引入
RAW 冲突"。当阶段二把这份已经过冲突消解的 tensor IR 机械改写成 `spm.l1_alloc` + 计算 op 直接消费
memdesc 时，**每个 `alloc` 出发的 def-use chain 内部天然无冲突**——这是本设计能够直接照搬 Triton
`AliasInfo`（porting doc §3）那套"链内无需判断，只需追踪谁和谁共享物理存储"规则的前提。

职责划分：

- **bufferize 阶段一**：保证单条 def-use chain 内部不会有 RAW 冲突。
- **本文的 SRAM 分配器**：保证**不同 alloc 之间**不会被分配到重叠的物理地址/时间段，以及总占用
  超容量时如何把数据挪到下一级。

分配器的 alias 分析不需要重新证明"in-place 安全"，只需判断"这个 SSA 值是不是某个 alloc 的别名"。

---

## 3. Op 家族：tier-relative 的 load/store

### 3.1 命名约定：load = 进入本层，store = 离开本层

> `lN_load` 把数据**搬进** tier N（源在更慢的一层）；`lN_store` 把数据**搬出** tier N（目标在更慢的一层）。
> 名字里的层级永远指**本层**，也就是"离计算更近的那一端"。

| Op | 源 | 目标 |
|---|---|---|
| `spm.l1_load` | `!spm.l2mem` | `!spm.l1mem` |
| `spm.l1_store` | `!spm.l1mem` | `!spm.l2mem` |
| `spm.l2_load` | `!spm.dram` | `!spm.l2mem` |
| `spm.l2_store` | `!spm.l2mem` | `!spm.dram` |

**L1 侧的源/目标类型只能是 `!spm.l2mem`，不能是 `!spm.dram`**——见 §3.6。

直接后果：**"把 L2 里的数据读到计算域"这件事没有对应的 op，也不需要有**。计算 op（`linalg.*` 等）
只接受 `!spm.l1mem` 操作数和 `outs`，所有计算都发生在 L1 上（§4 canonical form 的第 1 条不变量）。

曾考虑过通用的 `spm.dma %src, %dst`（靠操作数类型区分方向，歧义为零）。没有采用，是因为每一层的
分配器需要"扫描属于本层的 op"来发现工作集，按层命名让这一步是纯 op 类型匹配。理由不强，如果实现时
发现 tier-relative 命名反复被读错，换成 `spm.dma` 是可接受的重构（§1）。

### 3.2 完整 op 表

| 阶段 | Op | 状态 | 语义 |
|---|---|---|---|
| 分配 | `spm.l1_alloc` / `spm.l2_alloc` / `spm.dram_alloc` | **新增** | 产生一块新 buffer，此时不带地址。**唯一开启别名链的 op，也是 `offset` 属性唯一的挂载点** |
| 释放提示 | `spm.l1_dealloc` / `spm.l2_dealloc` / `spm.dram_dealloc` | **新增** | 纯注释，**不参与生命周期计算，也不具备提前释放能力**（§3.4 / §7.2） |
| 跨层搬运 | `spm.l1_load` / `spm.l1_store` / `spm.l2_load` / `spm.l2_store` | **语义改写** | 见 §3.1。一律**无返回值**，写入一个调用方预先 alloc 好的目标 memdesc |

三层严格只和相邻层对话，v0 不支持跳级；§3.6 说明这条在 L1 侧是硬件约束而非取舍。

**Verifier 条件（补）**：`l1_load`/`l1_store` 的源和目标必须 `shape`、`elementType` 完全一致；
目标必须 `mutableMemory`。特别地，spill 往返必须**保型**——§5 里 fill 出来的新值类型必须和被 spill 的
victim 完全一致（含 `mutableMemory`），否则下游对 victim 的写会失效。

### 3.3 spill 就是 store，fill 就是 load

一旦 load/store 改成 tier-relative（§3.1），"为溢出专门设计的 op"和 store/load 的签名、语义、
验证条件就完全重合了：两者都是"把一块已有 buffer 的内容拷进另一块已有 buffer，不产生新值"。
因此不存在独立的 spill/fill op。

```mlir
// spill：L1 装不下 %c 了，把它挪到 L2 —— 这就是一次普通的 l1_store
%c_home = spm.l2_alloc : !spm.l2mem<64x64xf32>
spm.l1_store %c, %c_home : !spm.l1mem<64x64xf32> -> !spm.l2mem<64x64xf32>

// fill：要用回来了，先在 L1 显式 alloc，再拷回来 —— 这就是一次普通的 l1_load
%c2 = spm.l1_alloc : !spm.l1mem<64x64xf32>
spm.l1_load %c_home, %c2 : !spm.l2mem<64x64xf32> -> !spm.l1mem<64x64xf32>
// 后续所有原本用 %c 的地方，从这里开始改用 %c2
```

这么做换来两个性质，后面反复用到：

1. **溢出产生的 IR 和正常数据加载产生的 IR 在结构上完全不可区分。** 层与层之间的耦合被压缩成
   "上一层跑完之后 IR 里多了几个本层的 `alloc` + 搬运 op"，分配算法不需要任何"这是溢出 buffer"的后门。
2. **搬运 op 不产生 SSA 值**，因此不参与别名格的传播。这是 §9.2 里"L2 的别名分析可以纯增量更新"
   的唯一原因——如果 spill 返回一个值，它就是别名格里的一个新节点，可能跨 `scf` region 边界合并
   别名集合，逼着整个格重新传播。

目标必须显式预先 `alloc`，因为每一层的分配算法靠"扫描这一层所有 `*_alloc` op"发现分配对象
（porting doc §7 骨架第 2 步）。搬运 op 若自己"生出"目标 buffer，那块 buffer 就没有 op 可以承载
算出来的 `offset` 属性。

### 3.4 `alloc` 和 `dealloc` 的地位并不对称

**`dealloc` 是纯注释**（本轮定论，路线 (a)）：buffer 的死亡点完全由"最后一次真实 use"决定。
代价是 `dealloc` **不能**用来提前释放空间——这一点和 porting doc §2 里 Triton `local_dealloc`
声称的用途之一相悖，具体的取舍论证见 §7.2。整个分配算法可以在完全没有 `dealloc` op 的 IR 上正确工作。

**`alloc` 是 load-bearing 的**，有三个不可替代的职责：

1. 它是那个 SSA 值的 **def**——"一块物理 buffer"在 IR 里就是靠这个值来命名和传递的，别名分析的
   第一条 seed 规则就是它；
2. 它是 `offset` 属性**唯一的挂载点**；
3. 它是每一层分配器的**扫描入口**。

更关键的是：**计算产出的 buffer 根本没有对应的 load**。`%c` 作为 `linalg.matmul` 的 `outs` 操作数，
数据不是从任何地方搬进来的，而 DPS 语义又要求 `outs` 操作数事先存在——所以即使把 `alloc` 融进搬运 op，
计算输出仍然必须有独立的 `alloc`。既然无法统一消掉，统一保留才一致。

### 3.5 `!spm.dram` 类型

```tablegen
def SPM_DRAMType : SPM_MemDescTypeBase<"DRAM", "dram"> {
  // 字段完全镜像 l1mem/l2mem：shape, elementType, mutableMemory, allocShape
}
```

和 `l1mem`/`l2mem` 保持同一套字段、同一个 `ShapedTypeInterface` 基类，是第三个独立类型而非
"memref + address space attr"。

### 3.6 L1 不能直连 DRAM（硬件约束，非取舍）

**硬件不支持 DRAM ↔ L1 直达搬运。** 所有进出 L1 的数据必须经过 L2。因此：

- `spm.l1_load` 的源类型只能是 `!spm.l2mem`，`spm.l1_store` 的目标类型只能是 `!spm.l2mem`；
- canonical form 里每个 DRAM 输入都必须有 transit staging（§4.1 第 2 条）；
- L1 的溢出目标只能是 L2，不存在"L1 满了直接丢 DRAM"这个选项。

这条约束有一个容易被忽视的好处：**它让 §14.1 里"L1 假设 L2 无限所以代价估计失真"这个缺陷大幅降级。**
既然 L1 只有一个可能的溢出目标，"知道 L2 还剩多少"这个信息**在选择目标上没有任何用处**——没有
第二个目标可选。它只剩一个二阶影响：victim 的**选择**（不是目标）仍然和 L2 压力有关，因为如果
某个 victim 最终会被 L2 转手扔到 DRAM，它的真实代价是四跳而不是两跳。这个二阶效应需要 L2 lookahead，
v0 不做。

---

## 4. Canonical form：分配前所有数据一律显式落在 L1

### 4.1 定义

分配算法的输入 IR 必须处于 **L1-resident canonical form**，即满足：

1. **所有计算发生在 L1 上**。计算 op 只接受 `!spm.l1mem` 操作数和 `outs`。
2. **每个 DRAM 侧的值（函数参数、常量、`dram_alloc`）后面紧跟到 L1 的完整搬运链**：
   `l2_alloc` + `l2_load` + `l1_alloc` + `l1_load`。
3. **中间结果永不回落**。计算产出的 buffer 就留在 L1，不存在"这个中间结果先放 L2"的 IR。
4. **只有整个计算图 return 的时候才写回 DRAM**：`l1_store` + `l2_store` 级联下去。

也就是说，canonical form 是一个**假设 L1 容量无限**的 IR。它一定不满足真实容量约束，这是故意的。

### 4.2 为什么要这个假设

它把 **placement 这个自由度彻底消灭了**。canonical form 里不存在"这块数据放 L1 还是放 L2"的决策——
答案永远是 L1。所有存储层级决策被推迟到唯一一个地方：spilling。

于是整套流程和经典寄存器分配严格同构：

| 寄存器分配 | 本设计 |
|---|---|
| 假设所有值都在寄存器里 | canonical form：所有 buffer 都在 L1 |
| 构建干涉图 + 着色 | alias + liveness + 图着色 |
| 着色失败 | maxLive > capacity（§6） |
| 插入 spill code（store/load 到栈） | 插入 `l1_store`/`l1_load`（到 L2） |
| 栈槽复用（spill slot coloring） | L2 层再跑一遍同一个 `AllocateTier`（§10） |
| 重建干涉图、重新着色 | 增量更新 + 暖启动（§9） |

最后两行是这个 canonical form 真正的收益：**L2 的需求变成纯派生量**。L1 pass 跑完之后，L2 需要装的
东西（transit staging + L1 溢出的 home）已经全部以普普通通的 `spm.l2_alloc` op 的形式显式存在于
IR 里了。L2 的 `AllocateTier` 因此可以**原样**是同一个算法跑在一个更小的、派生出来的问题上。

### 4.3 为什么这个假设不悲观

liveness 仍然在最后一个消费者处杀掉 buffer，所以 canonical form 的 L1 峰值就等于**程序自然的
工作集峰值**，没有任何额外膨胀。canonical form 唯一引入的额外占用是 transit buffer，而 transit
只影响 **L2** 占用，不影响 L1：`l1_load %a_l2, %a` 执行时 L2 侧多一块 `%a_l2`、L1 侧就是 `%a` 本身。

反过来看也成立：一个"产出很早、消费很晚"的 buffer，在 canonical form 里会一直占着 L1 从而触发 spill；
而一个"聪明的 placement"会一开始就把它放 L2、要用时再取。这两件事的最终 IR 是**同一个**。所以
canonical form + spilling 不会比手工 placement 差，它只是把决策推迟到了信息最完整的时候。

### 4.4 常量策略

v0 规定：**所有常量一律先在 DRAM 落地**，然后走和普通输入完全相同的搬运链进 L1。不为常量开特殊路径。

理由：如果允许"小的标量 splat / 立即数直接在 L1 上就地物化"，就需要重新引入一个
tensor（或 scalar）→ `l1mem` 的物化 op，也就是 §3.1 刚删掉的那类 op，canonical form 第 2 条不变量
随之破掉。常量走 DRAM 会浪费一点带宽，但保住了"L1 上的每一块 buffer 要么是 `l1_load` 搬进来的、
要么是计算 op 产出的"这条二分法。

### 4.5 前置条件：tiling 已保证每一块 buffer 单独装得进 L1

§4.1 要求每个 buffer **整块**落在 L1。如果一块 buffer 本身就比 L1 大，canonical form 不是
"超容量"而是**不可满足**——spilling 按整块 buffer 操作，永远修不了它。

**因此 canonical form 隐含一条硬前置条件，必须显式声明：**

> 上游的 tiling / 循环分块已经把每一块单独的 buffer 切到能装进 L1，并且任意单个计算 op 的
> 所有操作数 + `outs` + scratch 之和也能装进 L1。

这条前提有一个不舒服的后果：**tiling 和存储分配之间有循环依赖。** tiling pass 必须知道 L1 容量；
而要选对 tile 尺寸，它还得知道同时活着几块 buffer——那正是本分配器算出来的东西。这是 tile 尺寸
选择和存储分配之间的经典鸡生蛋，本设计把它整个推到了上游。

v0 的破环方式（必须写死，否则两个 pass 会互相甩锅）：

- tiling 只保证**局部**约束："单个 op 的操作数 + outs + scratch ≤ `L1_TILING_BUDGET`"，
  其中 `L1_TILING_BUDGET` 是一个 < `capacityL1` 的常量（建议先取 50%，靠实测调）。
- **全局**约束（跨 op 的并发活跃总量）完全由本分配器负责，用 spilling 解决。
- 容量常量因此在两个 pass 里都出现，必须来自同一个单一数据源（同一个 target description），
  不允许各写一份。

如果 tiling 连局部约束都满足不了（单个 op 自己就超预算），那是 tiling 的硬错误，本分配器只负责
检测并报错（§11 边界条件二），不尝试修复。

---

## 5. Home location：spilling 的正确抽象

这一节是本轮最主要的结构性改动。原设计把 spilling 描述成一次"插入 store + 插入 load"的 IR 改写。
这个描述层级太低，导致三个本该是同一件事的东西被分别发明（或者被漏掉）：clean buffer 免 store、
重复 spill 的 slot 合并、以及 transit buffer 被荒唐地 spill 回 DRAM。

正确的抽象是**写回式缓存纪律**（和寄存器分配器的 spill slot 管理是同一个东西）：

### 5.1 每个 L1 buffer 携带两个分析属性

| 属性 | 含义 | 怎么算出来 |
|---|---|---|
| `home` | 这块 buffer 的内容在**更慢层**的一份有效副本（`!spm.l2mem` 或 `!spm.dram` 值），可以为空 | 见 §5.2 的规则 |
| `dirty` | 自从 `home` 建立以来，这块 buffer 有没有被写过 | 该 memdesc 值是否作为某个 op 的写目标（`outs`、`store` 的 `dst`…）出现过。一次线性扫 use |

这两个都是**分析结果，不是 IR 上的属性**——不需要新增 attr，不需要给算法开后门，`home`/`dirty`
完全由现有 IR 的 def-use 结构决定。这一点很重要，它保住了 §3.3 性质 1（溢出 IR 和正常 IR 不可区分）。

### 5.2 四条规则

```text
初始（canonical form）:
  DRAM 输入经搬运链进 L1 的 buffer   →  home = 那个 !spm.dram 值,  dirty = false
  计算 op 产出的 buffer              →  home = 空,                 dirty = true

spill(v):
  if v.dirty:                     # 必须先建立 home
     h = insert <slower>_alloc
     insert <tier>_store(v, h)
     v.home = h ; v.dirty = false
  # 无论 dirty 与否，spill 动作本身就是"不再占用本层空间"，不产生额外 op
  # clean 的 buffer 因此 **零搬运、零新增 slot**

fill(v) at p:
  w = insert <tier>_alloc before p
  insert 从 v.home 到 w 的完整搬运链 before p     # 可能跨多层，见 §5.3
  w.home = v.home ; w.dirty = false
  rewrite uses of v at/after p to w

任何对 w 的写:
  w.dirty = true ; w.home 失效（下次 spill 必须重新建立）
```

### 5.3 `home` 永远取**最深**的那一份有效副本

这是本节最有价值的一条推论，它同时解决了两个问题。

考虑输入 `a`：DRAM(`%in_a`) → L2(`%a_l2`) → L1(`%a`)。`a` 被选中 spill 时，有三种可能的 home：

| home 选择 | spill 代价 | fill 代价 | L2 占用 |
|---|---|---|---|
| 新建一块 L2 slot | 一次 L1→L2 写 | 一次 L2→L1 读 | 新 slot，活跃区间 = 整个 vacated 区间 |
| 沿用 transit buffer `%a_l2` | 零 | 一次 L2→L1 读 | `%a_l2` 的区间被撑长到整个 vacated 区间 |
| **`%in_a`（DRAM）** | **零** | 一次 DRAM→L2→L1（新建一块短命 transit） | **仅一块短命 transit** |

**第三个是对的。** 因为 `a` 是 clean 的，DRAM 里那一份始终有效，而 DRAM 副本在 canonical form 里
本来就一直留在 IR 里（`dram_alloc` / 函数参数没有被消耗掉）。取最深的 home 让**中间层的占用最小**——
这正是我们在为之挣扎的那一层。

推广成规则：**`home` 取该 buffer 内容当前有效副本中所在层级最慢的那一个。** 于是：

- **clean 的 L1 buffer（权重、以及任何 fill 回来但没被写过的 buffer）被 spill 时，
  完全不产生 L2 slot，也完全不产生搬运。** 只在 fill 时付一条搬运链。
- transit buffer 永远是 clean 的且 home = DRAM，所以 L2 的 `AllocateTier` 想 spill 一个
  transit buffer 时，动作是"什么都不做"——那个"一条指令之前才刚从 DRAM 读进来、又被写回 DRAM"的
  荒唐往返自动消失，不需要任何特判。

这条规则对 §10（L2 复用质量）的影响是决定性的，见 §10.1。

### 5.4 Spill slot 合并与重复 spill 消除

考虑一个 buffer 被 spill、fill、使用（**只读**）、再次被 spill。按 §5.2 的规则：第二次 spill 时
`dirty = false`（fill 之后没被写过），`home` 仍然是第一次建立的那个 L2 slot——**所以第二次 spill
零搬运、不新建 slot**。重复 spill 消除是规则的自然推论，不需要额外机制。

代价是那块 home slot 的 L2 活跃区间被撑到"第一次 spill 到最后一次 fill"的整个跨度，而不是两段
短区间。这是一个真实的取舍：

| | 一块长 slot（沿用 home） | 两块短 slot（每次重建） |
|---|---|---|
| 搬运流量 | 一次写 | 两次写 |
| L2 峰值压力 | 高（长区间和更多东西冲突） | 低 |

v0 选**沿用 home**（省流量），理由是 L2 相对 L1 宽裕，而 DMA 流量是 NPU 上更稀缺的资源。如果实测
发现 L2 压力才是瓶颈，改成"L2 压力高时主动放弃 home、重建短 slot"是一个局部决策，不影响其他部分。

反过来，如果 fill 之后**被写过**（`dirty = true`），home 失效，第二次 spill 必须新建 slot——
这是正确性要求，没有取舍空间。

---

## 6. 判据：可行性（maxLive）和打包质量必须分开判

这是本轮第二处结构性改动。原设计的 `overflow = { p : occupancy(p) > capacity }` 里，`occupancy`
量的是**图着色打包之后的高水位**。这把两件性质完全不同的事混在了一起。

### 6.1 三个量的层级

对同一个程序点 `p`，有三个不同的数：

```
maxLive(p)              ≤   packedHighWater(p)          ≤   capacity?
（p 处活着的 buffer                （贪心打包之后
  尺寸之和）                        实际用到的最高地址）
```

- **`maxLive(p)`** 是**基本下界**：任何分配方案在 `p` 处都至少要占这么多。它只依赖 liveness，
  **和 offset 完全无关**。
- **`packedHighWater(p)`** 是当前打包方案的实际占用。两者之差就是碎片。porting doc §5 明确指出
  Triton 那套 `calculateStarts`/`buildInterferenceGraph`/`allocate` **完全不感知容量**，只管摆紧凑，
  所以它给出的是一个可行解，不是最优解。而 offline dynamic storage allocation 本身是 NP-hard 的，
  贪心的近似比是常数倍而不是 1——**这个差距是真实存在的，不是实现 bug。**

### 6.2 两阶段触发

于是判据必须分两级：

```text
if maxLive(p) > capacity  for some p:
    # 基本不可行：任何打包方案都救不了，必须 spill
    → SPILL（§11）

elif packedHighWater(p) > capacity  for some p:
    # 可行但当前打包不够好
    → 先 REPACK（冷启动重跑 calculateStarts，§9.1）
    → 重打包后仍然超：这是贪心近似比造成的真实失败
      → 此时才允许 SPILL，并在诊断里明确标注"因碎片而 spill"

else:
    → 收敛，贴 offset
```

**为什么不能只用 `packedHighWater`：**

1. **它会为碎片而 spill。** 用 DMA 流量去解决编译器打包不够紧的问题，是拿带宽买编译器的懒。
2. **它让增量分配不可能。** 这是更硬的理由，见 §9.4——`packedHighWater` 在暖启动下根本不会下降，
   算法会陷入"spill 了但判据没改善 → 继续 spill"的死循环。

**为什么不能只用 `maxLive`：** 它是下界，`maxLive ≤ capacity` 不保证真的摆得下。必须在最后
真正贴 offset 时用 `packedHighWater` 做一次硬校验。

### 6.3 待讨论：`maxLive` 本身要多精确

`maxLive` 怎么算，直接决定会不会有虚假 spill。porting doc §4 的做法是：后序编号之后取一个值
活跃的所有 op 的 **min/max ID**，得到区间 `[minId, maxId)`。这个"区间塌陷"引入两层过估计：

| 过估计 | 机制 | 例子 |
|---|---|---|
| **① 空洞塌陷** | buffer 在 t1 用、t100 再用、中间全程不活跃，min/max 把它算成 `[t1,t100)` 全程占用 | 任何"产出早、消费晚"的 buffer |
| **② 分支互斥丢失** | `scf.if` 两个分支里的 buffer，区间都落在 if 的 ID 范围内 → 判定相交 → 不能共享 | 任何有分支的图 |

关键观察：**这两层过估计完全是"塌陷成区间"这个动作的产物，不是 `mlir::Liveness` 的问题。**
`mlir::Liveness` 本身是 CFG-aware 的、逐 op 精确的；只在把逐-op 活跃集合压成 `[min, max]` 时才丢信息。

于是选项空间比想象的小：

| 方案 | maxLive 精度 | 对 porting doc §5 打包算法的影响 |
|---|---|---|
| **A：`[minId, maxId)` 区间** | 含 ①② 两层过估计 | 零。原样照搬 |
| **B：精确活跃 op 集合**（post-order ID 上的 bitvector） | 精确 | **大改**。`calculateStarts` 的空闲槽 multimap 假设区间连续，非连续区间要重写 |
| **C：判据用精确集合，打包仍用区间** | 精确 | **零** |

**推荐 C，而且它几乎是免费的：** `maxLive` 根本不需要连续区间——它就是"对每个程序点，把此刻活着的
buffer 尺寸加起来"，一次遍历、用逐-op 活跃集合直接算，完全不碰打包算法。打包继续用保守区间
（区间保守只会浪费空间，不会算错），浪费掉的部分由 §6.2 的第二阶段（repack）兜住。

**仍然开放的是一个策略问题，不是精度问题：** 当 `maxLive ≤ capacity` 但重打包之后
`packedHighWater` 仍然超容量时（§6.2 第二个分支的末路），该怎么办？

- (i) 投入更好的打包算法（有更优的 DSA 近似算法，但要自己写，不能照搬 Triton）；
- (ii) 就当不可行，spill（简单，但确实在为碎片付带宽）；
- (iii) 直接报错，让上游 tiling 调小（把问题甩回 §4.5 那个循环依赖）。

这一条需要继续讨论，v0 建议先 (ii) 且在诊断里明确区分两种 spill 来源，先积累"到底有多少 spill
是碎片造成的"这个实测数据，再决定值不值得投 (i)。

---

## 7. 每层的 Alias / Liveness

### 7.1 零新增别名规则

porting doc §3 的四条别名规则（seed / view 透传 / select 取并 / 其他一律断言）**原样保留、一条都不用加**，
按内存类型分层独立跑（L1 的分析只看 `!spm.l1mem` 值，L2 的只看 `!spm.l2mem` 值，互不干扰）。

原因：所有搬运 op 都是没有返回值的纯拷贝，根本不产生新 SSA 值，**不参与别名格的传播**——它们只是
对某个已有 alloc 值的一次"使用"。真正"开启新别名链"的永远只有 `*_alloc`，而且为溢出新插入的 alloc
和 canonical form 里原生的 alloc 在 IR 里毫无区别（§3.3）。

这条性质在 §9.2 里被用来证明层间交接的别名分析可以**纯增量**更新。

### 7.2 `dealloc` 必须从 liveness 里显式排除

**这是一个必须主动处理、否则静默灾难的实现陷阱。**

`spm.l1_dealloc %c` 的 `%c` 是一次货真价实的 operand use。`mlir::Liveness` 会因此把 `%c` 的活跃
区间**延长到 dealloc 那一点**。也就是说，在"原样用 `mlir::Liveness`"的实现下，`dealloc` 只可能让
区间变长，永远不可能变短——和 porting doc §2 里它声称的用途之一（"提前腾出空间"）正好相反。

最坏情况很安静也很致命：如果 canonical form 的 lowering 图省事，把所有 `l1_dealloc` 都堆在函数末尾
（一个完全合理的保守选择），那么**每个 buffer 都活到函数结束，图着色一次复用都做不了**，`maxLive`
直接变成所有 buffer 尺寸之和，然后触发一大堆莫名其妙的 spill。

**本设计选路线 (a)：liveness 计算时显式跳过所有 `*_dealloc` op 的 operand。** 具体地，在把
`mlir::Liveness` 的逐-op 活跃集合汇总成 buffer 区间时，遇到 `*_dealloc` 一律不计入。

相应的取舍必须承认：

- `dealloc` 因此**彻底没有提前释放能力**，它就是一条注释 + 一个"此后再用是 UB"的验证钩子。
- 放弃 porting doc §2 提到的"用户显式提前释放"这个用途。这是有意的：那个用途的价值远小于
  "忘插 dealloc 或插晚了会静默改变分配结果"这个风险，而 canonical form 是机器生成的、
  没有人类用户会手写 dealloc。
- 与此对应，`*_dealloc` op 应当在验证阶段检查"其后没有该值的真实 use"，把它的作用限定在纯校验。

**推论**：整个分配算法在完全删掉所有 `dealloc` op 的 IR 上必须给出**逐位相同**的结果。这应该作为
一条 lit 测试写下来（同一个输入，有/无 dealloc 两个版本，比对 offset 输出），它是路线 (a) 的可执行定义。

### 7.3 Scratch buffer

porting doc §4 有第三类 buffer：op 自己的临时工作区，区间是 `[opId, opId+1)`。canonical form 里
它们就是活跃区间极短的 `l1_alloc`，**不需要任何特殊处理**——这正是"统一保留 alloc"（§3.4）的又一个
好处。但两处必须把它们算进去：

- §4.5 那条前置条件（单个 op 的操作数 + outs + **scratch** ≤ 预算）；
- §11 边界条件二的硬错误检测。

它们不应该成为 spill victim（区间不跨窗口，天然被 §11 的候选规则排除）。

---

## 8. 核心算法：单层 `AllocateTier`

```text
AllocateTier(IR, tier, capacity, nextTier):
  state = FullAnalysis(IR, tier)          # alias + liveness + 打包，见 §9.1
  loop:
    if exists p: state.maxLive(p) > capacity:
      # ---- 基本不可行分支（§6.2）----
      if nextTier is null:
        error("buffer 在最后一级存储上仍然放不下，无法继续溢出")
      windows = maximal intervals where maxLive > capacity
      window  = earliest of windows
      victims = SelectSpillVictims(window, state, capacity)      # §11
      if victims is empty:
        error("窗口内没有可溢出的候选：单个 op 的工作集就超容量，需要 tiling 而非 spilling")
      for v in victims:
        ApplySpill(v, window, state)      # §5.2 的 spill/fill，§9.1 的增量更新
      continue

    if exists p: state.packedHighWater(p) > capacity:
      # ---- 可行但打包不够好分支（§6.2）----
      if not state.repackedThisRound:
        state.Repack()                    # 冷启动重跑 calculateStarts，§9.1
        state.repackedThisRound = true
        continue
      # 重打包之后仍然超：贪心近似比造成的真实失败
      diagnose("因碎片而 spill"); 按不可行分支处理（策略待定，§6.3）
      continue

    # ---- 收敛 ----
    stamp `offset` attribute onto every *_alloc op at this tier
    return
```

顶层驱动，严格级联、不跳级（§3.6）：

```text
AllocateTier(IR, L1,   capacityL1,   nextTier = L2)
AllocateTier(IR, L2,   capacityL2,   nextTier = DRAM)
AllocateTier(IR, DRAM, capacityDRAM, nextTier = null)   # v0 可以是 trivial bump 分配器（§14）
```

`repackedThisRound` 在每次 `ApplySpill` 之后要清掉——spill 改变了 liveness，值得再给打包一次机会。

---

## 9. 增量分析：内层循环与层间交接

这一节回答"L1 分配完、IR 被改了，L2 要不要从头重做 liveness"。**先纠正一个问题的前提**：
层间交接是这里最便宜的部分，昂贵的是 §8 的内层 `loop`。

- **层间交接只发生 2 次**（L1→L2、L2→DRAM），每次最多一次全量重算。
- **内层循环发生 O(#窗口) 次**，而每次都跑一遍完整的 alias + liveness + 打包不动点。

所以增量机制应该为内层循环而建，层间交接自然搭便车。

### 9.1 内层循环：一次 spill 的影响是 O(1) 的

关键观察：`ApplySpill(v)` 对本层 liveness 的影响极小。

| 对象 | 变化 |
|---|---|
| victim `v` | 活跃区间从 `[def, lastUse]` **截短**为 `[def, spillPoint]` |
| 新值 `refillDst` | 新增区间 `[fillPoint, lastUse]`（原区间的一个**后缀子集**） |
| **本层其他所有 buffer** | **活跃区间完全不变**（没有任何已有值的 use 被触碰） |
| 下一层 | 新增一个 `alloc`（clean victim 连这个都没有，§5.3） |

也就是说，一次 spill 的 liveness 更新是"**把一个区间劈成两段**"，是 O(1) 的工作，而不是一次
重新分析。不需要重跑 `mlir::Liveness`，不需要重跑别名分析（新 alloc 是一个 singleton 种子，
按 §7.1 纯增量加入）。

**干涉图更新**：`v` 只会**丢失**边（它不再和只与被删掉的中段重叠的 buffer 冲突），`refillDst`
是一个新节点需要连边。两者都是 O(#该点活跃 buffer)，不是 O(V²)。

**打包暖启动**：这里有一个很好的性质。porting doc §5 那套算法的最终产物是一组 offset，满足
"活跃区间相交的两块 buffer，offset 区间不相交"。**截短一个区间只会移除约束，不会新增约束——
所以旧的 offset 分配在截短之后仍然是一个合法解。** 因此不需要重跑 `calculateStarts`，只需要
给 `refillDst` 这一个新节点找位置，然后跑一遍 offset-推挤不动点。

`Repack()`（§8 里的冷启动）保留为一个显式动作，只在 §6.2 的第二阶段调用，以及在最终收敛前调用
一次拿到最紧的打包：

> **策略：搜索阶段用暖启动（关心的是可行性），最终答案前做一次冷启动重打包（关心的是紧凑度）。**

### 9.2 层间交接：为什么 L2 不需要重跑别名分析

L1 的 pass 对 IR 做的全部改动是：

1. 插入 `l2_alloc` op —— 引入**新的** L2 值；
2. 插入 `l1_store`/`l1_load` op —— 引入对 L1 值和 L2 值的**新 use**；
3. 重写 L1 值的 use。

对 L2 的分析而言：

- **别名分析：纯增量。** 新的 `l2_alloc` 各自是一个 singleton 种子；`l1_store`/`l1_load`
  **不产生任何 SSA 值**（§3.3 性质 2），所以它们不是别名格里的节点，无法把两个别名集合合并到一起，
  也无法跨 region 边界传播。结论：`L2 别名结果_new = L2 别名结果_old ∪ {新 alloc ↦ 自身}`，
  **不需要任何传播**。这是"搬运 op 无返回值"这个设计决定最实际的一次兑现。
- **每个既存 L2 值的 use 集合完全没变。** L1 的 pass 从不触碰 transit buffer 的 def 或 use。
- **既存 L2 值的逐-op 活跃集合也没变**，因为插入的 op 不切分基本块，只是插在既有 op 之间。
  一个跨过插入点的值现在也活跃在那些新 op 上——这是可以机械推导的，不需要重解数据流不动点。

**唯一真正失效的东西是后序编号**：插入 op 会让所有 ID 平移，于是所有 `[minId, maxId)` 区间的
数值都变了（尽管它们表达的语义没变）。

### 9.3 用带间隙的编号消除"插入即失效"

既然唯一失效的是编号，就把编号做成对插入稳定的：**给 post-order ID 留间隙**（比如全部乘以一个
步长，或者直接用 order-maintenance / list-labeling 结构维护相对顺序）。这样：

- 插入一个 op 时，给它一个落在两个邻居之间的 ID，**所有已有区间原封不动继续有效**；
- 间隙用尽时才做一次全局重标号（摊还 O(1)，实际上一个 kernel 里根本不会触发几次）。

配合 §9.1，一次 spill 迭代的分析开销就从"全量重跑 alias + liveness + 打包不动点"降到
"劈一个区间 + 改几条边 + 放一个新节点"。§14.1 里"编译时间 O(窗口数 × 着色开销)"那条缺陷由此关闭。

### 9.4 暖启动和判据的相容性——这是 §6 必须用 maxLive 的硬理由

暖启动有一个不能忽视的约束，它反过来锁死了 §6 的判据选择。

设想仍然用 `packedHighWater` 做溢出判据，同时暖启动打包。spill 掉 `v` 之后，`v` 的旧 offset
仍然留在其他 buffer 的分配里（暖启动不会去压缩地址空间），所以：

- 如果 `v` 原来占的不是最高那一块地址，**`packedHighWater` 根本不会下降**；
- 判据于是报告"还是超容量"，算法继续 spill，再次不下降……**死循环**。

而 `maxLive` 是 offset-无关的：截短 `v` 的区间之后，窗口处的 `maxLive` 必定下降 `size(v)`。
**只有 offset-无关的判据才能和增量/暖启动分配相容。** 这个论证独立于 §6.1 那些"别为碎片付带宽"
的理由，而且更硬——它不是质量问题，是终止性问题。

---

## 10. L2 侧的复用质量：spill slot 会不会毁掉 L2 的复用

这一节回答"L1 每次 spill 都临时生成新的 `l2_alloc`，会不会导致 L2 没法复用、或者复用不是最优解"。
结论：**会有一个真实的问题，但不是"新建 alloc"造成的，而是生命周期分布造成的；并且 §5.3 让问题
比想象的小得多。**

### 10.1 先看 L2 上到底有哪些 buffer

| 类别 | 数量 | 活跃区间长度 | 来源 |
|---|---|---|---|
| **transit-in** | O(#DRAM 输入) | 2 个 op（`l2_load` 到 `l1_load`） | canonical form |
| **transit-out** | O(#图输出) | 2 个 op | canonical form |
| **spill home（dirty victim）** | O(#dirty spill) | 至少跨过一个溢出窗口，可能很长 | L1 的 spilling |
| **spill home（clean victim）** | **0** | — | §5.3：clean victim 的 home 是 DRAM，**根本不建 L2 slot** |

最后一行是关键。按 §5.3，只有 **dirty** 的 victim 才会在 L2 上占一块 home。clean 的（全部权重、
所有"fill 回来只读"的 buffer）零 slot、零搬运。对权重占主体的推理负载，这一条把 L2 上的 spill
压力砍掉一大块。**所以"L1 spill 会不会把 L2 挤爆"这个担心，绝大部分被 §5.3 消化掉了。**

### 10.2 真实的问题：长短寿命混住导致碎片

剩下的 dirty home 有一个不好的性质：**它们和 transit buffer 的寿命差了两个数量级**
（跨窗口 vs. 2 个 op），而数量上 transit 远多于 home。

porting doc §5 的 `calculateStarts` 按 **size 从大到小**排序摆放，**完全不看寿命**。于是一块长寿命的
dirty home 很可能被摆在地址空间中间，把剩余空间劈成两半，后面成百上千个短命 transit buffer 只能
在碎片里挤——这是分配器里最经典的长寿命/短寿命混住问题。

注意这个问题的形态：它**不会**导致分配失败（`maxLive` 完全正常），只会让 `packedHighWater`
远高于 `maxLive`，然后被 §6.2 的第二阶段捕捉成"可行但打包不够好"。也就是说，判据分层
（§6）恰好让这个问题以正确的形式暴露出来——如果还用 `packedHighWater` 当溢出判据，
它会被误诊成"L2 容量不够"，进而触发 L2→DRAM 的虚假 spill。

### 10.3 解法：按寿命分区，而且不需要任何后门

修法是标准的**按寿命分离地址空间**：长寿命的从低地址向上摆，短寿命的从高地址向下摆（或者反过来）。

关键在于分类**不需要给算法开后门**。一个自然的想法是让 L1 的 pass 给它创建的 home 打个
`{spill_slot}` 标记——但那会破坏 §3.3 性质 1（溢出 IR 和正常 IR 不可区分），也就破坏了
"L2 的分配器分不出、也不需要分出哪些是 transit、哪些是溢出下来的"这个核心收益。

**不需要标记：寿命是内在属性，而且本来就已经算出来了。** 分类判据就是活跃区间长度（或者
"是否跨过任何一个 op 之外的东西"），是 liveness 的直接产物。L2 的分配器照旧只看到一堆
`l2_alloc`，只不过它现在按区间长度把它们分成两组分别摆。这条改进对 L1 层同样适用（L1 上
spill 回来的 fill 目标寿命也偏短），所以它是 `AllocateTier` 的通用改进，不是 L2 特供。

### 10.4 复用**是**最优解吗：不是，而且有一部分是结构性的

诚实的回答分三部分：

1. **同一个窗口里的多个 victim 一定互相冲突，这部分损失是不可避免的。** 为解决同一个溢出窗口而
   spill 出来的 k 个 home，活跃区间全都包含那个窗口，所以在 L2 上必然两两冲突、无法共享。
   这是正确的——它们持有互不相同的活数据。**L2 上 spill 部分的峰值需求 ≈ L1 的欠额**，
   这是设计的意图，不是缺陷。
2. **§5.4 的"沿用 home"选择主动加长了 L2 区间**，换取搬运流量。这是一个已知的、有意的取舍，
   不是最优解，见 §5.4 的对照表。
3. **victim 的插入点选择直接决定 L2 区间长度，而 §11 的规则已经为此优化过**：贴着窗口插
   （而不是"spill 尽量早、fill 尽量晚"）让每个 home 的 L2 区间尽可能短，这是对 L2 复用质量
   最直接的一个改善。见 §11 插入点小节。

所以"复用不是最优解"是对的，但可归因的部分基本都在 §5.4 和 §11 的取舍里，而不是在"临时新建
`l2_alloc`"这个动作上——那个动作本身是无害的，L2 的图着色对它和一个原生 alloc 一视同仁。

---

## 11. Spill 候选与打分：`SelectSpillVictims`

### 11.1 候选集合

候选必须满足：

1. **活跃区间严格包含整个 window**（不只是"相交"）。诞生于窗口内的 buffer 被排除——不仅因为
   spill 它没有意义，更因为算法对它**没有定义**：`insertPoint = 窗口之前的最后一次使用` 对一个
   在窗口内诞生的 buffer 根本不存在，无处可插。
2. **窗口之后还有真实使用**。否则它本来就要自然死亡，等着就行，图着色自己会处理。
3. **必须是 seed `alloc`，不能是 view。** 如果一个 `allocShape` 宽于 `shape` 的 view 被选中，
   "spill 它"实际上要 spill 整个底层 allocation 并重写所有 view。v0 直接把 view 排除在候选之外；
   如果一个底层 allocation 只通过 view 被使用，则以那个 seed alloc 为候选，spill 时连带重写它的
   所有 view（v0 可以先对这种情形报 unsupported）。
4. Scratch buffer 天然被规则 1 排除（§7.3）。

### 11.2 打分：`size` 会约掉，真正的判别式是 clean/dirty 加一个覆盖问题

上一版用 `score(v) = size(v) × distanceToNextUse(v)`。这个公式**两个因子都不是判别式**：

- **`size` 在收益和代价里同阶，约掉了。** 收益（腾出空间）= `size(v)`；代价（搬运流量）
  = `(dirty ? 2 : 1) × size(v)`。代价/收益比在 `size` 上是常数。把 `size` 放在分子会系统性地
  偏向踢掉最大的 buffer——为了填 16K 的缺口去搬一块 1MB 的 buffer，产生 2MB 流量解决一个 16K 的问题。
- **`distanceToNextUse` 既不影响代价也不影响收益。** 它在**寄存器**分配里重要，是因为那里会反复
  spill/reload 造成 thrashing；而这里 spill/fill 是一次性插在固定点的，代价就是一次 store + 一次
  load，与距离无关。任何跨过窗口的 victim 都同等地修掉这个窗口。

正确的形式是一个**覆盖问题**：

```text
deficit = max over p in window of (maxLive(p) - capacity)

在候选集合里选一个子集 S，使得
    sum{ size(v) : v in S } >= deficit
并最小化
    sum{ (v.dirty ? 2 : 1) * size(v) : v in S }        # 搬运流量，§5.3
```

贪心实现和它的直接推论：

1. **优先踢 clean buffer**——按 §5.3，clean victim 的搬运代价是 **0**（连 home 都不用建），
   它是免费的空间。**所有 clean 候选应该在任何 dirty 候选之前被考虑。**
2. **在 dirty 候选里，选总尺寸刚够填上缺口的那一组**，不是最大的那个。
3. **Belady 距离降级为 tie-break**：尺寸和 dirty 状态都相当时，选下次使用更远的那个——理由不是
   本窗口的代价，而是它更可能顺手把后面的窗口也一起修掉（从而省掉一整轮 spill）。

这个公式把 §5 的 home/dirty 模型和 victim 选择连了起来：**clean/dirty 不是一个可选优化，它是
cost model 的主要输入。**

### 11.3 插入点：贴着窗口，不要"尽量早/尽量晚"

上一版的规则是"spill 尽量早插、fill 尽量晚插，让空间尽快腾出来"。**这条是错的。**

超出窗口的那段 L1 空闲时间**收益是零**（窗口该修的已经修了），代价却是实打实的：spill home 在
**L2** 上的活跃区间被拉到最长（`[t6, t18]` 而窗口只是 `[t11, t12]`），直接恶化 §10 的 L2 复用质量。

正确规则：

- **spill 插在"窗口之前最晚的合法点"**（紧跟窗口前最后一次真实使用之后）；
- **fill 插在"窗口之后最早的合法点"**（紧贴窗口后第一次真实使用之前）；
- **唯一该扩大的情形**：victim 跨过多个窗口且中间没有真实使用——那时一次宽 spill 比两次往返便宜，
  应该把 vacated 区间扩到覆盖所有这些窗口。

一句话：**覆盖你打算一次修掉的那些窗口，不多不少。**

### 11.4 `resumePoint` 的支配关系（未解决）

`rewrite all uses of v at/after resumePoint` 这一步，只在 `resumePoint` **支配（dominate）**所有
待重写的 use 时才是良定义的。victim 在窗口之后的 use 分散在 `scf.if` 的不同分支、或跨循环边界时，
不存在单一 `resumePoint`。

正确做法是"在所有后续 use 的最近共同支配点插 fill"，必要时插入多个 fill（每个分支一个）。
**v0 先检测这种情形并报 unsupported**，不要生成可能不正确的代码。见 §14.1。

---

## 12. 完整示例

所有 buffer 都是 `64x64xf32` = 16KB，记一个"单位" = 16KB。为聚焦 L1，下面省略 transit buffer 的
`l2_alloc`（用注释代替），它们不影响 L1 占用（§4.3）。

**Canonical form（分配前）：**

```mlir
func.func @accum_across_burst(%in_a: !spm.dram<64x64xf32>, %in_b: !spm.dram<64x64xf32>,
                               %in_d: !spm.dram<64x64xf32>, %in_e: !spm.dram<64x64xf32>)
    -> (!spm.dram<64x64xf32>, !spm.dram<64x64xf32>) {
  // ---- 阶段一：c = a @ b ----
  %a = spm.l1_alloc : !spm.l1mem<64x64xf32>     // t1   home = %in_a, clean
  // DRAM->L2->L1: %in_a ==> %a
  %b = spm.l1_alloc : !spm.l1mem<64x64xf32>     // t3   home = %in_b, clean
  // DRAM->L2->L1: %in_b ==> %b
  %c = spm.l1_alloc : !spm.l1mem<64x64xf32>     // t5   home = 空, dirty
  linalg.matmul ins(%a, %b : ...) outs(%c : !spm.l1mem<64x64xf32>)          // t6

  // ---- 阶段二：完全不碰 c 的"突发"计算，f = d @ e ----
  %d = spm.l1_alloc : !spm.l1mem<64x64xf32>     // t7   home = %in_d, clean
  // DRAM->L2->L1: %in_d ==> %d
  %e = spm.l1_alloc : !spm.l1mem<64x64xf32>     // t9   home = %in_e, clean
  // DRAM->L2->L1: %in_e ==> %e
  %f = spm.l1_alloc : !spm.l1mem<64x64xf32>     // t11  home = 空, dirty
  linalg.matmul ins(%d, %e : ...) outs(%f : !spm.l1mem<64x64xf32>)          // t12
  // f 写回 DRAM（返回路径）
  %f_l2 = spm.l2_alloc : !spm.l2mem<64x64xf32>
  spm.l1_store %f, %f_l2 : ...                                             // t14
  %f_dram = spm.dram_alloc : !spm.dram<64x64xf32>
  spm.l2_store %f_l2, %f_dram : ...

  // ---- 阶段三：终于再用到 c ----
  %g = spm.l1_alloc : !spm.l1mem<64x64xf32>     // t17
  linalg.exp ins(%c : ...) outs(%g : ...)                                  // t18
  %g_l2 = spm.l2_alloc : !spm.l2mem<64x64xf32>
  spm.l1_store %g, %g_l2 : ...
  %g_dram = spm.dram_alloc : !spm.dram<64x64xf32>
  spm.l2_store %g_l2, %g_dram : ...
  return %f_dram, %g_dram : !spm.dram<64x64xf32>, !spm.dram<64x64xf32>
}
```

L1 上的 `maxLive`（§6.1；精确逐-op 活跃集合，§6.3 方案 C）：

| 程序点 | 事件 | 活着的 L1 buffer | `maxLive` |
|---|---|---|---|
| t5–t6 | `alloc c`、`matmul1` | a, b, c | 48K —— peak① |
| t6 之后 | a、b 最后一次使用已发生 | c | 16K |
| t11–t12 | `alloc f`、`matmul2`（**c 全程用不上，只是恰好还活着**） | c, d, e, f | **64K** —— peak② |
| t12 之后 | d、e 最后一次使用已发生 | c, f | 32K |
| t14 | `l1_store f` | c | 16K |
| t17–t18 | `alloc g`、`exp` | c, g | 32K |

peak② 的关键点：`matmul2` 本身只需要 d/e/f 三块同时在场（这正好等于 §4.5 那条局部约束的下限），
`c` 只是"活着但这段时间完全用不上"。**这正是可以被安全 spill 掉的那种 buffer。**

### 12.1 容量足够（capacity = 64K）

`maxLive` 峰值 64K = 容量，`packedHighWater` 也能做到 64K，直接收敛，只贴属性、不改结构：

```mlir
  %a = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
  %b = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>
  %c = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>
  %d = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // 复用 a 的位置
  %e = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>   // 复用 b 的位置
  %f = spm.l1_alloc {offset = 49152 : i64} : !spm.l1mem<64x64xf32>   // t11/t12 时 c,d,e 都活着
  %g = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
```

### 12.2 容量不够（capacity = 48K）

`maxLive(t11..t12) = 64K > 48K`，`deficit = 16K`，窗口 = `[t11, t12]`。

按 §11.1 筛候选：

- `d`、`e`：最后一次使用在 t12（窗口末尾），窗口后无使用 → **规则 2 排除**。
- `f`：t11 诞生，等于窗口起点，区间不严格包含窗口 → **规则 1 排除**。（顺带说明规则 1 为什么必要：
  `f` 没有"窗口之前的最后一次使用"，`insertPoint` 无处可放。上一版把 `f` 算进候选并给它打分，
  是一处自相矛盾。）
- `c`：区间 `[t5, t18]` 严格包含 `[t11, t12]`，窗口后在 t18 有真实使用 → **合法候选**。

**本例的合法候选只有 `c`**，不需要打分。按 §11.2 补一句它的代价：`c` 是 `matmul1` 的 `outs`，
**dirty**，所以必须建 home、付一次 store + 一次 load。（如果被选中的是一个 clean buffer，
比如权重 `a`，代价会是 **0** ——只需要不占 L1，fill 时从 DRAM 重走一遍搬运链，L2 上不留任何 home。）

按 §11.3 定插入点（**贴着窗口**）：`insertPoint` = 窗口前最晚合法点 = t6 之后；
`resumePoint` = 窗口后最早合法点 = t18 之前。

```mlir
  linalg.matmul ins(%a, %b : ...) outs(%c : !spm.l1mem<64x64xf32>)          // t6

  // >>> spill：c 是 dirty，先建 home（§5.2），再一次普通的 l1_store <<<
  %c_home = spm.l2_alloc : !spm.l2mem<64x64xf32>
  spm.l1_store %c, %c_home : !spm.l1mem<64x64xf32> -> !spm.l2mem<64x64xf32>
  // 此后 c.home = %c_home, c.dirty = false

  %d = spm.l1_alloc : !spm.l1mem<64x64xf32>
  ... // 阶段二不变
  spm.l2_store %f_l2, %f_dram : ...

  // >>> fill：一次普通的 l1_load，得到全新的 SSA 值 %c2（保型，§3.2）<<<
  %c2 = spm.l1_alloc : !spm.l1mem<64x64xf32>
  spm.l1_load %c_home, %c2 : !spm.l2mem<64x64xf32> -> !spm.l1mem<64x64xf32>
  // %c2.home = %c_home, %c2.dirty = false —— 若它之后再被 spill，零搬运（§5.4）

  %g = spm.l1_alloc : !spm.l1mem<64x64xf32>
  linalg.exp ins(%c2 : ...) outs(%g : ...)                                 // t18
```

按 §9.1 做增量更新，而不是重新分析：`c` 的区间从 `[t5, t18]` 截短为 `[t5, spillPoint]`，
新增 `%c2` 的区间 `[fillPoint, t18]`，**其他所有 buffer 区间不变**。窗口处 `maxLive` 降到 48K，
一次迭代就收敛。最终冷启动重打包一次（§9.1）拿到最紧的 offset：

```mlir
  %a      = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
  %b      = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>
  %c      = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>
  %c_home = spm.l2_alloc {offset = 0 : i64}     : !spm.l2mem<64x64xf32>   // L2 那一遍独立跑
  %d      = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>   // 复用 a
  %e      = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>   // 复用 b
  %f      = spm.l1_alloc {offset = 32768 : i64} : !spm.l1mem<64x64xf32>   // c 已 spill 走，复用 c
  %c2     = spm.l1_alloc {offset = 0 : i64}     : !spm.l1mem<64x64xf32>
  %g      = spm.l1_alloc {offset = 16384 : i64} : !spm.l1mem<64x64xf32>
```

L1 回到 3×16K = 48K。注意 `%c_home` 在 L2 上的活跃区间是 `[t6, t18]`——按 §11.3 贴窗口插已经是
最短的可能（`c` 在 t6 和 t18 之间确实没有别的使用了）。它属于 §10.1 的"spill home（dirty）"类，
寿命远长于同层的 transit buffer，正是 §10.3 要按寿命分区的对象。

---

## 13. 收敛性与分层不变量

### 13.1 单层收敛

外层 `loop` 每一轮要么收敛，要么至少把窗口内一个 buffer 的物理占用时间严格缩短。截短不会在别处
抬高本层 `maxLive`：`refillDst` 的活跃区间是原 victim 区间的一个**后缀子集**，同尺寸、更短，
所以逐点 `maxLive` 单调不增。由于 buffer 数量有限，外层循环必然终止。

三个死循环风险，都必须显式检测：

1. `victims` 算出空集而 `maxLive` 仍超容量 → §8 显式报错（单个 op 工作集超容量，需要 tiling）。
2. 判据用了 offset-相关的量 → §9.4 那个不下降的死循环。判据必须是 `maxLive`。
3. `Repack()` 反复被调用而不推进 → §8 的 `repackedThisRound` 标志，每轮最多重打包一次。

### 13.2 分层不变量（级联正确性的地基）

§8 三层顺序跑、L1 的结果不被后续 pass 破坏，靠的是一条此前从未写下来的不变量：

> **每一层的 `AllocateTier` 只允许插入触及本层及更慢层的 op，绝不触及更快的层。**

这条成立时，L2 的 pass 往 IR 里插 `dram_alloc`/`l2_store`/`l2_load` 完全不影响 L1 已经钉好的 offset
（它们不碰任何 `!spm.l1mem` 值），级联因此可以是单向的三次调用而不是不动点迭代。

它一破——比如某个未来的优化想在 L2 的 pass 里插一个 L1 双缓冲——级联就必须推倒重来变成
跨层不动点。所以任何新优化都要先对照这条不变量。

配合"溢出方向严格向下（L1→L2→DRAM）、不存在环"，三层级联必然终止。

工程上和 porting doc §5 的内层不动点一样，没有给出严格的收敛上界证明。

---

## 14. 循环（`scf.for`）内 buffer 怎么办

如果溢出窗口落在循环体内部、且被选中的 victim 是一个通过 `iter_args` 跨迭代传递的 buffer，
spill/fill 会插在循环体内部，结构上随循环体每次迭代都执行一遍。这样得到的结果**带宽上不是最优**
（如果这个 buffer 每轮都会立刻被用到，每轮 spill 再 fill 就是纯浪费）。把"循环不变的 spill/fill
提到循环外"是后续的 LICM 式优化。

但**正确性本身还没有被论证**：`iter_args` 参与时，"重写 `resumePoint` 之后所有 use"要连带重写
循环的 `iter_args` / `scf.yield`，而 `resumePoint` 在循环体内时不存在对循环外后续 use 的支配关系。
这是 §11.4 的具体化。**v0 对"victim 是 `iter_args` 携带的 buffer"这种情形直接报 unsupported**，
先把正确的东西做出来，再扩大覆盖面。

---

## 15. v0 明确排除 / 简化的范围

| 内容 | 为什么先不做 |
|---|---|
| 跨级跳跃（L1 直接丢 DRAM） | **不是取舍，是硬件约束**（§3.6）。L1 必须经 L2 |
| 异步 spill/fill（和计算重叠的 DMA prefetch） | 和搬运 op 的既有同步语义一致，等 NPU 异步搬运原语设计出来后再接。副作用：canonical form 是性能上最差的调度，所以 §11.2 的 cost model 在异步落地之前**无法被实测验证** |
| DRAM 侧真正的空间复用 | v0 里 DRAM 层的 `AllocateTier` 可以先是不做复用的 trivial bump 分配器。真成为问题时直接把 §8 完整算法套到 DRAM 层即可，op/IR 层不用改 |
| 循环不变 spill/fill 外提（LICM 式） | 见 §14 |
| `iter_args` 携带的 buffer 作为 victim | §14，v0 报 unsupported |
| 后续 use 不被单一 `resumePoint` 支配 | §11.4，v0 报 unsupported |
| view 作为 victim | §11.1 规则 3，v0 报 unsupported |
| 带宽 / bank / 对齐约束 | memdesc 类型目前没有 encoding/layout 字段，占用按字节总量计。"字节数装得下"不等于"真的放得下"（bank 冲突、对齐空洞），等硬件约束明确后再补。这会同时影响 `maxLive` 和打包 |
| 多执行引擎并发（porting doc §5 的 AsyncRegions 连边规则） | v0 单引擎。有多引擎时"活跃区间不重叠也要连边"的规则要接回来 |
| 跨函数 | canonical form 按"整个计算图"表述，存在 `func.call` 时 callee 的 L1 占用和 caller 活着的 buffer 没有协调机制。v0 前置条件是**全部内联** |
| Spill 决策感知"计算耗时"以判断能否被 DMA 隐藏 | v0 cost 模型不建模计算-访存重叠时间线 |

### 15.1 仍然开放的问题

1. **`maxLive` 的精度**（§6.3）。推荐方案 C（判据用精确逐-op 活跃集合，打包仍用保守区间），
   它几乎免费。**仍需讨论的是那个策略问题**：`maxLive ≤ capacity` 但重打包后 `packedHighWater`
   仍超容量时，选 (i) 更好的打包算法 / (ii) 就当不可行去 spill / (iii) 报错让 tiling 调小。
   v0 建议 (ii) 并在诊断里区分两种 spill 来源，先积累"多少 spill 是碎片造成的"这个数据。
2. **§5.4 的取舍**（沿用 home 省流量 vs. 重建短 slot 省 L2 压力）在真实负载上哪个对。
   v0 选沿用 home，等实测。
3. **victim 选择的 L2 lookahead**（§3.6 末尾的二阶效应）：如果某个 victim 最终会被 L2 转手扔到
   DRAM，它的真实代价是四跳而不是两跳。L1 目前看不到这一点。需不需要做取决于 L2 实际有多紧张。
4. **§4.5 的 `L1_TILING_BUDGET` 取多少**，以及它和分配器共享容量常量的具体机制。

---

## 16. 建议实现顺序

1. `!spm.dram` 类型（照抄 `l1mem`/`l2mem` 字段）。
2. 三层 `*_alloc`/`*_dealloc`（六个 op）。`dealloc` 只做验证（§7.2）。
3. **改写**现有四个 load/store op 为 tier-relative、无返回值，L1 侧源/目标限定为 `!spm.l2mem`
   （§3.1 / §3.6）。补 §3.2 的保型 verifier。这一步会破坏现有 lit 测试，一并更新。
4. canonical form 的产生者（"tensor → SPM lowering" pass）：把计算图降到 §4.1 的四条不变量上。
   纯结构改写，不涉及任何分配决策。
5. `SRAMAliasAnalysis`，按内存类型参数化，先只覆盖 L1（4 条规则 + alloc = seed）。
6. Liveness：`mlir::Liveness` + 别名扩展 + **显式排除 `*_dealloc` operand**（§7.2）。
   立刻写那条"有/无 dealloc 输出逐位相同"的 lit 测试——它是路线 (a) 的可执行定义。
7. **`maxLive` 计算**（精确逐-op 活跃集合，§6.3 方案 C）。这一步独立于打包，先单独测。
8. 图着色打包（porting doc §5）+ `packedHighWater`。先跑通"L1 假设容量无限"的版本，
   验证和 Triton 行为一致。
9. **两阶段判据**（§6.2）：`maxLive` 触发 spill、`packedHighWater` 触发 repack。先只做到
   "该 spill 的地方报错"，确认两级判据分得对——**这一步必须在任何 spill 代码之前做完**，
   否则后面所有 spill 测试都在验证一个错的触发条件。
10. **home / dirty 分析**（§5.1-§5.3）。它是 §11.2 cost model 的主要输入，必须在 victim 选择之前。
    包含"clean victim 零搬运零 slot"这条，以及 home 取最深副本的规则。
11. `SelectSpillVictims`（§11：候选规则 + 覆盖问题 + 贴窗口插入点）+ IR 重写。同时实现 §13.1 的
    三个死循环检测，以及 §11.1/§11.4/§14 的三处 unsupported 报错。先只验证 L1 单层。
12. **增量分析**（§9.1 + §9.3 的间隙编号 + 暖启动打包 + 收敛前一次冷启动重打包）。
    功能上不改变任何输出，只改编译时间——所以可以用"和非增量版本输出逐位相同"来测。
13. 把 §8 的三层级联接起来（L1→L2→DRAM），复用第 5-12 步的单层逻辑。`!spm.dram` 到这一步才第一次
    真正被驱动到。同时验证 §9.2 的层间纯增量别名更新。
14. **按寿命分区的打包**（§10.3）。这一步的收益应该能直接从"repack 触发次数"和
    `packedHighWater / maxLive` 的比值上看出来。
15. 循环场景：先写测试确认 §14 的 unsupported 报错真的触发（而不是"偶然"没触发），再决定
    要不要投入做支配关系正确的多点 fill。

每一步都能独立编译、独立写 lit 测试验证，出问题时容易定位是"分配算法本身"还是"溢出改写逻辑"的问题。
