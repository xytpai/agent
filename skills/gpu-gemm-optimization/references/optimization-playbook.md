# GEMM 优化技巧速查：先诊断，再选杠杆

这不是推荐全部一起修改。每行都是一个“现象—证据—实验—风险”单元。

| 现象 | 先查证据 | 最小实验 | 回归风险 |
|---|---|---|---|
| 小 M 很慢，CU 空闲 | output tile 数、active waves、grid | 小 tile / split-K | 原子/初始化成本、partial 精度 |
| 大 tile 反而慢 | VGPR/LDS 分配档位、spill、waves | 减 repeats/stages 或 tile | reuse 下降、memory traffic 增加 |
| LDS 读等待很高 | lane 地址、bank conflict PMC | K permutation / swizzle 配对 | A/B K 对应关系、其他 layouts |
| 每轮 VALU 地址指令很多 | loop ASM，uniform vs lane | uniform K offset 用 soffset | offset 单位/范围、K-major stride |
| 高层表达有 rmem | regmem-to-SSA 与 scratch ISA | SSA/vector/fragment 重排 | loop-carried state 漏传 |
| 显式 scale 常量占寄存器 | intrinsic/ISA 编码 | 已证明等价的默认无scale状态 | 不同 dtype/op_sel/架构不等价 |
| MFMA 间等待或重复 operand | LDS issue 顺序与寄存器依赖 | traversal / serpentine | register pressure、不同 repeat geometry |
| 主循环 launch? branch 多 | 实际回边与 K sweep | constexpr 小倍数 unroll | code size、I-cache、VGPR、remainder |
| 内存 latency 未隐藏 | DMA/MFMA overlap、wait ledger | stages / load batch 边界 | LDS 容量、计数失效、提前覆盖 |
| cache 带宽效率差 | L2/DRAM、block map、重复 A/B | group_m/XCD swizzle | 小 grid、非整除 grid、局部热点 |
| epilogue 很长 | last MFMA 后指令、scale load、C stores | scale reuse、向量convert/store | rounding 顺序、M/N guard、跨wave交换 |
| 大 K 有空隙、尾部全串行 | trace/counter、最后几组 MMA | quadrant writeback overlap | barrier phase、AB/C union lifetime |
| split-K 加速不明显 | atomics、init/reset、partitions | 调 split 数或减少同步流量 | 可见性、重放 races、scratch lifetime |
| Graph 快、eager 慢 | API/memory trace、host墙钟 | cache dispatch、减少host复制/校验 | 动态 stride/stream/cache key 错误 |

## 1. Tile / waves / MMA geometry

先算静态资源和工作量：
~~~text
block_threads = m_waves * n_waves * k_waves * wave_size
A+B staged LDS ≈ stages * (block_m + block_n) * block_k * input_bytes
C-shuffle LDS ≈ k_waves * block_m * block_n * shuffle_bytes
union 复用通常取 max，不复用通常取 sum（看真实 allocator）
M repeats = block_m / (m_waves * mma_m)
N repeats = block_n / (n_waves * mma_n)
K repeats per slice = block_k / (k_waves * mma_k)
~~~

HTI 用半 tile 算 repeats/每 batch DMA 数。整数覆盖、thread vector覆盖与
shared-memory union alignment 都要验证，不只检查 divisibility 一个条件。
MMA32 并非必定比MMA16快：K深度、每thread输出、tile shape、LDS map不同。

改变 tile policy 与优化“老 kernel”分开评估。若只换 policy，明确是调参收益，
不要称 ASM 根因修复。保留原 policy 的性能结果。

## 2. G2S → LDS → registers → MFMA

- G2S coalescing 看实际物理 stride 和 transaction；logical NT 本身不是证明。
- MMA K packing必须符合 atom ABI。lane-group与per-lane value分别占哪些 K bits？
- 无冲突 LDS 不代表 global load也好；swizzle要同时检查写/read双方。
- Uniform offset 合到 soffset 能减少每lane v_add，但 K-major 时 offset须乘leading
  stride，byte/element单位和descriptor边界也必须匹配。
- 某些 async DMA 对 m0/wave LDS base有时序要求；compiler插入s_nop不一定可删。
- 不知道 barrier 前后 outstanding 指令数时，不能修改手工vmcnt。
- 避免独立维护两份长 loader：有证据收益后只给相关分支加最小 specialization。

## 3. MFMA schedule / compiler register allocation

- KNM/KMN/serpentine 改变哪个operand被连续复用、哪些 LDS read优先被消费；
  所有 permutation都相同 FLOPs，但 dependency chain不同。
- ISA看到数百个 `v_mov`，先区分零初始化、epilogue打包、循环内搬运；
  不能把文件级总数当作每K轮额外开销。
- 过早 load所有A/B fragment会提高 live range；按M/N/K chunk consume可能省寄存器。
- `sched_barrier` 既可能保住手工pipeline，也可能阻止调度器隐藏latency。
  只在已知需要的batch边界留；删除后必须重验wait/ASM。
- Unroll收益不是“分支少了”一句话：需要看code size、I-cache、live ranges和残段。
- 寄存器上限/occupancy hint不是免费的优化；避免强制limit导致scratch spill。

## 4. PTPC epilogue

通常 scale_b可跨M repeats复用，scale_a可跨N repeats复用。
但把所有scale提前加载会延长live range，还可能影响手工vmcnt统计。
从“循环外复用”和“加载位置”分别做消融。

向量FP32缩放→BF16转换→C-shuffle store可能比逐元素流程更友好，
也可能被compiler还原。检查实际vector convert/pack和LDS store，不靠源码判断。

K越短 epilogue/launch固定成本占比越大。大的overlap改动可能增加barrier，
即使8192³变快，256³也会回退。需要fallback时必须有明确适用范围与额外key成本。

## 5. Split-K vs slice-K

| 模式 | 并行位置 | 通信 | 常见收益 | 常见代价 |
|---|---|---|---|---|
| split-K | workgroup间 | global atomic或单独reduce | 输出tile少时增加grid | scratch/launch/init、原子争用、reduction精度 |
| slice-K | workgroup内wave间 | LDS + barrier | 减少每wave K、填充wave并行度 | 多份C、LDS容量、local reduction |
| 组合 | 两者 | 两者 | 特定shape提升并行度 | 最多同步、最复杂精度/lifetime |

先固定输出tile规模，再扫split数；并行度足够后继续增加split通常只添开销。
slice-K每wave直接读自身 K范围可比逐slice分支更简单，
但动态索引必须落在LDS源而不是非法动态register索引。编译器支持要实测。

## 6. Scope control / 最小改动

最终补丁应写清：
~~~text
适用：HTI && NT && MMA=16x16x128 [必要的其他条件]
不改：full-tile、MMA32、非NT、split-K协议、公共ABI
证据：单变量A/B + counter/ISA + 对应所有路径tests
~~~

如果泛化条件没有验证，不要顺手推广。如果用户要通用实现，
不能永久删除旧feature以维持快路径；用清晰参数与specialization保留功能。
优化报告可以说“主因已修复、尚差2us”，不能靠巨大rewrite隐藏剩余差距。
