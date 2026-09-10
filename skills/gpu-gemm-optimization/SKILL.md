---
name: gpu-gemm-optimization
description: Optimize and root-cause GPU GEMM kernels with a reproducible baseline, IR/ASM inspection, ROCm trace and hardware counters, controlled ablations, and correctness/performance gates. Use for FlyDSL, gfx950 FP8/BF16/FP16 GEMM, HTI, full-tile, split-K, slice-K, layout/swizzle, pipelining, or regression-free refactors.
---

# GPU/GEMM 优化 Skill

## 触发条件

用户要求：优化 kernel、解释性能差距、分析 ASM/trace、调整 GEMM pipeline /
swizzle / MMA / epilogue、简化代码但不得退化，或补全相关回归测试。

以本仓库 FlyDSL + ROCm/gfx950 为主要实例。方法适用于其他 GPU；
**指令语义、bank 分组、计数器单位与同步规则必须按目标架构重新确认**。

## 必须遵守的工作规则

1. **先保存当前工作区**，不是默认用 `git HEAD` 作基线。不覆盖未提交改动，
   不删除用户缓存、不偷偷改 GPU clock，不擅自 commit/push。
2. 先确定算子契约和目标路径，禁止以降低精度、删布局、屏蔽 split-K、
   路由到另一个 kernel 或特化输入内容来伪装“优化”。
3. 所有“已测到”结论必须有命令、原始数据和版本记录。区分：
   **假设 / 证据支持 / 实验证实 / 尚未验证**。
4. 改动一项、测一项。优先查硬件计数器和指令，再做大规模调参；
   不连续堆十几个优化后反推根因。
5. 不能仅凭 Python 行数、VGPR 数、静态指令数或一次计时作结论。
6. 不为通过测试扩大误差阈值。已存在的 bug 单独记录；
   本次引入的 bug 必须修复/回滚，不可加 xfail 隐藏。
7. “无退化”只针对**实际测试的 shape、路径、精度和测量模式**。
   有短 K/小 M 回退必须列出来，而不是只展示 8192³。

## 完整 SOP（按顺序执行）

### 0. 确认契约和成功标准

- 数学式、输入/输出 dtype、accumulator 与 partial dtype、scale 语义、
  bias 的位置、layout/stride、对齐、K-tail。
- 模式：full-tile/HTI、MMA shape、stages、wave map、split-K/slice-K。
- 用户关心 kernel latency、Graph latency 还是 host/API 端到端 latency？
- 目标 shape、回归 shape、可接受误差、性能噪声容忍度和资源约束。
- 问清“老 kernel”指哪个当前文件/commit/config。同名 kernel 不一定同代码。

**输出**：contract + benchmark matrix + acceptance gates。

### 1. 保存环境与可回退基线

- `git status --short`；复制 kernel、共享 utils、tests、dispatch/policy。
- 记录 source SHA、git commit、dirty diff、Python/Torch/FlyDSL/LLVM/ROCm、
  GPU arch、可见设备到物理设备映射、clock/power/温度、并发任务。
- baseline/candidate 各用独立 JIT cache **和独立 IR dump 目录**。
- 涉及 shared utils/compiler 改动时，两份 module import 不算隔离：
  使用单独 worktree/源码快照/环境，确保 baseline 不 import candidate helper。

**输出**：manifest、不可变 baseline、复现命令。见 [测量协议](references/measurement.md)。

### 2. 复现差距并先通过基线正确性

- 同 GPU、同 shape/layout/stride/policy、同输入分布、同输出精度。
- preallocate → 首次编译/launch → 显式 warmup → 同步 → 计时。
- A/B 使用相同输入；保持 Graph、hot/rotary 模式一致。
- baseline 错误先列清；不要把错误的高速结果当优化目标。
- 不相同数学语义的两条路径只能比较各自算法，不是等价 A/B。

**门禁**：复现不了稳定差距，先修测量，不改 kernel。

### 3. 先分类瓶颈

- trace：host gap？多了 launch/copy/contiguous/alloc？隐式 synchronize？
  Graph 是否消除了用户真实关心的开销？
- launch：grid/wave/LDS 是否限制 occupancy？尾块浪费？小 M 并行度不足？
- kernel：计算吞吐、global memory、LDS bank conflict、寄存器 spill、
  指令 issue/依赖、barrier、epilogue？
- 固定 M/N 扫 K；固定 K 扫 M/N。K 斜率提示 steady-state，截距提示
  prologue/epilogue/launch，但 cache/occupancy 变化时不能机械线性归因。
- `2*M*N*K` 是标准 GEMM FLOPs；FP8 字节数、partial scratch、重复 tile
  加载和 cache reuse 要分别算。Roofline 是边界估计，不是根因证明。

**输出**：2–3 个可证伪假设，每个写“预计哪条指令/哪个 counter 怎样变”。

### 4. 建立源码 → IR → ASM 对应

依次查：
1. 实际 dispatch、grid/block、kernel symbol、MMA encoding。
2. LDS/VGPR/SGPR/scratch/spill，按架构分配粒度判 occupancy 档位。
3. prologue、主循环、remainder、drain、epilogue，标注回边和展开因子。
4. normalized per-K-tile 的 MFMA、LDS、DMA、VALU/SALU、wait/barrier。
5. lane 地址、copy/MMA layout、K packing、C-shuffle ownership 和依赖链。
6. 高层不同写法是否在 regmem-to-SSA 后变成同一机器代码。

使用 [ASM/IR 分析](references/asm-ir.md) 和 `scripts/asm_summary.py`。
脚本仅给**候选循环/统计**，不是反汇编器、CFG 或性能模型。

### 5. trace / 硬件计数器证实

- trace 用于 dispatch/host timeline；PMC 用于 kernel 内资源与 issue。
- 先查询本机可用工具/counter 与单位，再选少量可同时采集的 counter。
- 过滤目标 symbol/iteration；baseline/candidate 用相同 pass、workload、
  cache 条件、设备实例和 aggregate 方式。
- 对应假设选择 LDS、MFMA、VALU、wait、cache/DRAM 或 occupancy 指标。
- 不把 profiler 插桩耗时当最终性能，也不把等待计数直接换算为 wall time。
- 有需要且平台支持才用 PC sampling/thread trace 定位热点指令。

使用 [trace/PMC 分析](references/trace-counters.md) 和
`scripts/counter_summary.py`。

### 6. 单变量消融，找最小改动

记录每个实验：source diff、正确性、median/离散度、ISA/counter 变化。
先挑影响最大的原因修正，再处理次要原因；组合优化后做撤销其中一项的反证。

GEMM 优先级建议：
1. 错误 lane/K packing 与 LDS swizzle 组合。
2. 不必要 dtype/scale/MMA encoding、地址计算及寄存器搬运。
3. 真实 spill / occupancy cliff。
4. DMA batch、MMA traversal、load reuse 与 compiler scheduling。
5. stages/unroll/wave map、grouped block swizzle、split-K/slice-K。
6. epilogue vectorization/reuse/overlap；重验同步和 partial 精度。

**禁止凭收益推断机制**：更快 ≠ 已知为什么更快。
机制不清只能写经验优化，不能写 root cause。

### 7. 正确性与回归门禁

- 独立参考 + exact/binary-scale probes + 稠密随机检查。
- 覆盖 full/HTI、MMA、layouts、split/slice、bias/dtypes、pipeline 边界、
  tails、padded stride/offset、guard、stream/Graph、动态 cache 复用。
- 数值 oracle 区分真实输出 dtype 与 intermediate/atomic dtype。
- 对重排同步：stress、最小 K、跨 wave 数据交换、重复初始化/回收都要测。
- compiler abort 子进程隔离；timeout/非已知失败正常报错。
- 全新 JIT cache 完整测试；旧路径要求不动时比较 emitted code。

技巧与风险见 [GEMM 优化速查](references/optimization-playbook.md)。
使用 [正确性门禁](references/correctness.md)。
有相同 ISA 是很强的**设备代码**非回归证据，不代表 host/dispatch 无变化。

### 8. 最终性能验证

- no-profiler，同输入/内存模式、足量 warmup，交替顺序，多轮原始样本。
- 至少再开一轮独立运行；小于噪声的结果记为持平。
- 同时报告 primary shapes 和 short-K/small-M/rectangular/partial cases。
- 保留 baseline + candidate + 原 reference 三者，避免只和不稳定参考比。
- 回退要么修复/回滚，要么经用户接受后明确列为 tradeoff。

可用 `scripts/benchmark_ab.py`，配合 `examples/scaled_gemm_adapter.py`。
默认测 Graph；不能替代用户应用的端到端 benchmark。

### 9. 交付与停止条件

必须交付：
- 最小 patch，改变/未改变的模块和语义；
- 根因证据链（具体地址/指令/counter、单变量效果、排除项）；
- 正确性结果、known failures、性能表和 measurement protocol；
- baseline、manifest、raw JSON/CSV/ISA/trace、复现命令；
- 未验证项与退化，commit/push 状态。

使用 [报告模板](references/report-template.md)。
案例见 [本仓库历史实战](references/case-studies.md)。

## 何时暂停优化

- 设备繁忙/时钟漂移导致噪声大于差距 → 先建立可靠测量。
- 数学结果、同步或内存安全不确定 → 先验证，不继续比速度。
- 只有“代码更短”但无 codegen/计时证据 → 只能称重构。
- 已满足目标，后续只能大改或伤害其他路径 → 停止并报告 tradeoff。

## 文件索引

| 文件 | 用途 |
|---|---|
| `references/measurement.md` | 环境、baseline、Graph/rotary/端到端测量 |
| `references/asm-ir.md` | IR pass、ISA、循环、bank 地址和 barrier ledger |
| `references/trace-counters.md` | rocprofv3 trace/PMC 与解释边界 |
| `references/optimization-playbook.md` | 按瓶颈选择 GEMM 最小优化杠杆 |
| `references/correctness.md` | 全路径矩阵、精度、同步与负面测试 |
| `references/case-studies.md` | 历史消融与 bank-conflict / epilogue 案例 |
| `references/report-template.md` | 交付模板与验收清单 |
| [scripts/README.md](scripts/README.md) / `examples/` | 可运行工具、限制与 adapter 契约 |
