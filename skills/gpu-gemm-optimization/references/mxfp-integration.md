# MXFP4/8 集成、调参、测试与分析门禁

与 [优化手册](mxfp-optimization.md)、[历史案例](mxfp-case-studies.md) 配套。
历史路径名不代表现在的 PyTorch/aiter/examples 功能仍相同。
出处 `文件:L行号` 指 `temp/文件.jsonl` 的一基物理行。

## 1. 参数工厂必须与真实 allocation / ABI 一致

分别建模 **input packed bits / accumulator dtype / C-shuffle dtype /
global output dtype / split partial dtype**，不能用一个 dtype 算所有容量。

~~~text
AB bytes ≈ stages * (block_m + block_n) * block_k_bytes
scale bytes = halves * scale_slots * (sa_stage_bytes + sb_stage_bytes)
C bytes = actual_C_elements * cshuffle_bits / 8
union bytes = max(AB+scale, C)       # 仅在真实 allocation/lifetime 允许时
~~~

- 非chunk HTI按两个half、每half的stages计scale；chunk slots不与AB stages混淆。
- `sa_stage_bytes`是一个half/slot，**容量乘half数，单half DMA次数不乘**。
- MXFP4/8 C-shuffle历史常用BF16；dense用输入FP16/BF16，即使最终输出FP32。
  与 `SharedStorage.c` 同步，不能只修参数数字。
- 一个half C临时区和四个half独立地址的方案，C容量相差4倍；
  不能不改kernel就把估算除4。
- 常用HTI AB128KiB+scale16KiB=144KiB，不是136KiB。
  gfx950当时容量160KiB，目标架构需重新确认，未知arch明确处理。
- 分开验证MMA整除、copy向量覆盖、repeat预算、wave数、HTI stages/slice限制、
  LDS/XOR granule、wait编码范围。资源pruner不是硬件数学合法性。
- XOR要求正2次幂时用 `x > 0 and x & (x-1) == 0`；另一实现若有非2次幂
  fallback，不照搬拒绝条件。
- 历史HTI校验形如 `2*half_b_wait+half_a_wait < 63`，按实际协议推导；
  `(stages-2)*...`在HTI stages2恒为0，不能代替。63不是所有AMD指令通用常数。
- 参数校验用显式错误；assert在 `python -O` 下失效。
- **参数构造成功≠GPU支持**：后期examples只改参数层时，下游还缺scale
  array、scaled MFMA及入口，不能把host枚举说成MXFP执行验收。

出处：`mxfp4_torch6:L18,L20`；`mxfp4_torch5:L428,L434,L468,L474`。

## 2. Split-K：动态值、精度和完整调用成本分别看

可选策略（不代表所有仓库均支持）：
1. **FP32 partial + reduce**：workspace逻辑`[S,M,N]`，`S*M*N*4`字节，
   每split写自己的slab；固定顺序求和、bias只加一次、最后转output。
   S=1不分配额外workspace、不reduce。
2. **semaphore + atomic**：初始化→发布→atomic→最后到达者reset，
   省reduce但有状态、初始化、等待、竞争与舍入成本。

BF16 atomic历史21个shape都有超差元素；FP32 atomic+cast精度通过却可能更慢。
不为更快候选放宽阈值；随机数值通过也不等于replay同步安全。

aiter历史动态契约：
- `ks1/ksd`区分是否split，实际S存CSV `splitK`、由config选，
  不额外暴露用户覆盖；同一ksd产物服务2/4/7/8等合法S。
- launcher把S转为aligned working_k和grid.y，主kernel只需
  `ks_begin=block_idx.y*working_k`、`ks_end=min(...,K)`；
  删冗余主kernel形参**不是删split-K功能**。
- `use_split_k_semaphore`为内部配置轴，split1不重复枚举；
  name/cache/AOT区分模式（历史`_ksd_sem1`），旧CSV保留默认行为。
- 高split动态reduce循环依赖是独立瓶颈。固定S展开只作诊断；
  保留动态ABI时可实验分组展开/批量load，不能偷换静态接口。
- slice-K在WG内归约，不再乘全局workspace；本wave直接索引自身K片，
  避免逐kwave条件写fragment导致SSA dominance及寄存器问题。
- 计入reduce或cast。profiler kernel duration之和不含host分配/launch gap，
  Graph和eager端到端另报。workspace引用、stream隔离和生命周期必须安全。
- semaphore容量、初始化线程覆盖、跨WG发布与复位协议独立验证，
  `s_barrier`不是跨WG同步；遵守 [正确性门禁](correctness.md)。

出处：`mxfp8_aiter:L270,L337,L459,L495,L510,L514,L518`；
`mxfp8_aiter2:L114,L177,L183`。

## 3. Tuner：先修实现和误裁剪，再扩大搜索空间

顺序：固定policy复现 → 审计候选/pruner → 代表shape单变量 → 冻结kernel及
编译身份 → compile-only → retune → 独立复测 → AOT/run-only。

可复用hgemm轴，但按MX重审：
- M/N/K tile、stages、m/n/k waves、MMA、HTI/FT、group_m。
- split包含合法非2次幂，如K7168的7/14/28、K384的3。
- `direct_b`/`b_to_lds`统一命名，真正在同一合法policy下枚举，
  不能由FT/HTI硬编码后声称“两条都tune过”。
- semaphore/reduce分别留finalist，**旧表赢家保留为候选**。

已见误裁：
- M=1只按padding IOU偏爱BM16，漏BM32；
- split grid rounds=2，把基础grid96的split7裁掉，漏约12.6%收益；
- BF16的每wave MMA repeats≤4不是MXFP硬件条件；
- occupancy估计相同就删slice，忽略了单wave工作量变化。
这些是定向审计理由，不是无条件放开全部候选。

**统一优化目标**：单buffer hot、3套rotary、31套rotary可能选不同赢家。
Graph初筛、profiler复选、最终对比使用相同工作集，或明确差异并证明未误杀。
区分tuner漏搜、测量目标不同和kernel固定开销。

更大空间不保证无回退：历史M3联合调优geomean1.0486×，仍17/100退化；
3个semaphore初选在独立复测都输给非semaphore。
不能把整体收益归给新轴，也不应直接覆盖整表。

### 编译与测速分离

- EXHAUSTIVE首轮久，先看CPU、产物增长和实际候选数，不直接判GPU卡死；
  减reps不解决上千policy冷编译。
- shape动态不代表dtype/tile/stage/waves不需分别编译。
- 检查Inductor候选有无 `precompile()`；只设置compile_threads无法补接线。
- CPU子进程compile-only可并行，同卡性能候选串行测；别直接去掉保护共享
  JIT状态的锁。大规模LLVM资源耗尽时按shape/候选分块、独立进程回收。
- `--mp`可能只并行profiler、不并行主进程Graph筛选；
  按shape绑卡、batch足够，每卡独立shard可同时并行两阶段。
- 断点含kernel/config-space hash，新输出避免旧行跳过，完整验收后替换正式表。

历史用户定向pruner：M,N,K均≥4096限定256² HTI waves2×4，
合法BK/group_m继续搜，大幅减少候选。**这是当时gfx950的定向策略，
不是通用最优证明**；测阈值前后、disabled tuning和grouped不受影响。

出处：`mxfp8_aiter:L205,L223,L337,L341,L474`；
`mxfp8_aiter2:L177-L217`；`mxfp4_torch4:L467-L508`。

## 4. Runtime → CSV → AOT → run-only 必须同一契约

- MX配置独立于PTPC/block128；可以复用模型shape，不能给旧格式耗时改标签。
- 按架构**和**CU数筛选，gfx950/gfx1250可能都有256 CU。
- 检查kernelName parser、dtype、bias、B/scale布局、direct-B、split模式、
  target arch、ABI版本与CSV一致，失配明确拒绝。
- 公共入口、`out=`、显式/current stream、Graph、Inductor均要接通；
  gfx950原始scale与gfx1250原shuffle ABI分离，不覆盖其他架构默认行为。
- CPU-only AOT用tiny real CPU tensors构造相同layout-dynamic ABI，
  不分配模型大tensor、不launch GPU；compile-only模式、arch要一致。
- **新进程** `FLYDSL_RUNTIME_RUN_ONLY=1` 经真实公共入口验证cache命中、
  dirty out、动态split/shape、bias和B路径，不能只直接调用编译函数。
- 改签名/name/import路径后旧AOT或Inductor cache可能失效。
- 多设备dispatcher需要设备/module隔离，不能动态shape却跨设备错用module。
- 静态特化必须进入真实cache identity；用户要动态接口就不偷偷加
  specialization key。kernel动态ABI不等于整个torch.compile支持任意符号shape。

出处：`mxfp8_aiter:L391,L424,L434,L543`；`mxfp8_aiter2:L16,L24,L114`；
`mxfp4_torch4:L118,L138`；`mxfp4_torch5:L355`。

## 5. 测量防坑

1. 尊重允许使用的GPU集合，记录物理PCI/UUID和可见编号，不照抄历史卡号。
2. 保存dirty baseline与共享helper，独立缓存、同输入、足量warmup、
   交替/反序、多轮独立进程；不删用户缓存、不抢他人GPU。
3. 核对实际module/symbol/grid/tile/stages/waves；历史打印256²但真实缓存128²，
   约2140 TFLOPS曾被误认成回退。
4. DSL launch使用**被capture的current stream**；排除空Graph/异步漏计。
   用Graph内输入/scale更新与输出变化验证真的执行目标kernel。
5. 区分profiler duration之和 / 批量Graph每GEMM时间 / 单次Graph replay /
   host wall time；小kernel尤其不能混用。
6. 量化、A/B/scale preshuffle、alloc、reduce是否计时逐项列清。
7. 原表环境/config不全，只能说“相对历史表下降”，不能归因于最后重命名；
   固定policy ISA一致也不代替autotune/dispatch端到端回归。
8. geomean提升不能隐藏小M/窄N/短K下降；重新计时不等于重新调优。
9. 8192³的5P目标约219.902µs（2MNK/时间），只是单位换算，不是性能保证。

## 6. 数值与路径矩阵

独立reference：
- FP4拆nibble，按ABI确认低/高nibble顺序；E2M1幅值表
  `[0,.5,1,1.5,2,3,4,6]`及符号位，不能把packed byte直接当浮点值。
- E8M0先转整数再减127（防uint8下溢），沿连续K32展开后FP32 matmul。
- exact probe限制幅度，证明partial/bias/output均可精确表示；
  dense随机与真实量化补充，不只两个kernel互比。
- Torch参考精度开关在框架tearDown检查前恢复，不能只靠更晚addCleanup。

| 类别 | 必测内容 |
|---|---|
| 格式 | FP4/FP8、uint8/E8M0 view、SA/SB混合合法表示、非均匀K-group |
| 几何 | full/HTI、支持MMA、tile/half边界、非对称waves、LDS边界 |
| K | 最小、drain、main、stage wrap、chunk切换/回绕、tail、多MMA K步 |
| layout | NN/NT/TN/TT、padded stride、offset、singleton non-unit stride |
| scale | group0/group2分离、K16 strip/impulse、指数边界/特殊值按契约 |
| reduction | 非2次幂/高split/不均匀尾、slice、合法组合、bias只加一次 |
| store | 输出dtype、direct/C-shuffle、N向量宽度、脏out、双端guard |
| runtime | dispatcher跨shape/stride/device、双stream、Graph更新A/B/scale |
| 集成 | 默认/强制policy、DEFAULT/EXHAUSTIVE、CSV→AOT→run-only、旧dense/grouped |

singleton `is_contiguous()`可能为真但stride≠1，`.contiguous()`可能不复制，
必须按DSL约束处理。旧store要求N为8倍数是实现条件，不是MX标准。
不支持组合先host reject，不能发危险kernel后skip。
历史扩展曾9712 passed /18 failed（dominance、split初始化），不能说全通过。
known failures、compiler abort子进程、Graph压力及guards遵守
[正确性门禁](correctness.md)。subTest可减重复，但测试条数不等于语义覆盖。

## 7. 精简与迁移：共享代码，不强制共享调度

- 保留原full/HTI函数及参数顺序，用constexpr dtype分支、追加可选scale；
  历史用空tuple表示无scale（None默认参数不支持），版本变化重验。
- 共享E8M0 loader、AB helper、launcher、reference/test helper；
  公共名字用mxfp，不让FP4共享函数还叫mxfp8。
- 不加core/wrapper只为挪行数；净增、总增删、PR merge-base三点diff分开报。
- 不改路径检查AST/ISA/资源及双轮性能，包含共享helper消费者grouped。
- 用户优先简洁可撤interleave/direct，但记录性能损失；分开保存实验最佳
  和正式简洁版，不能用实验峰值宣传正式版。
- examples后期经用户要求删legacy feature是范围决定，**不是优化技巧**；
  新任务默认不能删split/slice/FP32/PTPC。
- `mxfp4_aiter`短日志未完成实现/验证，不能写成aiter FP4已接通。

## 8. Thread trace专项

先读 [trace/PMC SOP](trace-counters.md)，再用以下经验：
- 查rocprofv3 ATT能力、decoder ABI/版本；保留raw `.att`、code object、
  schema及源码hash。API timeline不是thread trace。
- 当时selected-regions暂停接口/JSON输出曾崩溃；可先CSV timeline定位warm
  dispatch，再symbol+iteration采ATT离线decode。**别照抄历史第53次**。
- 检查丢包、unmapped PC、不完整wave；ISA内inline asm字符串可能被统计器漏掉，
  用独立反汇编核验，不能拿错误静态计数归因。
- 当时gfx9 decoder：time为首次尝试、time+stall为成功issue，
  duration=stall+issue，**不是访存完成latency**，换版本重新查schema。
- barrier可能同时有issue和wait记录；计指令不双计，stall单独统计。
- 4/8-wave按同WG或每1024 MFMA归一化，再划prologue/steady/drain；
  主循环覆盖不同K数时分母分别计算。
- decoder workgroup_id可能全相同；只能以驻留区间、完整SIMD覆盖、wave数、
  MFMA总数推断cohort，并明确标注推断。
- 合并采样SIMD的成功MFMA issue时间，区分单wave等待和**采样CU**全SIMD
  发射空隙，不是全GPU停机，也不能加总wave stall当wall time。
- 复采另一CU，记录gap中的wait→scale DMA→phase barrier→LDS链条。
  gap缩短而无profiler计时持平，可能停顿转移，不能直接删除热点barrier。

出处：`mxfp4_torch:L530-L637,L679`；`mxfp4_torch2:L54`。
