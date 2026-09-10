# ASM / IR 深入分析 SOP

## A. 获取实际生成代码

**先查 dispatch**：打印/记录被选 symbol、constexpr config、grid、block、dynamic
shape/stride 和 cache key。只看 Python 函数名可能分析了错误 specialization。

FlyDSL 示例（先用小合法 shape 触发同一 specialization）：
~~~bash
# compile_probe.py 是你为当前 policy 写的单次调用程序，不含 autotuner。
HIP_VISIBLE_DEVICES=0 \
FLYDSL_RUNTIME_CACHE_DIR="$RUN/cache-baseline" \
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$RUN/ir-baseline" \
python compile_probe.py --variant baseline > "$RUN/compile-baseline.log" 2>&1

HIP_VISIBLE_DEVICES=0 \
FLYDSL_RUNTIME_CACHE_DIR="$RUN/cache-candidate" \
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR="$RUN/ir-candidate" \
python compile_probe.py --variant candidate > "$RUN/compile-candidate.log" 2>&1

find "$RUN"/ir-* -name '*.s' -print
~~~

上述环境变量在本仓库实战使用过，换 FlyDSL 版本仍要检查是否存在。
**相同 symbol 的多个 variant 不能共用 dump 目录**，后一次可能覆盖前一次。
pipeline 数字编号不是稳定 API，按 pass 名找文件。

没有 ISA dump 时，用当前 ROCm `llvm-objdump --help` 确认反汇编方式，
对保存的 code object 做 `llvm-objdump -d`；记录目标 arch。
不要对不可运行的 dump 宣称“实测生成结果”。

## B. 逐层检查 IR

1. origin：Tensor shape/stride、tiled_mma、copy atom、permutation。
2. canonicalize/layout lowering：哪些 shape/stride 已 constexpr？
   K uniform offset 是否进入 per-lane 地址？swizzle 作用在哪些 bits？
3. atom-call-to-SSA / regmem-to-vector-SSA：
   `fx.gemm` 的 register memory 是否完全提升？实际残留 load/store 是否 scratch？
4. ROCDL/LLVM：目标 intrinsic、scale 常量、format、op_sel、向量宽度，
   loop-carried values、pointer address space、alias/volatile/atomic。
5. ISA：最终指令和寄存器分配才是证据，IR 中的 rmem 名字不等于实际 spill。

### 代码风格与 SSA

`consume(c00, a0, b0, True)` 可通过原地更新 Python fragment list 实现，
元素仍为 SSA Vector；不必为了无等号变成 GPU store。

但编译器的 AST loop-state 分析未必看见函数内副作用。动态 loop 边界可能仍需：
~~~python
c00, c01, c10, c11 = compute_k_pair(k_tile, c00, c01, c10, c11)
~~~
以编译器实现、IR iter_args、跨多次迭代的正确性为准。不为风格删必要返回值。

## C. ISA 六步检查

### 1. Kernel metadata / occupancy

记录 VGPR、SGPR、AGPR（若有）、LDS、scratch、spill、wave size、workgroup size。
- ISA metadata 与 profiler 的 register 数可能采用不同编码/分配单位。
- 246→248 个 VGPR 不自动代表 occupancy 下降；检查分配粒度/阈值和 LDS 限制。
- static spill=0 不能单独证明 memory 路径无问题。
- ABI 删参可能改变 SGPR、指令调度；不能先承诺 ISA 不变。

### 2. 划分控制流区域

标注 prologue / steady-state loop / remainder / drain / epilogue。
回边地址、小块名字和真实 induction/update 都要读；脚本只识别候选回边。

~~~bash
python scripts/asm_summary.py "$RUN"/ir-baseline/*/*.s \
  --output "$RUN/baseline-asm.json"
python scripts/asm_summary.py "$RUN"/ir-candidate/*/*.s \
  --output "$RUN/candidate-asm.json"
diff -u "$RUN/baseline-asm.json" "$RUN/candidate-asm.json"
~~~

不要把全文件计数直接比较：4-K unroll 的静态 MFMA 是 2-K unroll 的两倍。
填写：
~~~text
loop A: 128 MFMA / 4 K tiles = 32 MFMA per tile
loop B:  64 MFMA / 2 K tiles = 32 MFMA per tile
dynamic total = sum(region executions * instructions per region)
~~~
也检查 remainder 是否实际上执行，partial/早退出时次数是否变化。

### 3. 指令类别与数据路径

- MFMA shape、`mfma_scale` vs `mfma`、format 与 scale/op_sel。
- `ds_read_b128` / transposed LDS read，`ds_write_b16`，向量/标量访问。
- async buffer load LDS 数量、`m0` 设置、`soffset` vs `voffset`。
- address VALU/SALU、额外 `v_mov`、convert/shuffle/permute。
- scratch load/store、atomic、global store 向量宽度。
- 数量相等仍可能有不同 bank conflict、依赖和 issue 顺序。

### 4. Lane 地址与 LDS bank

建立表而不是只读 layout 字符串：
~~~text
lane | wave/group | A/B row | logical K | vector bytes |
swizzled byte offset | bank words touched | hardware service subgroup
~~~

步骤：
1. 展开 MMA TV layout：thread bits 和 value bits 分别落在 M/N/K 哪些维度。
2. 展开 tiled-copy/retile：每条 LDS 指令实际读取的 lane 起始地址/宽度。
3. 展开 LDS physical layout 和 XOR；G2S 写地址必须与 S2R 读地址一致。
4. 以**目标架构** bank 宽度/数量、read 指令 service subgroup、广播规则计算冲突。
   bank-index 的起点通常形如 `(byte_offset / bank_word_bytes) % bank_count`，
   但跨整 wave 简单计数会忽略服务分组/广播，不能代替 ISA 文档或 PMC。
5. 验证 vec alignment 和 swizzle dest bits 是否越出 contiguous extent。
6. 修改 permutation 必须对 A/B、logical coordinate 和 MMA ABI 保持一致。

历史关键案例：16x16x128，K permutation 从
`(16,4,2):(1,16,64)` 改成 `(16,2,4):(1,64,16)`，
从每 lane 连续 K+0/K+16 变为 K+0/K+64，在当时 `Swizzle<3,4,4>`
和 `ds_read_b128` 下消除了主体冲突。**不是所有 block_k/MMA/layout 都适用**。

### 5. 关键路径和同步

画每个 wave 的：
~~~text
LDS load -> lgkm wait -> MFMA consumer
DMA issue -> vmcnt completion -> CTA communication -> LDS read
last AB read -> storage reuse as C -> ds write -> barrier -> global store
~~~

区分：
- `rocdl.sched_barrier(0)`：约束 compiler scheduling，不是跨线程同步。
- `s_barrier`：硬件同步与 wave phase，不能当作任意 memory-fence 替代物。
- `gpu.barrier`：DSL/IR memory effects 可能引入额外 fence/wait，检查实际 lowering。
- `s_waitcnt(vmcnt=...)`：等待计数器，不负责其他 wave 都抵达。
- 不同架构 load/store/LDS counter 语义不同；从 gfx950 迁到其他 arch 必须重查。

对手工 vmcnt：
~~~text
位置 | 自上个 wait 后各 batch issue 数 | 总未完成数 |
保留哪几批 outstanding | 即将读取哪片 LDS | buffer 可覆写时刻
~~~
改 load 数、batch fence、scale/bias memory placement 后全部重算。

对 HTI 错相 barrier：
~~~text
阶段 | M-wave group 0 已遇 barrier 数 | group 1 已遇数 |
该 phase 交换的地址集合 | 谁读/谁写
~~~
“表面 barrier 多余”不是删它的理由。全局 C-shuffle 可能跨 M groups，
另一个参考实现的 wave-local shuffle 则不需要相同 phase balancing。

### 6. 验证改动是否必要

- 对等重构优先 bytewise ISA/hash（含 metadata），再测试。
- 若只是 symbol/调试位置变化，可以另比较 stripped instruction stream，
  但同时保留原文件与 resources/ABI，不能用“归一化相同”替代完整证明。
- 用 microbenchmark/PMC 证实推断。VGPR/指令数下降但耗时不变应如实报告。
- 单改 `traversal_order`、K permutation、scale 状态优于全套重写；
  收益有限时不要声称它解释了全部差距。
