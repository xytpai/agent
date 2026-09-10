# 工具使用

所有 CLI 都支持 `--help`。ASM/CSV 工具与 CPU 单元测试仅用 Python 标准库；
benchmark 需 ROCm PyTorch、FlyDSL 和合法的 gfx950 policy。

## asm_summary.py

输入 **emitted AMD `.s` 文本**，不是 LLVM IR，也不是含地址/机器码栏的 objdump 输出。

~~~bash
python asm_summary.py /path/to/21_final_isa.s --output asm.json
~~~

输出：SHA-256、函数声明、metadata各条目、全文件opcode计数、候选回边区间计数。
不推断trip count、不模拟issue、不证明bank冲突、不自动计算occupancy。
多个文件分别输出，避免把相同symbol的不同variant混在一起。

## counter_summary.py

~~~bash
python counter_summary.py /path/to/counter_collection.csv --output counters.json
~~~

要求 rocprofv3 的 Dispatch_Id / Agent_Id / Kernel_Name / Grid_Size /
Workgroup_Size / Counter_Name / Counter_Value 列，其他字段保留在原CSV。
按kernel、agent、grid、workgroup区分配置；每dispatch聚合后才求均值。

若一个dispatch/counter多行，默认报错，防止维度重复导致假结论。
只有确认counter是可加的raw count、不同row确实是不同维度而非重复采样时才用：
~~~bash
python counter_summary.py counters.csv --sum-dimensions --output sum.json
~~~
工具无法自动知道rate/derived metric是否可加。单位必须从本机counter说明取得。

## benchmark_ab.py

~~~bash
HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_CACHE_DIR=/tmp/gemm-skill-cache-unique \
python benchmark_ab.py \
  --adapter ../examples/scaled_gemm_adapter.py \
  --baseline /path/to/snapshot/baseline.py \
  --candidate /path/to/repo/kernels/scaled_gemm_gfx950.py \
  --shape 256 256 256 --slots 2 --rounds 4 --launches 8 --warmup 2 \
  --mode graph --output smoke.json
~~~

上面是小尺寸工具smoke，不是正式性能协议。
正式测试再提高rounds/launches/rotary slots，记录环境和多次独立运行。

模式：
- `graph`：Graph捕获后event计时；不含Python wrapper逐次调用成本。
- `events`：Python提交循环的GPU event elapsed，可能含提交空洞。
- `wall`：batch前后synchronize的CPU墙钟；含Python/dispatch/同步成本。

先 compile/warmup所有slot，再调用adapter的独立oracle，最后计时。
输出保存 raw_us、median/p10/p90/MAD、顺序、paired变化%、policy、
Torch/HIP/FlyDSL路径和源码SHA。测量过程中源文件改变会报错，不写有效结果。

隔离导出：
~~~bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/ir-A-unique \
FLYDSL_RUNTIME_CACHE_DIR=/tmp/cache-A-unique \
python benchmark_ab.py ... --only baseline --output baseline.json
~~~

`...` 需替换为必需的 adapter/baseline/candidate/shape参数。
candidate另用`--only candidate`和独立目录。不要在一次`--only both`
进程中将同名kernel的两个IR dump写到同一目录。

**边界**：
- 模板只支持NT/HTI/BF16/no-bias；别拿它验证所有paths。
- shared dependencies仍来自candidate所在repo，记录了hash但没有版本隔离。
  helper/compiler有改动应另用独立snapshot/worktree环境。
- benchmark输出的%不自动代表显著性；同源码A/A也会有噪声。
- artifact/profiler/timing用同一工作集策略才能比较；无profiler final run仍必需。
