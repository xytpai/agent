# 工具使用：不依赖其他源码仓库

以下命令均从 `agent/skills/gpu-gemm-optimization` 目录执行。
所有 CLI 支持 `--help`；ASM/CSV 工具及 CPU 测试只需 Python 标准库。
benchmark 的库内示例仅需安装 GPU-enabled PyTorch，不要求 FlyDSL 或其他源码库。

## 库内文件

- [asm_summary.py](asm_summary.py)：分析用户本轮生成并保存在 agent 内的 emitted ISA。
- [counter_summary.py](counter_summary.py)：分析本轮保存的 profiler CSV。
- [benchmark_ab.py](benchmark_ab.py)：通用公平顺序计时器。
- [scaled_gemm_adapter.py](../examples/scaled_gemm_adapter.py)：
  库内 PyTorch 诊断 adapter，无私有 helper 导入。
- [torch_scaled_demo.py](../examples/torch_scaled_demo.py)：
  BF16 matmul + FP32 行列缩放，**不是 MXFP/HTI，不是优化 kernel**。

运行产物使用新建的 agent 内目录：
~~~bash
mkdir -p ../../temp
RUN=$(mktemp -d ../../temp/gemm-skill.XXXXXX)
export RUN
~~~

## asm_summary.py

输入 emitted AMD `.s` 文本，不是 LLVM IR 或含机器码栏的 objdump 输出。
`$RUN/isa.s` 是本轮准备的数据文件，不是本 skill 随附历史产物。

~~~bash
python scripts/asm_summary.py "$RUN/isa.s" --output "$RUN/asm.json"
~~~

输出 SHA256、函数声明、metadata、全文件opcode数和候选回边区间。
不推断trip count、issue、bank冲突或occupancy。不同variant分别输出。

## counter_summary.py

~~~bash
python scripts/counter_summary.py "$RUN/counters.csv" --output "$RUN/counters.json"
~~~

需要 Dispatch_Id / Agent_Id / Kernel_Name / Grid_Size / Workgroup_Size /
Counter_Name / Counter_Value。按kernel、agent、grid、WG分组。
默认拒绝每dispatch/counter多行，避免重复计数；仅确认维度可加时使用
`--sum-dimensions`。rate/derived metric的聚合和单位仍需查安装工具的说明。

## benchmark_ab.py：库内 A/A 诊断

~~~bash
python scripts/benchmark_ab.py \
  --adapter examples/scaled_gemm_adapter.py \
  --baseline examples/torch_scaled_demo.py \
  --candidate examples/torch_scaled_demo.py \
  --shape 64 64 64 --slots 2 --rounds 4 --launches 8 --warmup 2 \
  --mode graph --output "$RUN/aa.json"
~~~

两边故意为**同一实现**，只验证分配、独立oracle、顺序、Graph和计时接线。
不要将结果称为MXFP吞吐、HTI收益或历史kernel复现；没有GPU环境时仅跑CPU测试。
本轮真实优化代码及专用adapter应保存到agent库内，再替换相应参数。
如果更改shared dependency/compiler，需要独立快照/环境，不能同进程混用版本。

模式：
- graph：Graph捕获后event计时，不含Python wrapper逐次调用成本。
- events：Python提交循环的GPU event elapsed，可能含提交空洞。
- wall：batch前后同步的CPU墙钟，包含host/dispatch/同步成本。

先warmup所有slot、计时外validate，然后capture/测量。
输出raw_us、median/p10/p90/MAD、顺序、paired变化%、adapter metadata和源码SHA。
源文件测量期间变化则报错；%不自动代表显著性，A/A也有噪声。
`--only baseline` / `--only candidate` 用于分别运行和采集。

只有本轮adapter确实调用FlyDSL时，才能使用FlyDSL dump环境变量获得其ISA；
库内PyTorch诊断**不会生成自定义FlyDSL kernel的IR/ISA**。
实际采集用独立cache/dump目录且放在 `$RUN` 内，不引用历史外部产物。

## CPU测试

~~~bash
python -m unittest discover -s tests -v
~~~

只验证本skill工具与文档结构，不代表GPU kernel正确性/性能验收。
