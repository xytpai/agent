# Trace / PMC / PC sampling SOP

## 1. 先问 timeline 还是 kernel 内部

**timeline 症状**：
- 同一个 GEMM 用户 API 多发 kernel、memcpy 或 host synchronize。
- launch 间出现 CPU gap；Graph 下快但 eager 慢。
- 输出 alloc、`.contiguous()`、编译、scale reshape 或 sync buffer 初始化混入计时。
- 多 stream 意外串行、default-stream dependency、Graph 捕获了错误 stream。

**kernel 内部症状**：
- 同 dispatch 资源/输入条件下 device duration 仍不同。
- 问题随 K 稳定增长；怀疑 LDS/VALU/内存/计算/等待。
- occupancy cliff、bank conflict、atomic contention、pipeline drain。

先最轻量 kernel/API trace，再针对性加 PMC；不要一次采所有 counter。

## 2. 查询工具能力

~~~bash
rocprofv3 --version
rocprofv3 --help
rocprofv3-avail --help
rocprofv3-avail list --agent
rocprofv3-avail -d 0 list --pmc
rocprofv3-avail -d 0 info --pmc SQ_LDS_BANK_CONFLICT SQ_WAIT_INST_LDS SQ_INSTS_MFMA
rocprofv3-avail pmc-check --help
~~~

`-d 0` 是查询工具的 device，不保证等于 `HIP_VISIBLE_DEVICES=0` 映射。
确认 gcnArchName、PCI/UUID、agent ID，记录 exact commands。
换版本时字段、counter/derived metric 和同时可采数量都会变化。

## 3. Timeline trace

~~~bash
HIP_VISIBLE_DEVICES=0 rocprofv3 \
  --kernel-trace --hip-trace --memory-copy-trace --marker-trace \
  --output-format csv --output-directory "$RUN/trace-baseline" \
  -- python trace_probe.py --variant baseline
~~~

`trace_probe.py` 要含明确 warmup 与稳定重复区间。可以 ROCTx 标记 shape/variant；
不要靠进程启动后第 N 秒猜 warmup 结束。

分析：
1. 定位实际 kernel symbol，确认不是不同算法/policy。
2. 将 API enqueue、kernel start/end、memcpy、sync 配对到 correlation/queue。
3. 同 stream 计算 dispatch 间空洞；跨 stream 不可机械相加 GPU duration。
4. 分开 warmup、first JIT、steady-state、Graph capture/replay。
5. 是否频繁 runtime alloc / contiguous copy？是否 benchmark 自身逐次 synchronize？
6. 对照无 profiler baseline，量化 observer effect。

CSV 适合统计；可用工具支持的 pftrace/Perfetto 查看 timeline，
先查该版本输出格式。保存原 trace，不能只留截图。

## 4. 小批 PMC

当时 gfx950 环境验证过的 counter 集合：
~~~bash
HIP_VISIBLE_DEVICES=0 rocprofv3 \
  --pmc SQ_LDS_BANK_CONFLICT SQ_INSTS_MFMA SQ_WAIT_INST_LDS \
        SQ_WAIT_ANY SQ_INSTS_VALU SQ_INSTS_SALU \
  --kernel-include-regex 'hgemm_fp8' \
  --kernel-iteration-range 3-6 \
  --output-format csv --output-directory "$RUN/pmc-baseline" \
  -- python counter_probe.py --variant baseline
~~~

这是示例，不是对未来工具版本的保证。先检查同时采集能力；
不能同时采的集合拆 pass，并确认各 pass workload 相同。
iteration-range 的包含语义先看工具说明/输出，不假定“第3次一定 warm”。

counter probe：
- 一个已固定编译的 specialization，重复调用+同步；
- 不运行整个 pytest/autotuner；
- 只采目标 kernel；过滤后核验 dispatch 数、shape、grid、VGPR、LDS；
- hot/rotary 模式与比较方相同，PMC probe 模式与正式基准不同则注明。

## 5. 选择与解释指标

| 假设 | 看什么 | 不能推出什么 |
|---|---|---|
| LDS bank conflict | `SQ_LDS_BANK_CONFLICT`，`SQ_WAIT_INST_LDS`，lane 地址 | counter 高不等于全部 wall time 都浪费在 LDS |
| 工作量变化 | `SQ_INSTS_MFMA`、VALU/SALU/VMEM/LDS | ISA 静态数不能直接替代动态 count |
| pipeline issue | WAIT/LDS issue、MFMA utilization/coexec | 不同 wave 的等待不可加成 kernel latency |
| global memory | bytes/request、cache hit/miss、DRAM busy | cold hot 模式不同会完全改变结论 |
| occupancy | active waves、resources、grid distribution | VGPR +2 不一定跨档 |
| split-K 原子争用 | atomic requests、memory wait、分区数 | 加 split 并非必然提升并行效率 |

逐个阅读 Description、Block、Dimensions 和单位：
- per-SE/per-XCC/per-SIMD；工具是否已聚合。
- cycles/quad-cycles/wave-cycles/requests，不能混加。
- derived counter 可能是 ratio/average，不能对 dimension 无脑求和。
- profiling metadata register 编码可能不同于 ISA `.vgpr_count`。

`counter_summary.py` 默认要求**每 dispatch 每 counter 一行**；
遇到多 dimension 行会报错，只有确认 raw count 可加和时才传 `--sum-dimensions`。
它按 kernel/agent/grid/block 分组，不把不同配置混成平均数。

~~~bash
python scripts/counter_summary.py "$RUN"/pmc-baseline/**/*counter_collection.csv \
  --output "$RUN/pmc-summary.json"
~~~
shell 的 `**` 需启用 globstar，或用 `find` 得到具体路径。

比较每 dispatch 的 raw count + mean/median，必要时归一化：
`count / waves`、`count / MFMA`、`count / K_tiles`。分母和单位必须写清。

## 6. PC sampling / thread trace

只有 timeline+PMC 还定位不清时升级：
1. `rocprofv3-avail list --pc-sampling` / `info --pc-sampling` 查询支持。
2. 查 sampling method、interval、unit、beta/support 要求，不能照抄别的 arch。
3. 保存 code object 与 symbol/source map，将 hot PCs 映射到 LDS wait、
   MFMA dependency、branch、scratch/atomic。
4. 高 sample 数是热点相关证据，不自动等于该指令 latency：
   sampling 偏置、采样方法、等待归因规则都要核验。
5. 若用 thread trace/SQTT，先查 ROCm 版本支持、权限和开销；
   只采少量目标 dispatch，避免巨大文件/运行时扰动。
6. 用单变量 patch + 无 profiler benchmark 再证实，而不是以热图收尾。

## 7. 一条合格的根因结论

> 两者动态 MFMA 数一致、LDS/occupancy 未变；baseline 的 bank-conflict
> 比 candidate 高。地址推导发现 lane value K 位与 swizzle 冲突。
> 仅改 K permutation 后冲突数下降、LDS issue wait 下降，正确性保持，
> 无 profiler A/B 重复改善。因此确认该 mapping 是主要原因之一。
> 剩余差距另由指令 encoding/调度/尾部解释，未归因部分明确保留。

**不是**：“看起来像 LDS 问题，所以我改了所有 barrier 后变快了。”
