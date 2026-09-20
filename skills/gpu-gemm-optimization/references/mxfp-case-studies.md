# MXFP4/8 历史案例、失败实验与归因纠错

## 证据范围

本次是**文档整理，不是 GPU 复测**。覆盖 `temp` 的12份MXFP专项JSONL；
不把本轮 `mxfpsummary.jsonl` 纳入来源，避免循环引用。
其他非MX日志未作为本次新增MX结论来源，已有通用GEMM案例仍保留。

下文 `文件:L行` 对应 `temp/文件.jsonl` 的一基物理行，不是被引用回复里的行号。
[source index](mxfp-source-index.json) 固化12份日志的SHA256、范围及关键记录。
索引区分历史assistant总结、原始observation摘录、用户范围决定；
历史总结不是本轮独立复测，observation也须识别是否只在引用更早历史。

**自包含约定**：本 skill 的方法、伪代码、案例表和精选证据都保存在 agent 库内。
不引用外部 kernel、私有 helper、实验目录或复现脚本；原始 `temp` 日志仅为
agent 内可选溯源，清理日志也不影响阅读和 CPU 测试。
索引内摘录已去掉外部路径/代码命令，并单独记录清理后摘录的 hash；
原记录 hash 仍标识未经修改的历史日志，二者不能混用。
历史数据不等于保存了完整 kernel/IR/ATT，不能声称只凭摘要可逐指令复现。
重新实验时须为当前任务在 agent 库内准备独立实现与完整测量产物。

## 1. MXFP8 scale优化链条

gfx950，8192³、BF16输出，早期主要是256²×K128 HTI、2×4 waves、
15套rotary输入。各行是**不同阶段自己的对照**，不能把百分比相加。

| 阶段 | 历史结果 | 可复用经验 | 来源 |
|---|---|---|---|
| 临用global byte → dword direct-to-LDS | 1.979→0.616ms，556→1784 TFLOP/s | scale重复加载和主循环spill比“有没有scaled MFMA”更关键 | `mxfp8:L106` |
| 逐tile → 单8-tile chunk | 约1784→2567/2587 TFLOP/s | 减少填充与重复搬运、跨迭代复用 | `mxfp8:L136` |
| 单8-tile → 两个4-tile slot | 约2580→2728/2740 | 同144KiB LDS，隐藏更新load后立即wait | `mxfp8:L165` |
| chunk开关单变量 | 0.610→0.400ms，1803→2745 | 约52%吞吐收益，不能当可选小调参 | `mxfp8:L262` |
| FT四tile chunk+调度 | 0.605→0.473ms，1817→2324 | LDS136→144KiB，后续必须加短K/occupancy回退 | `mxfp8:L282,L302` |

后期PyTorch固定HTI约2800–2835 TFLOP/s是另一阶段/计时口径；
不代表每个版本都零spill，也不保证autotune必然选同一配置。

### “copy atom更快”的完整纠错链

`mxfp8_torch_bench:L158,L166` 最初把2630→2800归给copy atom+ROCDL等待。
后续：
- L191：恢复共享inline wait/barrier无回退；
- L199：scale inline vs atom基本持平；
- L221：AB换inline时回退，但同时删了sched fence，不是单变量；
- L227原始输出：inline **2619.2**、fenced **2815.6**、atom **2816.6**；
- L229：两种fenced的private segment36B，unfenced56B；
- L245：最终结论是**load前后调度栅栏**，保留全inline；
- L279：BF16/FT全开fence有快有慢，不泛化。

所以skill应教“保持fence位置一致再比较加载表达”，而非强制改copy atom。

## 2. MXFP4：修正K单位与scale chunk，远比换wave数重要

`mxfp4_torch:L148`：FP4默认BK256没有接BK128专属chunk，导致61 spills /
248B private segment；按固定逻辑K512改成2 tile chunk后，spill/private均0，
LDS同为147456B。PyTorch Graph同场：

| 路径 | TFLOP/s |
|---|---:|
| FT 256×256×256、4×4 waves | 3543.6 |
| HTI BK128、2×4 waves | 3774.6 |
| HTI BK256、2×4 waves、chunk修复 | 4231.0 |
| DEFAULT autotune选该HTI | 4151.9 |

这是当时的特定配置，不是后期所有动态版本的资源数。

### Bank conflict不能直接换成可节省时间

`mxfp4_torch:L356`，相同HTI完整PMC：
- 完整冲突14,680,064；C-only诊断2,097,152；
- 去scale LDS读取的固定scale诊断2,097,152；
- 完整kernel改C XOR后12,582,912，但性能4306→4276 TFLOP/s；
- 去AB swizzle冲突升至90,177,536。

C冲突降了但总kernel没快；scale贡献约86%冲突计数，不等于86%耗时。
宽读scale的某些方案未降冲突，却引入80/104B profiler Scratch_Size。

## 3. 接近5P的两条路线，契约不能混写

以下是gfx950、8192³、MXFP4→BF16、8-wave HTI、多轮Graph历史结果，
大多是实验kernel；量化及重排不计时，静态/动态须分别标注。

| 路线/阶段 | 历史吞吐 | 关键限制 | 来源 |
|---|---:|---|---|
| non-shuffle原调度 | 约4.37P | 普通A/B/scale | `mxfp4_torch:L679` |
| B+scale custom preshuffle、同步复用 | 4.64–4.65P | A未重排；custom ABI | `mxfp4_torch:L721` |
| 安全DMA/MFMA交错 | 约4.76P | 保留reader barrier、不carry额外fragment | `mxfp4_torch2:L151` |
| AB6/12、scale14 | 4.82–4.83P | custom B/scale，direct store | `mxfp4_torch2:L194,L196,L202` |
| M-major | 约4.871P | 仍未5P，出现单VGPR spill | `mxfp4_torch2:L365` |
| 全A/B/scale preshuffle + fixed8192 | 4.972/4.986P | 固定shape/stride，非动态通用接口 | `mxfp4_torch2:L483` |
| 同写法恢复标准non-shuffle | 约4.399P | scale layout及op_sel同时恢复 | `mxfp4_torch2:L497` |
| non-shuffle AB scalar-base | 4.725–4.731P | exact实验，252 VGPR、无spill | `mxfp4_torch3:L39,L65` |
| 再scale scalar-base | 约4.73–4.75P | 246 VGPR，scale单项收益小 | `mxfp4_torch3:L41,L45,L108` |
| prologue+C00 DMA+scale交错 | 4.810/4.805P | 标准输入，但仍是实验专用优化 | `mxfp4_torch3:L280` |
| 最新交错+全preshuffle | 5.004/4.999/5.005/5.007P | 四轮中一轮低于5P，无明显余量 | `mxfp4_torch3:L287,L289,L291,L292` |

**结论**：non-shuffle已见约4.81P，不是硬件上限，但历史没有稳定达到5P。
全preshuffle曾到5P附近，不可当标准动态PyTorch端到端性能。

### 从实验最佳到最终简洁版

这是用户取舍，不是后续优化自动保留：
- `mxfp4_torch4:L118` 静态集成：正式kernel约4.80P，compile约4.78P，
  第二轮尚未收齐，不能说最终验收完成。
- L138恢复统一动态layout/cache接口后，正式约4.65–4.67P，
  compile约4.63–4.65P；不再引用静态4.8P。
- L242,L270：scale挂到AB loader，删callback，持平略好，共享代码不强行改FP8调度。
- L373：按要求删interleave，FP4约239.4→255.3–255.9µs，延迟增约6.7%。
- `mxfp4_torch5:L274,L285`：后期direct vs C-shuffle约4214.7/4194.1 TFLOP/s；
  删除direct净减83行，正式复测约4207.1→4190.1，降约0.40%。

此前custom preshuffle direct vs C-shuffle约4.82/4.60P（约4.6%差距，
`mxfp4_torch2:L232`）不与后期约0.5%矛盾：代码、ownership、调度、输入契约不同。

## 4. 失败实验是下一轮搜索的边界

| 尝试 | 历史观察 | 下一次必须改变什么 |
|---|---|---|
| GROUP_M扫描 | 0最佳，1约慢6.6%，4/8略慢 | 先证明cache/block map瓶颈，不拿group_m解释全部gap |
| 原生HTI 4-wave | 每线程accumulator翻倍，180B scratch，约1.77P | 先配套寄存器/epilogue设计 |
| HTI 128 AGPR真驻留 | 零spill仍约慢4%，K-pair6344→7168 cycles | 分组asm/NOP/等待调度，不只强塞AGPR |
| 完整官方pipeline移植 | 148/208B scratch，约2.99/2.05P | 缩live range，保留原同步做最小交错 |
| 提前到reader barrier前 | 长K约11–12%元素错误 | 不能越过最后读者 |
| scale宽读/打包/laneshare | 多数spill或更慢，部分零spill也慢 | 推导真实bank+live range，不能只减少指令 |
| 只减chunk wait budget | gap缩短但吞吐持平 | 热点不等于可独立删除的总耗时 |
| 去chunk同步/移动DMA | 新scratch、更多长gap、退化 | 停顿可能转移，重新画依赖链 |
| MXFP8照搬FP4交错 | 多候选慢1–5%；粗load曾重复两倍 | 每consume长度、chunk边界、scale复用单独设计 |
| FP4 scale lookahead | 正确、无spill、ISA确实重叠，但更慢 | 不盲加预取深度 |
| 8-byte direct-to-LDS合并 | invalid operand，未收尾 | 先查合法ISA，不能写成成功技巧 |
| epilogue lag2 | 仅约0.1–0.2%且接近噪声 | 不为亚微秒收益默认增加复杂度 |

出处：`mxfp4_torch2:L28,L54,L83,L151,L322`；
`mxfp4_torch:L595-L637,L679`；`mxfp4_torch3:L108,L195`；
`mxfp4_torch4:L43,L75,L192`。

## 5. aiter：direct-B、split-K与tuner的独立收益

- `mxfp8_aiter:L223,L270`：先扩大空间没有填平结构差距；
  preshuffled B仍经LDS、scale临用加载、BF16 atomic精度淘汰高split才是问题。
- L295纠正：早期204个DSV4 tuning是**block128**，不是标准MXFP8。
  L391重新用标准block32调全部204个shape；L424后改为M3的100个shape，
  不沿用旧格式时间。只是模型shape，不是全模型量化推理验收。
- L474：47个完全同policy FT配对，direct-B胜42，geomean1.15×；
  另28个LDS超容量需双方同降stage/tile，不混进原policy统计。
  大tile FT控制组LDS反而更好，原HTI又更快，不强制所有路径direct。
- L510：13个同policy split形状，semaphore BF16 atomic几何平均慢7.7%，
  FP32 atomic+cast慢40.7%；全部21个BF16 atomic形状有超差元素。
  **这是当时候选/阈值，不是任何semaphore实现必错。**
- L518：partial+reduce时间确实包含reduce，但仅GPU duration求和，
  不含host分配、launch gaps。
- `mxfp8_aiter2:L114` 后保留两策略、config轴选；
  L177联合retune100/100完成，geomean1.0486×、17退化，
  3个sem初选独立复测全部输给非sem，正式CSV未覆盖。
- `mxfp4_aiter:L1-L8`仅请求/勘察，不能据此声称aiter FP4已经落地。

## 6. 必须保留的未验证项与反例

- `mxfp8_opt2:L85` 的MXFP扩展是9712通过/18失败，不是全绿；
  失败含LLVM dominance及split初始化非确定性。
- `mxfp8:L240,L251` 某PR swizzle=4偶发数值异常，swizzle=0通过该轮检查，
  根因未定位，不能说XCD swizzle普遍错误或取最高吞吐作为合格结果。
- `mxfp4_torch5:L142,L162,L214`：MXFP8历史表对照17/17更高，
  MXFP4仍9/17低于旧表；前四卡复测主要下降约29.2%、18.1%、15.2%。
  geomean上升不能掩盖；沿用旧选型未retune，不能归因最后重构。
- `mxfp4_torch4:L423` grouped full-tile K-tail旧编译问题，前后均复现，
  不可当本次新增bug，也不能称grouped无已知问题。
- `mxfp4_torch5:L467,L473,L474` general chunk仅host参数：
  86,400组、5,335接受，容量/边界自洽，**未做新公式GPU性能验收**。
- later examples与Torch的输出dtype、split/slice/PTPC支持范围曾变化；
  功能是否存在必须检查当前代码，不能凭本历史摘要推断。

## 7. 使用这些案例的正确方式

1. 先读 [优化手册](mxfp-optimization.md) 确认适用前提。
2. 用 [集成门禁](mxfp-integration.md) 建立契约/测量/验证矩阵。
3. 优先读库内source index的内嵌证据；agent内原日志仅作可选溯源，
   不跳转外部源码，也不把几十MB历史全部塞进上下文。
4. 保存当前代码baseline；重新做单变量A/B、ISA/PMC/ATT。
5. 报告区分历史结果、本轮验证、未验证与用户接受的tradeoff。
