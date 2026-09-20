# MXFP4 / MXFP8：gfx950 优化手册

从 `temp` 中 12 份 MXFP 专项历史提炼，不是当前性能承诺，也不是要求把所有
实验一起合入。原日志行号、历史效果、纠错见 [案例](mxfp-case-studies.md)，
完整来源清单见 [source index](mxfp-source-index.json)。
调参、集成、测试与 ATT 方法见 [集成门禁](mxfp-integration.md)。
下文 `文件:L行号` 均指 `temp/文件.jsonl` 的一基物理行。

## 1. 先分清量化语义、逻辑 K 和存储字节

标准 MX：沿每行**连续 32 个逻辑 K 元素**共用一个 E8M0。以 B 的逻辑
`[N,K]` 表示（外部 API 也可能传转置 view）：
~~~text
A_real[m,k] = decode(A[m,k]) * decode_e8m0(SA[m,k//32])
B_real[n,k] = decode(B[n,k]) * decode_e8m0(SB[n,k//32])
C[m,n] = sum_k(A_real[m,k] * B_real[n,k]) + bias[n]
SA: [M,K/32]; SB: [N,K/32]       # 整除为此 ABI 前提，tail 按具体接口定义
~~~

| 量 | MXFP8 历史常用 HTI | MXFP4 历史常用 HTI |
|---|---:|---:|
| 数据格式 | E4M3 | packed E2M1，2 元素/byte |
| tile M×N×K（逻辑） | 256×256×128 | 256×256×256 |
| `mma_k`（此指令形状） | 128 | 128 |
| 每 K tile 的 MMA K 步数 | 1 | 2 |
| `block_k_bytes` | 128 B | 128 B |
| 一个 HTI double-tile 的逻辑 K | 256 | 512 |
| 每 scale chunk 的 K tile 数 | 4 | 2 |
| 每行每 chunk 的 scale 字节 | 16 B | 16 B |

两者常用 stages=2、waves=2×4、GROUP_M=0；这不是万能默认。
`mma_k=128` **不是所有硬件形式的 K 上限**；双缓冲也不要求一个 tile 有两个
MMA K 步。double-tile、AB stages、chunk tile 数、scale buffer slot 数要分开。

FP4 接入尽量复用 FP8 框架，只改变：
- `block_k_bytes = block_k * input_bits // 8`，global/LDS 地址按 packed storage；
- copy/fragment 字节布局：历史借用 `MFMA(..., mma_k//2, FP8)` 描述 packed
  bytes；**计算**仍用 `MFMA_Scale(..., mma_k, Float4E2M1FN)`；
- dtype 编码、合法性及 scale 元数据，不新增运行时解包 GEMM 冒充原生 FP4。
- DMA granule=16B 时对应32个FP4或16个FP8，不是一个 MMA 步。

不要混淆：
- FP8 PTPC 外部行/列 FP32 scale 与 per-K32 MX scale。
- A 1×128 / B 128×128 blockscale：二次幂 E8M0 可广播成受约束的 K32 表，
  任意四个独立 K32 scale 却不能合成一个 block128 scale。
- **使用 scaled MFMA 不代表输入就是标准 MXFP8**。
- E8M0 原始 bits 用 `uint8` view/reinterpret，不是数值 cast。
  finite byte 的参考是 `2**(int(byte)-127)`；127是1，0**不是数值0**，
  255为特殊值，不能套 finite 公式。

出处：`mxfp4_torch:L88,L148`；`mxfp4_torch5:L295-L312`；
`mxfp8_aiter:L295,L391`。

## 2. Scale 和 operand 必须具有相同生命周期

HTI：`C00=A0*B0, C01=A0*B1, C10=A1*B0, C11=A1*B1`。
每份 A/B fragment 有两个消费者。如果仅把数据读入 VGPR，后续 MFMA 再从
LDS 读 scale，预取可能已覆盖旧 scale，形成“旧数据 × 新 scale”。

~~~text
load_operand_fragment() -> (data_fragment, scale_fragment)
consume() -> 同时复用二者，直到最后一个消费者结束
~~~

先画四种 lifetime：AB stage、scale chunk slot、寄存器 fragment、C union。
**数据已进寄存器，不代表 scale LDS 可以覆盖**，除非对应 scale 也已读走，
且所有跨 wave 读者结束。推迟 scale load 虽降低 live range，却可能越过回收点；
历史有低 VGPR 但数值错误的版本，必须拒绝。

最低测试：不同 K32 group 使用不同指数、跨多个 chunk、奇数 K tile、
最短/不均匀 split 分区、同一 Graph 更新输入与 scale、脏 out、输出哨兵。
出处：`mxfp8_torch_bench:L62,L166`；`mxfp4_torch3:L108`。

## 3. Scale 搬运：byte → dword DMA → chunk → 双缓冲

### 消掉 MFMA 中间的逐字节 global load

临用临读 global scale 会重复加载、把 vmcnt 等待插进 MFMA，并让高 VGPR
版本 spill。优先尝试 **32-bit direct-to-LDS**：每 lane 搬4个 E8M0，
与 AB 流水配套，LDS→register 后按 fragment 复用。

历史初版约556→1784 TFLOP/s；对应主循环 global scale byte load 48→0、
scratch 访问3→0。**当时整个 kernel 仍有少量 spill**，不能说完全消除。

### Chunk 首先解决有效 DMA 覆盖率

256×256、512 threads、HTI half rows=128、BK128：
~~~text
一轮 scale DMA 覆盖 W = 512 * 4B = 2048B
一个 half / 一个 K tile 的有效 scale = 128 * (128/32) = 512B
四个 tile = 128 rows * 16B = 2048B
~~~
旧 helper 通过重复有效行补齐线程覆盖；合并四个 tile 正好一轮。
FP4 BK256 每 tile 每行8B，两个 tile 即填满。
**四条 A0/B0/A1/B1 DMA 不代表四个 K tile**。
上述是有效数据/指令覆盖量，不是实测 HBM traffic。

### 两个4-tile slot，而不是一个8-tile slot

单8-tile slot 更新时“发load立刻wait”；两个4-tile slot 让当前计算与下一
chunk 预取重叠。历史8192³ MXFP8约2580→2730 TFLOP/s，总LDS同为144KiB。
固定配置chunk开关消融约1803→2745 TFLOP/s。

~~~text
slot = (k_tile // chunk_tiles) % 2
row_byte_offset = (k_tile % chunk_tiles) * (block_k // 32)
~~~

AB 两个stage仍按tile轮换，不是也保存整个scale chunk。
chunk DMA与逐tile AB DMA频率不同，**等待预算分开**；记录每次发射数、
最晚读者、slot回收点及两组wave phase。尾chunk安全读，不偷偷新增K对齐要求。
出处：`mxfp8:L106,L136,L165,L262`；`mxfp4_torch:L148`。

## 4. General chunk：按 threads×4B / half 有效字节推导

以下是后期 **host 参数层已验证、尚非通用 GPU 性能验证** 的策略。
前提：合法正整数配置、K32覆盖、共同chunk边界服务A/B、HTI每次推进两tile，
scale DMA每线程4B。

~~~python
W = block_threads * 4
Ra, Rb = block_m // 2, block_n // 2
D = 2 * min(Ra, Rb) * (block_k // 32)
chunk_tiles = 2 * ceil_div(W, D)
scale_row_bytes = chunk_tiles * block_k // 32
~~~

即让A/B较小half的有效数据至少填满一轮DMA，并对齐double-tile。
较大half可能需要多轮DMA，不能称“一次DMA全装下”。
外面的2是**double-tile对齐**，不是buffer slot数。

| threads / half rows A,B / BK | chunk tiles | 逻辑 chunk K |
|---|---:|---:|
| 512 / 128,128 / 128 | 4 | 512 |
| 512 / 128,128 / 256 | 2 | 512 |
| 512 / 64,128 / 128 | 8 | 1024 |
| 256 / 128,128 / 128 | 2 | 256 |

再按实际loader layout做row/slot padding、整数轮DMA及LDS检查。
若 `sa_stage_bytes/sb_stage_bytes` 定义为**一个half、一个chunk slot**：
~~~text
scale_total = 2 halves * 2 slots * (sa_stage_bytes + sb_stage_bytes)
actual_scale_dma_iters_a = sa_stage_bytes / (block_threads * 4)
~~~
历史 `ldg_sa_iters=ldg_sb_iters=0` 表示“不计入逐tile AB wait”，
**不是scale无DMA**；chunk发射数须另行推导、等待。

旧 `2*ceil((4*mma_k)/(2*block_k))` 只是保留512-K目标，没有随threads/half
泛化。新公式保留常用FP8 K128/FP4 K256方案，其余几何仍需接kernel、
测精度和性能；不能只删白名单就宣称所有HTI加速。
出处：`mxfp4_torch5:L450-L474`，原始host输出L467,L473。

## 5. Full-tile：大方阵策略必须带回退

历史FT BK128、256 rows、512 threads：有效scale1024B，一轮DMA2048B，
一半重复。四tile合并成4096B、两轮DMA，配双缓冲与MFMA traversal，
8192³约1817→2324 TFLOP/s；总LDS136→144KiB。

后续小M/短K回退3–5%，原合法配置因LDS增加被拒绝。修正策略：
1. **最短split-K分区**至少8个K tile，不能只看全局K。
2. 不超LDS，且不降低**LDS限制下**的WG驻留数。
3. 不适用则退回逐tile loader。
4. 若选择改变emitted code，进入真实编译身份；动态ABI避免隐形shape特化。

当时最初仅验证MMA16/BK128/stages2；MMA32大tile编译失败，不是硬件不支持
的证明。保留BK64 scalar、多stage、slice-K兼容路径，或明确reject未支持项。
出处：`mxfp8:L282,L302`。

## 6. Inline ASM / copy atom：以后续纠正为准

**已推翻的归因**：“必须换copy atom / ROCDL wait才能达到2800”。
后续隔离：
- 共享inline `__barrier/__waitcnt`与ROCDL基本持平。
- scale inline vs atom约0.1%，无明确收益。
- A/B回inline时曾同时丢了load前后的 `sched_barrier(0)`。
- 补回后inline与atom约2808/2809 TFLOP/s，缺fence约2619。

~~~text
sched_barrier(0)
buffer_load_lds_inline(...)
sched_barrier(0)
~~~

这是编译器调度边界，不是WG同步。旧inline本来就是global→LDS，不能解释成
“省了VGPR中转”。ISA辅证：静态waitcnt80→67、private segment56→36B，
**并非完全无spill**。要保持opcode、m0、wait、fence位置相同再比较表达式。
BF16/FT全开fence有稳定回退，不能推广到所有dtype/policy。
出处：`mxfp8_torch_bench:L191,L199,L227,L229,L245,L279`。

## 7. Non-preshuffle高收益杠杆：scalar-base addressing

~~~text
SGPR soffset = workgroup/tile 公共基址 + uniform K 偏移
VGPR voffset = lane-local row/K/swizzle 偏移
~~~

FP4 exact实验约4.40→4.73P，256 VGPR/6 spills/28B scratch →
252 VGPR/无spill；再给scale用公共基址降至246 VGPR，约4.73–4.75P，
后者单独收益较小。

要求：
- 证明值对相应wave uniform，不能把lane-dependent地址硬塞SGPR。
- 偏移统一bytes，检查FP4 packing、K-major stride、padded stride、descriptor
  range、tail安全地址；`readfirstlane`不是免费转标量。
- 不盲缓存全部地址；长live range会spill。细粒度重算也可能增加VALU，
  历史“为消一个VGPR spill重算所有A/B地址”无净收益。
- 静态exact收益不能直接沿用到动态layout版本。

出处：`mxfp4_torch3:L39,L41,L45,L65,L108`。

## 8. HTI DMA/MFMA交错：保留协议，缩短live range

官方4-wave内部只撤掉load/MFMA交错，动态工作量不变，约5.24→4.84P；
整套移到8-wave HTI却引入148/208B scratch，退到约2.99/2.05P。

安全演进：
1. 保留HTI错相barrier与wait budget。
2. 未来DMA在对应LDS **reader barrier之后**，不只看producer完成。
3. callback放MFMA中间，但不跨迭代携带多份完整fragment。
4. 分别消融AB发射、scale trigger、scale LDS→VGPR load。
5. prologue/steady/drain分别设计，最后pair不发无用scale。

历史slot仅供复现实例，**不是万能常量**：
- custom preshuffle：AB slot6/12、scale14，M-major优于N-major，
  约4.65→4.83→4.87P，后续还有5/12变体。
- non-shuffle：scale14→4仅约0.2–0.3%；更有效的是启动段
  `scale B0→B0, scale A0→A0, scale B1→B1, scale A1→A1`，
  C00内A1 data slot2/6、scale A0 slot0，其余scale slot2、其他AB slot5/12，
  约4.75→4.81P。
- slot为源码编号，核对ISA/ATT真实发射。提前到reader barrier前曾短K过、
  长K约11–12%元素错误，必须淘汰。

简化：FP4 BK256让stage0第一次AB loader顺带发scale、stage1复用K512 chunk，
删独立scale callback；历史动态版持平略好。
**共享helper不等于共享schedule**：FP8 BK128每consume8条MFMA、4 tile chunk；
FP4 BK256是16条、2 tile chunk。照搬FP4交错到FP8的多版本慢约1–5%。
FP8粗粒度拆load曾把scale LDS读取翻倍，精确去重后仍可能调度更差。
静态展开4tile以去chunk条件也曾产生228B scratch，不能只看分支数。

出处：`mxfp4_torch2:L54,L83,L151,L194,L279,L322,L365`；
`mxfp4_torch3:L240,L280`；`mxfp4_torch4:L178,L192,L242,L270`。

## 9. Scale LDS、bank conflict与最终MFMA mapping

### 少冲突、少指令、少寄存器不是充分条件

历史FP4 PMC完整bank conflict14,680,064；固定scale、省掉scale LDS读的诊断剩
2,097,152；C-only XOR可降至0，但完整kernel未加速。约86%是**冲突计数**
的scale贡献，不是86% wall time。固定scale=127改变通用输入语义，只能诊断。

Non-shuffle每WG 6144次 `ds_read_i8` 与preshuffle宽读的差异是线索。
直接b32/b64后shift/mask、lane-share、v_perm、ushort DMA可能增加live range、
scratch或串行fence；历史有零spill但更慢、读数减少却更慢的候选。

### 导出最终tiled layout，不只看atom或k_layout单层

记录 `tiled_mma.tv_layout_A_tiled`、S2R、scale lane/byte及opcode。
历史MXFP8组合 `k_layout=(16,2,4):(1,64,16)` 导出（m=0）：

| lane | A的逻辑K | 该lane提供的scale group |
|---:|---|---:|
| 0 | [0,16) ∪ [64,80) | 0 |
| 16 | [16,32) ∪ [80,96) | 1 |
| 32 | [32,48) ∪ [96,112) | 2 |
| 48 | [48,64) ∪ [112,128) | 3 |

不能用原生atom TV误说lane0连续K32，也不能假设一个lane的scale简单乘其全部
寄存器值。该emitted kernel独立probe：group0=1、group2=8，仅激活B K[64:80)，
输出128而不是16，证明最终用正确group；**不证明内部真有lane32→lane0 shuffle**。
软件无显式shuffle，严格硬件路由以对应scaled-MFMA ISA为准。
不要据此推广到其他permute/MMA/架构。

`op_sel`选择scale i32里的byte，不修复K-group错误。单byte扩i32选byte0；
打包四scale才按ABI选0/1/2/3。切换预排布须连scale layout/packing/op_sel匹配。
出处：`mxfp4_torch:L356`；`mxfp4_torch3:L108,L195`；
`mxfp4_torch2:L404,L435-L440`。中途“连续K32”等猜测以最终导出为准。

## 10. Waves / AGPR / traversal / block swizzle

- 固定256²输出，8→4 waves使每线程FP32 accumulator128→256；
  原生4-wave HTI曾180B scratch，不能只删wave数。
- AGPR要确认主循环真正驻留而非a/v往返；曾LLVM仅按单asm声明分配4/32个，
  声明完整预算后才驻留128个。零spill仍慢约4%，分组asm/保守NOP限制调度，
  不等于AGPR本身无用，也不允许任意删安全间隔。
- M/N-major、snake改变reuse和live range；官方2×2 snake不能无条件移到HTI。
- m0设置一次再递增是参考中的候选技巧，核对每wave LDS基址和时序；
  无隔离收益不能说本路径已验证。
- GROUP_M不是5P开关：一次0/1/2/4/8/16/32扫描中0最佳，1慢约6.6%，4/8略慢。
  检查XCD remap是否同时启用、小grid fallback、非整除和最后partial group。
- 更大chunk、128-bit scale DMA、cache flags、unroll、强制寄存器上限都只是
  实验轴，先查合法encoding/code size/spill。`buffer_load_dwordx2 ... lds`
  某次invalid operand且未完成，不能编成已验证优化。

出处：`mxfp4_torch:L161,L206,L679`；`mxfp4_torch2:L28,L54,L365`；
`mxfp8:L165,L189`；`mxfp4_torch4:L43`。

## 11. Preshuffle、direct-B与epilogue分别计成本

- 逐项记录A/B/SA/SB谁重排和ABI版本；custom不等于官方shuffle格式。
  静态权重可缓存，activation scale可能每次准备；分开kernel-only、
  含动态准备、量化融合链路，不能独立时间机械相加。
- FT preshuffled B→register可省LDS、放大合法tile/stages；
  大tile更重reuse时LDS可能更快。HTI没实现direct-B不能强制换FT冒充优化。
- Direct C store用lane exchange/permlane拼128-bit，省C LDS往返；
  AB+scale主导union时**不降低LDS分配**。
- 比direct early / direct late / C-shuffle；若同时恢复accumulator ownership、
  末轮overlap，就只能称整套策略而非store指令消融。
- C-shuffle前对齐HTI phase、union覆盖前等最后AB读者；
  历史某ownership需group0补barrier，不是所有HTI固定给group0补。
- 实验preshuffle direct收益约4.6%，后期动态版8192³约0.5%；短K/尾块可反向。
  用户优先简洁时可统一C-shuffle，但记录代价。
- epilogue lag2（完成行组2再写0）曾仅省0.19–0.47µs；
  用8B store替代lane exchange+16B反而慢，微收益不值得默认复杂化。

出处：`mxfp8_aiter:L474`；`mxfp4_torch2:L196,L232,L236`；
`mxfp4_torch3:L292`；`mxfp4_torch4:L75`；`mxfp4_torch5:L266,L274,L285`。
