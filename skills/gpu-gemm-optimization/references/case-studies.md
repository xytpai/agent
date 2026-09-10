# 历史案例：gfx950 GEMM（不是当前性能承诺）

这些是本次开发会话的**历史实验**。后来仓库可能重命名/删除了旧 kernel；
这里的 `fp8_ptpc` 表示当时保存在快路径中的对照，不代表当前仓库一定可 import。
测量条件、raw timing samples 和 PMC CSV 选列已固化在
[case-data.json](case-data.json)。其中记录源 artifact 路径与 SHA-256，
不要求 `/tmp` 永久存在；完整代码/IR 仍需实验目录或对应源码版本才能复现。

机器时钟记录与会话日期可能不同，本文不以它们宣称最新数据。
测试数量也是当时运行结果，不是未来 commit 的保证。

## 1. “scaled 约374us，PTPC 约360us”如何定位

同设备、相同 NT/HTI/256x256x128/2 stages/2x4 waves、
8192³、15 rotary slots (~4 GiB)、每样本128 graph launches。
独立编译、同输入输出、多轮比较。

### 不成功/次要的消融同样重要

第一组同时跑的中位耗时（µs）：

| 变体 | 中位耗时 | 说明 |
|---|---:|---|
| baseline | 380.088 | 当时 scaled |
| traversal only | 377.153 | 约3us，不解释全部 |
| batch fence only | 380.652 | 无明确收益 |
| soffset only | 377.313 | uniform K offset 改为标量地址 |
| DMA combined | 377.191 | 与上项相近 |
| unroll only | 380.259 | 不是主要原因 |
| reference PTPC | 363.564 | 目标差距仍存在 |

ISA：两者 LDS=128 KiB、spill=0，VGPR 248/246 不跨已观察占用档位。
regmem-to-SSA 后稳态无所谓的“rmem spill”；不能把性能差怪给 `fx.gemm` 风格。
即使后续组合让稳态 instruction count 接近，仍不能解释全部差距。

### PMC 给出关键线索

每 dispatch counter 均值（4次采样；单位按当时 counter Description）：

| counter | baseline | 仅改 K permutation | reference |
|---|---:|---:|---:|
| SQ_INSTS_MFMA | 16,777,216 | 16,777,216 | 16,777,216 |
| SQ_LDS_BANK_CONFLICT | 52,428,800 | 2,097,152 | 2,097,152 |
| SQ_WAIT_INST_LDS | 45,072,222.25 | 25,789,880.50 | 23,501,462.50 |

lane/table 推导定位到：
~~~python
# before: each lane reads adjacent K+0 / K+16 16B segments
k_perm = fx.make_layout((16, 4, 2), (1, 16, 64))
# after: lane group steps 16B, each lane reads K+0 / K+64
k_perm = fx.make_layout((16, 2, 4), (1, 64, 16))
~~~

与当时 `Swizzle<3,4,4>` / `ds_read_b128` 配合，主体 LDS bank conflict
下降96%，issue wait下降约43%。A/B 同时改相同 permutation，数学结果不变。
这是一条 **mapping 推导 + 单变量 counter + 性能 + 正确性** 证据链。

后续只保留三处局部优化：
- 上述 permutation；
- 默认 hardware scale lowering 成无 scale 操作数 MFMA，去掉显式 identity VGPR；
- HTI traversal 改 serpentine。

最终25轮同测法：**379.117 → 364.977 µs**；reference **363.309 µs**。
没有采用大规模 DMA/loop/epilogue 重写。

**限制**：这不是某行代码普遍等价的证明。默认 atom scale、TV layout、
block_k、MMA shape、layout、compiler 变了都要重新验证。
历史计时 harness 的双版本轮转+翻转存在顺序陷阱；主要结果含多个版本且有反复
消融，但未来复测必须使用 skill 附带的已测试公平顺序，不复制旧 harness 缺陷。

## 2. 把 HTI epilogue 改成参考中的交错方式

原来：完成最后全部 MMA → 四个 C half tiles 写 LDS → 同步 → global。
实验：C00/C01 写 LDS 与 consume(C10) 交错；
C00/C01 写 global；C10 写 LDS 与 consume(C11) 交错；最后逐块写回。

**不能直接删 wave-group phase balancing**：
当前 tiled `partition_C` / global-store map 跨两个 M groups；
参考版本的 ownership 不同。literal translation 在 NN/32x32x64 下失败：
3181/20736 元素不一致（15.3%），max abs diff 1.25。

最终保留相位对齐 + 参考交错，完整测试当时为
707 passed / 6 baseline xfails / 3 benchmark deselected。

| shape | before µs | after µs |
|---|---:|---:|
| 8192³（一次运行） | 369.522 | 366.706 |
| 256³（另一次运行） | 8.904 | 9.103 |

VGPR 246→252、LDS不变、无spill。大矩阵改善约0.76%，短K回退约2.23%。
**结论只能是 tradeoff，不能说“全面无退化”**。
reference barrier schedule 是与 lane ownership 配套的设计，不是可随意复制的模板。

## 3. 从测试中发现的问题

- 无效 block_k=64 / MMA-K=128 不应放入 accuracy suite 后跳过；
  改为显式 policy rejection，合法 MMA32/K64 另测。
- exact sparse FP8 + binary scale + output guards 能抓到宽 rtol 漏掉的错误。
- 图形 API 的“contiguous”可能允许 singleton stride2，但 DSL 不允许。
- 编译 abort 要子进程隔离，否则第一个 bad policy 杀死整个回归矩阵。
- split-K FP32+bias 初始化压力测试暴露过 opaque inline-ASM store 问题。
  更保守 fence 通过正确性却明显拖慢 split-K；改用可见 store 的针对性修正后
  再检查 ISA 与压力测试。此案例不能被解读为 volatile store 普遍替代 release。

## 4. 提炼为 skills 的经验

1. 优先实测 counter；不要先长时间枚举无关 scheduling 改动。
2. 所有“技巧”都有 applicability：相同 swizzle 对不同 lane map 并不等效。
3. 源码简化目标可以通过原地 SSA list 实现，不必变成 GPU memory。
4. 性能结论要带对象：kernel、shape、路径、measurement mode、版本。
5. 最终数据保留失败实验、短K回退与 known bugs，便于后人证伪。
