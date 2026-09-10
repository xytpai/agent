# 优化交付报告模板

## 目标和契约
- 数学式 / dtype / partial精度 / layout / scales / bias：
- 用户目标和不允许改变的行为：
- 目标 shapes + 回归矩阵：
- 验收阈值（如何处理测量噪声）：

## 环境与 baseline
- GPU arch / UUID或PCI / visible index / profiler agent：
- Torch/HIP/FlyDSL/LLVM/ROCm版本：
- git commit + dirty state + source/dependency SHA：
- baseline是否为当前工作区：
- baseline/candidate依赖隔离方式：
- cache/dump目录（不能覆盖）：
- clocks/power/温度/并发状态，是否锁定且已恢复：

## 根因证据链
| 假设 | 源码/地址推导 | IR/ISA变化 | trace/counter | 单变量计时 | 判定 |
|---|---|---|---|---|---|
| | | | | | 假设/支持/证实/排除 |

- counter单位、维度和aggregate方式：
- static instruction count如何按K tile/动态次数归一化：
- 已排除：spill/occupancy/host gap/无关展开等（只列真正验证过的）。
- 未解释的剩余差距：

## 最小补丁
- 修改文件/函数/行：
- 精确适用条件：
- 未改的路径/ABI/精度：
- 是否增加specialization/cache key/host开销：
- 是否需要fallback；理由与收益/代价：

## 同步/内存/数值证明
- 每个LDS区间的producer/last reader/reuse phase：
- 每个wait/barrier/fence的目的：
- split-K publish/observe/finish/reset：
- lane mapping与MMA ABI/G2S/S2R的一致性：
- precision/reference/容差依据：

## 正确性结果
- 命令（fresh cache）：
- passed / failed / xfailed / skipped / deselected：
- 新引入失败 vs baseline已有：
- shared consumers回归：
- exact / dense / guard / pressure / stream / Graph：

## 性能结果
- 测法：Graph/eager/kernel trace/host wall-clock，哪些开销计入：
- warmup/rounds/launches/slots/working-set/seed/顺序：
- 是否无profiler测量；baseline/ref/candidate同条件：
- raw数据文件：

| shape / config | baseline µs median [p10,p90] | candidate | latency变化 | 重复运行 | 结论 |
|---|---:|---:|---:|---|---|
| primary | | | | | |
| short K / small M | | | | | |
| rectangular / boundary | | | | | |

- 显式列出退化与用户是否接受：
- identical ISA（若适用）验证方式；不能据此推断host性能：

## 可复现产物
- sources / patch / manifest / versions：
- command log / raw JSON / PMC CSV / trace / ISA / pass IR：
- 持久存放位置（关键证据不要仅留/tmp）：
- 未验证范围、后续工作：
- 本地已更新？commit？push？（分开写）

## 完成检查
- [ ] 用户当前改动未覆盖、GPU状态未擅自更改
- [ ] 不改变数学语义/支持范围来伪装优化
- [ ] 根因有单变量实验与实际指令/计数证据
- [ ] final patch而非中间候选通过fresh-cache tests
- [ ] 无profiler多轮A/B、原始样本与执行顺序保存
- [ ] 两版本AB/BA确实交替，shared baseline依赖确实隔离
- [ ] short-K/其他路径回退没有隐去
- [ ] known失败明确、非宽泛xfail、未声称100% runtime branch coverage
