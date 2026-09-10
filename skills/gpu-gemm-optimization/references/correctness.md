# GEMM 正确性与回归门禁

## 1. 定义数值契约

`C = (A @ B) * scale_a[:,None] * scale_b[None,:] + bias` 与
“先把 bias 放进 accumulator 再 scale”不等价。

分别记录：
- A/B storage dtype；MFMA accumulate dtype；hardware scale 的格式/默认语义。
- C-shuffle dtype；local slice partial、global atomic partial 的 dtype。
- 最终 output dtype 不代表 intermediate dtype：FP32 output 可能经过 BF16 partial。
- bias 是哪种 dtype，是否仅在一个 slice / split initialization 中加入。
- NaN/Inf/overflow、denormal、rounding 的预期。普通随机测试未覆盖这些就不能声称覆盖。

**不要把 FP8 E8M0 编码常量当通用 identity**：检查当前 atom lowering、op_sel、
ISA 和边界值。某版本默认状态会 lowering 成无 scale 编码，这是该实现的证据，
不是 “所有硬件 scale=0 都表示 1”。

## 2. 最低路径矩阵

| 维度 | cases |
|---|---|
| pipeline | full-tile、HTI |
| MMA | 支持的 16x16x128 / 32x32x64 / 其他 |
| layout | NN、NT、TN、TT（逻辑 shape 不变） |
| reduction | no split、split2/4/更大、slice2/4、split+slice |
| output/bias | BF16/FP32 × bias off/on |
| K control flow | 最小合法、drain-only、main、wrap、bulk/remainder、多个 MMA K steps |
| split partitions | 相等、不等但合法、最后一段太短/空/不对齐（拒绝） |
| M/N | 小于 tile、完整 tile、partial、多 block、小 M/N、最小输出向量宽度 |
| memory layout | contiguous、不同 padded leading stride、aligned storage offset |
| scales/bias | strided、negative/zero/non-unit，singleton non-unit-stride |
| output | preallocated/allocate、inferred dtype、flat/invalid stride、首尾 guard |
| mapping | group_m=0/正值、阈值前后、XCD 整除/非整除、最后 partial group |
| runtime | 首次调用/缓存、dynamic shape/stride/split value、两 stream、Graph replay |

这不是无限笛卡尔积：用小 policy 做全交叉，用代表性大 policy 测资源/性能。
不支持的组合应 assert rejection，不能一边把它当合法 case，一边 skip 掉。
例：block_k=64 配 MMA-K=128 应拒绝，不等于 block_k=64 一律不支持。

## 3. 构造强 oracle

### 精确探针

- A/B 选 FP8 可精确表示的 -1/0/1 或小整数。
- A 每隔 16/32 个 K 位置非零，限制 partial 幅度。
- scales 选 0、±1/4、±1/2 等二进制数；bias 也为二进制小值。
- **证明每个 partial、加 bias 后结果和中间 reduction 都可精确表示**。
  不能只是因为输入是整数就断言任意 K/幅度都允许 atol=0。
- 与独立 FP32 matmul reference 比较 `atol=rtol=0`。
- 全零 A + 非零 bias：验证 bias 恰加一次。
- lane/K 改动用 index-coded/稀疏 impulse 检查位置，不只 uniform ones。

### 稠密随机

多个 seed、正负值、真实量级；独立 reference。
依据 partial dtype 和 K 估算并验证误差范围，不能沿用很宽的 rtol=.5 掩盖漏算。
减小尺寸以保持参考成本可控；大 shape 可额外比较 baseline，但不能只有互比。

### 内存检查

- 每次执行前 `out.fill_(nan)`，抓未覆盖元素。
- storage 首尾 sentinel，`out = storage[guard:-guard].view(M,N)`。
- 输入 padding 填 poison，aligned offset view 抓错 stride。
- guards 是有用证据但不是完整 OOB/race detector；工具支持时加 sanitizer。
- 别把 FP8 元素数与 byte offset 混淆。

## 4. 异步 pipeline / LDS lifetime

修改 HTI 尾部前画：
~~~text
buffer region | 当前拥有者/producer | 最后 reader | 可复用 phase
AB stage0/1    | DMA                | LDS reader  | ...
C00/C01/C10/C11| MMA epilogue       | global store| ...
~~~

每条 C LDS store 前需确保：
- 原 AB 对应地址的 reader 完成；
- 本 quadrant accumulator 完成所需 MMA；
- 所有 consumer wave 在正确 phase 获取数据。

每条 global store/atomic 前需确保：
- 对应 quadrant 的 LDS producer 完成；
- split-K C/bias initialization 对其他 workgroup 可见；
- shape guard/vec width 不越界。

**参考 epilogue 的顺序依赖其 C layout、thread ownership 和 barrier phase**。
不能只复制 call 顺序。新版本高 VGPR / 多 barrier 可能改善大 K 却拖慢短 K。

## 5. Split-K / slice-K 专项

生命周期：
~~~text
分配或复用 stream-local sync buffers
-> ks0 初始化 C 或 bias
-> 完成并发布 signal
-> 所有 partitions 观察 signal 后 atomic-add
-> 确认本组全部输出完成
-> 最后到达者 reset state
-> 下一个调用/Graph replay 可复用
~~~

- 检查具体 arch 的 cache policy、wait 和 atomic scope/order。
  barrier 不是自动的跨 workgroup publication。
- inline ASM 如隐藏了 memory 或 operand dependency，优化器可能错误重排。
  优先 IR-visible store/atomic；但 volatile/nontemporal/cache flag 不是通用
  release/acquire 等价物。需要内存模型、实际 lowering、pressure tests 三者。
- 全局 fence 常能保守修复，但可能带来巨大 cache flush 成本。
  不能只 correctness 通过就采用；也不能为了性能省掉必要同步。
- local slice reduction 先 scale/round 再 sum 与先 sum 后 scale/round 不等价。
- semaphore capacity 与 `M_tiles*N_tiles` 对齐检查；不能只看 split 数。
- 重复不同 shape/policy/split value，检查 dispatcher cache 和 state reset。
- 两 stream 使用不同 buffers；显式/当前 stream 两种 API 路径。
- Graph 保留所有 tensor 引用；修改原 scales/bias 后 replay，确认 capture 中
  contiguous materialization/参数地址仍正确。

## 6. 负面测试

不合法 input dtype/rank/shape/device、scales shape/dtype、output shape/stride/dtype、
bias shape、unknown kwarg、M/N<=0、K-tail、DMA 不对齐、LDS 容量、wave 数、
MMA coverage、HTI stages/k_waves、split 最后一段、semaphore capacity。

避免测试本身启动负 grid 或越界 kernel：先验证 host 会拒绝。
Host-only tests 可 mock arch 容量，不能 mock 掉需要验证的计算/同步路径。

Singleton 陷阱：Torch `is_contiguous()` 对长度1 stride2 的 view 可返回 True；
FlyDSL leading dimension 仍要求实际 stride1。测试这个边界，而不是假定
`.contiguous()` 必定产生新 unit-stride tensor。

## 7. 编译器 abort 和已知缺陷

- 可能 SIGABRT 的 path 用 subprocess + timeout + core dump 限制，
  记录完整 stderr、policy、IR 和环境。
- 只有已确认的**基线已有** bug 可 strict xfail；精确限定 case 和 error pattern。
- 任意异常、超时、其他 signal、数值失败不能被宽泛 xfail 吞掉。
- XPASS 必须使 suite 失败，修复后移除 marker。
- 不报 “all passed” 隐去 xfailed/skipped/deselected。
- 本仓库曾观察到 TN/mma16/slice4 LLVM dominance failure；未来 compiler
  可能已修复，不能预先将所有 slice4 都当失败。

## 8. 最终命令与结论边界

~~~bash
HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_CACHE_DIR="$RUN/cache-final-fresh" \
pytest -q -rx test_scaled_gemm_gfx950.py -k 'not benchmark'
~~~

此命令排除了原 benchmark，必须在报告写明。
Python branch coverage 对 tracing DSL 不等价于 GPU runtime 分支覆盖；
只有执行了对应 policy/input 的 kernel 才能声称该路径已测。
共享 utils 变动要复测 A16W16 等所有消费者。纯测试请求不默认改 kernel。
