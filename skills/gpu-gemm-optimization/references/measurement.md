# 基线与测量协议

## 1. 保存现场

从 `agent/skills/gpu-gemm-optimization` 执行以下库内诊断示例。
实际任务的实现、adapter、依赖快照也应先保存到 agent 内，不依赖其他工作区。
不要盲目占满 GPU；下面只创建目录/记录环境，未执行GPU基准。

~~~bash
mkdir -p ../../temp
RUN=$(mktemp -d ../../temp/gemm-opt.XXXXXX)
export RUN
git status --short | tee "$RUN/status.txt"
git rev-parse HEAD > "$RUN/commit.txt"
git diff --binary > "$RUN/working-tree.patch"
# git diff 不含 untracked 文件，必须按实际依赖另外复制。
cp examples/torch_scaled_demo.py "$RUN/baseline.py"
cp examples/scaled_gemm_adapter.py "$RUN/adapter.py"
sha256sum "$RUN"/baseline*.py > "$RUN/source.sha256"
date -Is > "$RUN/time.txt"
rocm-smi --showuse --showclocks --showpids > "$RUN/gpu.txt" 2>&1
python - <<'PY' > "$RUN/python-env.txt"
import sys, torch
from importlib.metadata import PackageNotFoundError, version
print(sys.version)
print("torch", torch.__version__, "HIP", torch.version.hip)
try:
    print("flydsl", version("flydsl"))
except PackageNotFoundError:
    print("flydsl", "not installed; optional for this diagnostic")
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0))
PY
~~~

再保存 `hipcc --version`、`rocprofv3 --version`、FlyDSL/compiler 构建 SHA。
选项不兼容先查 `--help`，记录失败，不编造结果。visible index 与 profiler
agent/device index 可能不同，必须映射到同一物理 GPU。

**baseline 隔离**：
- 只改 kernel 且依赖不变：可以用不同 module name 加载两份源文件。
- 改 shared utils/compiler：两份 module 可能仍共用 candidate 依赖；
  需要完整快照/独立 worktree、cache、环境，按 ABBA 顺序重复子进程测量。
- 不修改全局安装包、不清空用户的 JIT cache。
- 不与编译压力/正确性/其他 GPU 作业并行跑基准。
- clock 锁定需授权、记录和恢复；没锁 clock 就如实说明。

## 2. 分清测量目标

| 测法 | 包含 | 适合回答 |
|---|---|---|
| GPU profiler kernel duration | 单次 device dispatch | 哪个 kernel 慢 |
| HIP Graph + events | graph replay/device execution | 稳定 device-side A/B |
| Python launch loop + events | GPU elapsed，可能混入提交空洞 | 当前提交模式的实际延迟 |
| CPU wall-clock + synchronize | host + device + synchronization | API 端到端延迟 |
| 首次调用 | JIT/alloc/warmup 等 | 冷启动，单独报告 |

events 必须放在正确 stream 并等待完成。Graph 结果不代表 host 延迟；
profiler 插桩耗时不用于最终 speedup。

## 3. 推荐 A/B 流程

1. 同 GPU/输入/shape/layout/stride/dtype/policy/输出地址复用方式。
2. 预分配、编译、实际 launch 两个版本；固定随机种子。
   实际运行时的首次缓存填充可能执行 kernel，不能默认视为纯编译。
3. 计时外验证独立 oracle；两份输出相等本身不保证它们都正确。
4. warmup 所有 slot；split-K buffer 初始化、非连续输入物化方式要一致。
5. capture 内不引入首次 JIT/非法分配，所有 captured tensor 保持存活。
6. 8 次 warmup replay 是起点，不是固定标准；确认 clock/温度稳定。
7. 每样本 64–256 launch，15–25 轮，至少再独立运行一次。
8. 两版本 AB/BA 交替。N 版本轮转+镜像。
   **陷阱**：用 `r % N` 轮转同时 `r % 2` 翻转，N=2 时可能永远 AB！
   附带脚本用 `(r // 2) % N` 轮转、奇数轮翻转。
9. 保存 raw samples；报告 median、p10/p90、MAD，查看 paired latency ratio。
   不静默丢弃离群点；记录它是否与 profiler/clock/并发对应。
10. 噪声容忍度先约定。小于离散度且跨轮不一致的差别按持平处理。

## 4. Hot / rotary / cold

Rotary 不保证 cold cache。明确 working-set bytes、slots、allocation、
输入输出复用和 split-K read-modify-write 对 cache 的影响。

NT FP8 + BF16 每 slot 粗算：
~~~text
M*K + N*K + 2*M*N + 4*(M+N) bytes
~~~
再加 bias、工作区等。8192³ 的 15 slots 接近 4 GiB。
名为 “8 GiB target” 的变量不表示 resident footprint 为 8 GiB：读 slot 公式。
小 shape 限制 slots，否则 host/capture 元数据过大。

## 5. 可运行的库内 A/A 诊断

从 skill 目录执行：
~~~bash
HIP_VISIBLE_DEVICES=0 FLYDSL_RUNTIME_CACHE_DIR="$RUN/cache-ab" \
python scripts/benchmark_ab.py \
  --adapter examples/scaled_gemm_adapter.py \
  --baseline "$RUN/baseline.py" \
  --candidate examples/torch_scaled_demo.py \
  --shape 64 64 64 --slots 2 --rounds 4 --launches 8 --warmup 2 \
  --mode graph --output "$RUN/ab.json"
~~~

这两份实现内容相同，是BF16 matmul后行列缩放的**A/A接线诊断**，不是MXFP、
HTI或历史kernel，也不能证明原kernel性能。所用代码全部在agent内。
实际路径另写agent内adapter和oracle；shared dependency改动不能靠同进程
模板证明非回归。先small-shape smoke，容量检查后再申请rotary buffers。

## 6. 回归矩阵

| 类别 | 至少包含 |
|---|---|
| 目标 | 用户 shape，原 policy |
| 小工作量 | 小 M/N、最小合法 K |
| 循环边界 | drain-only、remainder、bulk+remainder、stage wrap |
| 形状 | square、rectangular、partial M/N |
| 模式 | patch 影响的 layouts/MMA/stages/split/slice |
| API | Graph + 实际调用模式；涉及 host 时测 wall-clock |

`latency_change_pct = (candidate / baseline - 1) * 100`，负值更快。
它不等于 throughput speedup，不混用两个百分比。
