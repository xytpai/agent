# GPU / GEMM 优化 Skill

适用于 GPU kernel 性能优化与根因分析，尤其是 FlyDSL/ROCm/gfx950 GEMM、MXFP4/MXFP8/E8M0、FP8/BF16、HTI、full-tile、split-K/slice-K、LDS swizzle、ASM/trace 分析及无回归重构。

完整技能入口：[SKILL.md](gpu-gemm-optimization/SKILL.md)。

使用时先读取该入口，再按任务读取 references；scripts、examples、tests 均位于同一技能目录，文档相对链接保持不变。

MXFP 专项入口：[优化技巧](gpu-gemm-optimization/references/mxfp-optimization.md) ·
[调参/集成/验证](gpu-gemm-optimization/references/mxfp-integration.md) ·
[历史案例与归因纠错](gpu-gemm-optimization/references/mxfp-case-studies.md)。

自包含约束：方法、案例、证据和示例保存在 agent 库内；不依赖其他源码仓库、
私有 helper 或历史实验目录。原日志仅作可选溯源，工具依赖安装的 SDK/库另行说明。
