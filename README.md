# VLA Robustness Lab

[![Repository checks](https://github.com/unno11/vla-robustness-lab/actions/workflows/check.yml/badge.svg)](https://github.com/unno11/vla-robustness-lab/actions/workflows/check.yml)

面向视觉扰动的 VLA 策略复现、鲁棒性评测与轻量适配。以 LIBERO 为初始仿真平台，提供可追溯的配置、运行入口、实验记录与部署说明。

**当前状态：独立环境与仿真诊断部署中。已提供有界取图脚本；尚未通过仿真验收，未完成策略运行、基准评测或训练，没有本项目成功率或改进结论。**

## 范围与模型

| 组件 | 用途 | 当前状态 |
| --- | --- | --- |
| LIBERO | 原始任务与原生成功判定 | 候选源码版本已记录，环境待验证 |
| OpenVLA-OFT | 使用上游配套检查点建立首个基线 | 权重与评测入口已查证，本仓库运行待验证 |
| 视觉扰动评测 | 比较原始与受控扰动条件 | 协议待确定；可参考 LIBERO-plus |
| SmolVLA | 后续训练与数据增强对照 | 可选扩展，未安装、未训练 |

初始资源目标为单张 RTX 4090 24 GB、Linux、无实体机器人。该硬件目标尚未通过本仓库实测。OpenVLA-OFT 的模型推理显存参考不等于策略与仿真合计峰值；默认训练配置不能假定适用于24 GB。

## 快速开始

目前可复验的是仓库自身检查，无需GPU或下载权重：

```bash
git clone https://github.com/unno11/vla-robustness-lab.git
cd vla-robustness-lab
python scripts/check_repository.py
```

Python 3.10及以上。此命令检查文档链接、配置和Python语法，不启动仿真，也不证明策略评测通过。

GPU部署入口正在准备，见[部署与验证状态](docs/deployment.md)。当前不提供未经实测的“一键完整复现”命令。

## 文档

| 文档 | 内容 |
| --- | --- |
| [部署与验证状态](docs/deployment.md) | 支持目标、上游入口、环境隔离和首条rollout验收 |
| [系统关系](docs/architecture.md) | 任务、观测、策略、动作执行和结果之间的关系 |
| [评测与实验协议](docs/evaluation.md) | 固定条件、数据划分、原始结果及比较口径 |
| [技术路线](docs/roadmap.md) | 面向项目交付的技术里程碑与验收 |
| [开发流程](CONTRIBUTING.md) | Issue、变更、验证、证据和发布流程 |
| [当前状态](results/status.json) | 机器可读的验证状态；没有占位成功率 |
| [第三方来源](THIRD_PARTY_NOTICES.md) | 上游代码、数据和检查点来源及许可边界 |

## 预期交付

- 固定版本的安装与运行说明，以及经过实测的入口。
- 原始LIBERO基线和单项视觉扰动对照，完整保留失败与未完成记录。
- 成功率、失败阶段、预测延迟、执行时间和显存测量。
- 若完成训练，提供训练配置、检查点来源和独立验证结果；若没有训练，明确标注仅使用上游权重。

LIBERO-plus、OpenVLA-OFT和SmolVLA已有方法均归属上游。本仓库后续贡献以实际完成的实现和实验为准；不将上游成功率写成本项目成绩，不预设实验必然改善。

## 复现资源

权重与数据通过上游入口获取，不直接存入Git。仓库记录来源、revision、必要的校验信息及配置；本地原始日志、观测和检查点与公开结果摘要分开存放。资料发布遵循各自许可。本仓库原创代码和文档暂未指定通用开源许可证。

论文复现范围已扩展至 OFT 的全部四套 LIBERO 主评测；完整原始数据和论文 RLDS 数据正在下载。具体覆盖、前置条件与限制见[论文复现范围](docs/paper-reproduction.md)。
