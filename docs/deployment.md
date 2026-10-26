# 部署与验证状态

## 当前可用范围

本仓库处于初始化阶段。当前仅仓库静态检查可直接运行，尚无通过实测的GPU环境安装锁文件或本仓库策略运行入口。以下是部署验证顺序与验收条件，不是已完成的安装教程。

| 项目 | 目标或已知上游条件 | 本仓库实测 |
| --- | --- | --- |
| 仓库检查 | Python 3.10+，无需GPU | 持续集成检查 |
| 策略与仿真 | Linux / NVIDIA GPU，单张4090 24 GB为目标 | 待验证 |
| OpenVLA-OFT版本 | [候选源码版本](../configs/source_versions.json) | 已记录来源，未验证兼容性 |
| 上游评测依赖 | 官方说明为Python 3.10.14、PyTorch 2.2.0及定制transformers | 待在隔离环境验证 |
| 模型显存 | 官方两路图像加状态的LIBERO推理参考约16.2 GB | 不能代替策略与仿真总峰值 |
| 训练 | 默认OFT最低配置参考25.6–25.7 GB；SmolVLA为后续候选 | 未训练，不承诺默认OFT配置适配24 GB |

参考[上游LIBERO说明](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md)和[资源FAQ](https://openvla-oft.github.io/#frequently-asked-questions)。CI使用的Python版本只用于检查文档与仓库，不作为策略依赖环境。

## GPU部署的验证顺序

1. 固定OpenVLA-OFT、LIBERO和依赖版本，在独立环境安装；不复用其他仿真项目的环境。
2. 独立验证LIBERO任务初始化、重置、相机取图和正常退出，记录渲染后端。
3. 从[官方任务套件权重列表](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md#launching-libero-evaluations)取得匹配检查点，记录revision与实际下载文件摘要。
4. 先离线预测，核对相机、状态、归一化、动作输出及训练和推理预处理一致性。
5. 在固定任务与初态执行一次有界闭环；保存真实预测块、实际执行动作和原生成功/超时状态。
6. 通过后再发布经过实测的安装步骤、依赖锁和运行入口，然后进行整组评测。

任务套件、checkpoint ID及revision尚未冻结，见[基线协议草案](../configs/baseline.draft.json)。草案不是可运行配置。

## 版本与数据管理

上游当前HEAD只作候选版本；不能仅因已经记录SHA就称其兼容性已验证。首次实际安装后更新验证状态，版本变化需要记录原因。

只评估上游权重时不必先下载完整训练数据。轻量适配开始前另行固定数据revision和划分。所有路径通过将来实现的本机配置覆盖，公开文档不依赖原开发服务器目录。

LIBERO-plus与原LIBERO可能提供同名Python模块；后续采用独立环境，避免在基线环境内原位替换造成对照污染。具体依赖方案仍需实测。
