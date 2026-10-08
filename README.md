# Frozen Exit Adapters

冻结语言模型的末端宽 FFN 出口：**表示拟合 ≠ 连续生成能力**。

本仓库对应中文审阅稿《从表示拟合到连续生成：冻结语言模型末端宽前馈适配器的实证研究》的宽 FFN 主实验及固定步数补充实验。主干、最终 RMSNorm 和 lm_head 全部冻结，仅训练残差适配器。原阶段保留深度 60–63，每层 16 种结构，共 64 次筛选 + 8 次验证集选择后的继续训练；补充阶段完成24组固定步数对照及全部63层候选的生成评测。

## 主要观察

| 配置 | 数学 / 100 | 代码 / 99 |
|---|---:|---:|
| 完整教师 | 94 | 71 |
| ValMSE-selected-60 | 17 | 37 |
| ValMSE-selected-61 | 62 | 38 |
| ValMSE-selected-62 | 95 | 43 |
| ValMSE-selected-63 | 92 | 64 |

ValMSE-selected仅指向量验证 MSE 最低的候选，不按生成测试分数挑选。8 组继续训练均无 patience 早停，但后期训练误差下降、验证误差上升。测试集此前已使用，结果是探索性的；原64组筛选没有多种子重复；新增关键对照使用3种子。没有跨模型实验或实测部署加速结论。

## 2026-10-08 补充实验

新增 **24组固定12000步训练、43配置×199题自由生成、80组轨迹配对**，全部完成；数据包含三种子仿射/非线性和同参数量无激活对照。核心对照的63层SiLU-1/SiLU-2代码均值为63.0/99；同参数量Linear-1/Linear-2为47.3/99和47.0/99；教师为71/99。数学增减方向并非在所有结构中一致。

全部16个原63层screen候选的归档BF16权重已评测。归档前验证MSE与代码成绩的描述性Spearman相关为−0.67，但最低MSE不对应最高任务成绩。单种子的4×20题轨迹中，同前缀重放top1一致率约95%–98%，自由序列完全一致率10%–25%；分叉后误差无统一上升，且相同前缀重放控制非零，不能据此确认因果机制。

分析从`main/supplement/results/`数值记录重算统计，生成新版8图及源数据；连同原阶段保留的图1/2，总计10张稿件图。该目录的生成表已移除提示、回答、参考、测试与token IDs；轨迹表保留逐token标量，不含词表ID。可以复算分析，不能重新执行回答判分。权重、大型向量和未删减回答均不公开。

`main/supplement/`为实验源代码；运行前需要按`init_run.py --help`创建新预算。它只创建预算文件，不租机器、连接远程或关机。完整16候选评测还需要原screen检查点，仓库未包含。新增环境报告了Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 5.18.0；模型不可变revision仍缺失。固定步数不等于等FLOPs或充分收敛，既有测试集复用且没有新独立盲测。

## 无需 GPU：重算统计和 10 张图

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python paper/analysis/make_figures.py
python paper/analysis/redraw_figures.py --overview-only
python paper/analysis/check_results.py
python paper/analysis/analyze_supplement.py
python paper/analysis/plot_supplement.py
```

前两步重算原阶段CSV与摘要并排版保留的图1/2；第三步核验原结果。最后两步重算补充统计与新版图3–10。输出位于`paper/figures/`、`paper/analysis/derived.json`及`paper/analysis/revision_20261008/`；未引用的旧阶段图作为历史分析保留。`results` 中的生成记录是公开分析表，保留逐题 ID、判分和诊断，去除了题目文本、参考程序与回答文本。可重算本文成绩和图表，不能用这些表重新执行回答判分。

## 代码和数据

- `main/prepare_data.py`：构建 5000 提示、4000/500/500 分组划分。
- `main/collect.py`：冻结教师生成和因果重放，保存同位置的各层表示。
- `main/wide/core.py`：SiLU、GELU、双隐层 SiLU、SwiGLU 残差出口。
- `main/wide/train.py`：仅保存向量监督、固定活动时间、验证选择。
- `main/wide/evaluate.py`：各深度连续生成，含实际执行层验证。
- `main/evaluate_baselines.py`：教师与直接退出 60/63 层基线的独立运行入口。
- `main/judge.py`、`main/code_worker.py`：数学精确匹配与受限 Python 测试。
- `main/wide/results/`：72 组拟合曲线、8 组向量测试及 1990 条生成判分。
- `main/results/`：复用的教师与直接退出基线，共 597 条生成判分。
- `main/data_manifest.json`：来源、划分统计及原始文件 SHA-256。

## 重新运行 GPU 实验

原实验为一张 A800 80GB、BF16 主干、隐藏维度 5120、64 个混合层块。需要兼容 `Qwen3_5ForConditionalGeneration` 的 Transformers，以及本地 Qwen3.8-27B 权重。**模型不可变revision缺失，不能承诺字节级复现。** `requirements-experiment.txt`是通用依赖列表；`requirements-experiment-reported.txt`记录补充阶段实际pip freeze的版本，未验证为可移植安装锁文件，原筛选环境没有完整锁定。

1. 安装实验依赖，设置 `MODEL_PATH` 为已下载模型目录。代码不自动下载模型。
2. 执行 `python main/download_data.py`，仅接受与归档 SHA-256 一致的 GSM8K、MBPP、Code Alpaca 文件；随后执行 `python main/prepare_data.py`。各数据集条款和来源见 `DATA.md`。
3. 执行 `python main/collect.py`。此脚本保持原采集方式，保存全部 65 个位置边界；需要预留约 107 GB 激活文件空间，以及模型和检查点空间。不会自动租机器。
4. 执行 `python main/evaluate_baselines.py`，生成新的 `main/run/supported_evaluation.json`。原环境排除了参考执行失败的 MBPP 123，留下 199 道测试题；若新环境题集不同，停止并记录差异，不能静默替换本文分母。
5. 在空的运行目录上执行 `python main/wide/init_run.py --training-hours <训练时限> --generation-hours <生成时限>`；它复制原始 133.639 秒/筛选和 458.191 秒/继续训练的活动时间配额。需给验证和加载另留时间；不会改变租赁平台设置或启动任务。
6. 依次执行 `python main/wide/train.py`、`python main/wide/evaluate.py`。结果写入 `run/`，不会覆盖已发布 `results/`。
7. 补充实验另执行`python main/supplement/init_run.py --training-hours <训练时限> --generation-hours <生成时限> --analysis-hours <诊断时限>`，再依次执行该目录的`verify.py`、`train.py`、`evaluate.py`、`trajectory.py`和`report.py`。查看`main/supplement/PROTOCOL.md`；预算不足会停止而非改变协议。原16个筛选候选需先完成并保留相应检查点。

代码判分工作器依赖 Linux、libseccomp 和切换到非特权 UID 的权限，缺少隔离条件时拒绝执行。请在独立的可丢弃评测环境运行，不能将它当作通用安全沙箱。macOS 本地仅用于分析表重算。

## 复现边界

发布代码经过语法检查及已存结果重算；公开整理版没有重新租 GPU 运行。模型、激活、适配器检查点和未删减回答未包含在Git仓库。原阶段按墙钟活动时间停止，不同硬件/软件可能得到不同步数；补充阶段按固定12000步停止，两者均不保证相同算力或充分收敛。归一化目标、数据划分与贪心生成策略详见 `PROTOCOL.md`。

OpenAI Codex 辅助了实验代码、分析、图表与论文撰写；作者负责最终内容。本文尚为审阅稿，未声称 arXiv 接收或同行评议通过。
