# Frozen Exit Adapters

冻结语言模型的末端宽 FFN 出口：**表示拟合 ≠ 连续生成能力**。

本仓库对应中文审阅稿《从表示拟合到连续生成：冻结语言模型末端宽前馈适配器的实证研究》的第二轮宽 FFN 实验。主干、最终 RMSNorm 和 lm_head 全部冻结，仅训练残差适配器。保留深度 60–63，每层 16 种结构，共 64 次筛选 + 8 次验证集选择后的继续训练。

## 主要观察

| 配置 | 数学 / 100 | 代码 / 99 |
|---|---:|---:|
| 完整教师 | 94 | 71 |
| Best-60 | 17 | 37 |
| Best-61 | 62 | 38 |
| Best-62 | 95 | 43 |
| Best-63 | 92 | 64 |

Best 仅指向量验证 MSE 最低的候选，不按生成测试分数挑选。8 组继续训练均无 patience 早停，但后期训练误差下降、验证误差上升。测试集此前已使用，结果是探索性的；没有多种子重复、跨模型实验或实测部署加速结论。

## 无需 GPU：重算统计和 10 张图

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python paper/analysis/make_figures.py
python paper/analysis/redraw_figures.py
python paper/analysis/check_results.py
```

第一步作图脚本重算 CSV、配对 bootstrap 和摘要；第二步应用最终排版；第三步核验数值和输入哈希。输出位于 `paper/figures/`、`paper/analysis/derived.json`。`results` 中的生成记录是公开分析表，保留逐题 ID、判分和诊断，去除了题目文本、参考程序与回答文本。可重算本文成绩和图表，不能用这些表重新执行回答判分。

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

原实验为一张 A800 80GB、BF16 主干、隐藏维度 5120、64 个混合层块。需要兼容 `Qwen3_5ForConditionalGeneration` 的 Transformers，以及本地 Qwen3.8-27B 权重。**未保存不可变模型 revision 和完整依赖锁文件，不能承诺字节级复现。** `requirements-experiment.txt` 是运行依赖列表，不是原环境锁定文件。

1. 安装实验依赖，设置 `MODEL_PATH` 为已下载模型目录。代码不自动下载模型。
2. 执行 `python main/download_data.py`，仅接受与归档 SHA-256 一致的 GSM8K、MBPP、Code Alpaca 文件；随后执行 `python main/prepare_data.py`。各数据集条款和来源见 `DATA.md`。
3. 执行 `python main/collect.py`。此脚本保持原采集方式，保存全部 65 个位置边界；需要预留约 107 GB 激活文件空间，以及模型和检查点空间。不会自动租机器。
4. 执行 `python main/evaluate_baselines.py`，生成新的 `main/run/supported_evaluation.json`。原环境排除了参考执行失败的 MBPP 123，留下 199 道测试题；若新环境题集不同，停止并记录差异，不能静默替换本文分母。
5. 在空的运行目录上执行 `python main/wide/init_run.py --training-hours <训练时限> --generation-hours <生成时限>`；它复制原始 133.639 秒/筛选和 458.191 秒/继续训练的活动时间配额。需给验证和加载另留时间；不会改变租赁平台设置或启动任务。
6. 依次执行 `python main/wide/train.py`、`python main/wide/evaluate.py`。结果写入 `run/`，不会覆盖已发布 `results/`。

代码判分工作器依赖 Linux、libseccomp 和切换到非特权 UID 的权限，缺少隔离条件时拒绝执行。请在独立的可丢弃评测环境运行，不能将它当作通用安全沙箱。macOS 本地仅用于分析表重算。

## 复现边界

发布代码经过语法检查及已存结果重算；公开整理版没有重新租 GPU 运行。模型、激活、80 个本地检查点和未删减回答未包含在 Git 仓库。训练按墙钟活动时间而非固定步数停止，不同硬件/软件可能得到不同步数。归一化目标、数据划分与贪心生成策略详见 `PROTOCOL.md`。

OpenAI Codex 辅助了实验代码、分析、图表与论文撰写；作者负责最终内容。本文尚为审阅稿，未声称 arXiv 接收或同行评议通过。
