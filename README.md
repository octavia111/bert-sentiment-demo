## 项目简介 / Overview

这是一个基于 BERT 的**中文情绪/情感分析工具**，可以对中文文本进行三分类：

- **消极（Negative）**
- **中性（Neutral）**
- **积极（Positive）**

项目特点：

- **科研友好**：支持从 CSV 批量读取文本，输出情绪标签和各类别概率，可直接用于论文中的数据统计与可视化。
- **工程规范**：核心模型封装在 `src/sentiment_model.py` 中，提供脚本化接口和示例脚本。
- **可扩展**：可以很容易地替换为你自己的数据或继续微调模型。

## 项目结构（推荐科研化视角）

当前仓库中的主要文件/目录与推荐结构的映射如下：

```text
bert-sentiment-demo/
├── data/                     # 数据
│   ├── test_samples.txt     # 示例测试文本
│   ├── labeled_texts.csv    # 训练集（由 download_dataset.py 生成）
│   ├── labeled_texts_eval.csv  # 评估集
│   └── input_texts.csv      # 待分析文本（批量分析时自建）
├── src/
│   └── sentiment_model.py   # BERT 模型与推理逻辑
├── scripts/
│   └── download_dataset.py  # 从 Hugging Face 下载 ChnSentiCorp 并生成 CSV
├── bert_pretrain/           # 预训练 BERT 配置与词表
├── pytorch_pretrained/      # 本地 BERT 实现依赖
├── saved_dict/
│   └── bert.ckpt            # 训练得到的模型权重
├── results/                 # 批量分析输出（运行后生成）
├── run_predict.py           # 快速 demo：少量句子情感分类
├── train_sentiment.py       # 训练脚本
├── eval_sentiment.py        # 评估脚本（准确率 + 混淆矩阵）
├── analyze_csv.py           # 批量 CSV 分析（科研/论文用）
├── requirements.txt
├── DATA_SOURCES.md          # 数据来源与引用说明
└── README.md
```

> 说明：若某些目录（如 `saved_dict/`、`results/`）在初次下载后不存在，会在你训练或运行脚本时自动创建或需要你手动创建。

在 GitHub 或科研简历中，你可以将该项目描述为：

> “基于 BERT 的中文心理学文本情感分析工具，支持批量文本标签预测、概率输出和结果统计，可用于心理学/社会科学文本数据的定量分析。”

## 环境与安装

1. 建议使用 Conda 创建虚拟环境（Python ≥ 3.8）：

```bash
conda create -n nlp python=3.10
conda activate nlp
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 确保以下资源就位：

- `bert_pretrain/` 目录下包含预训练 BERT 的必要文件（`bert_config.json`、`vocab.txt`、权重等）。
- `saved_dict/bert.ckpt` 为已训练好的情感分类模型权重。

> 如果你有自己的 BERT 权重或想重新训练，可以将新模型保存为 `saved_dict/bert.ckpt`，推理脚本会自动加载。

## 训练与评估数据（正规来源）

情感标注数据使用**可引用的公开数据集**生成：

- **积极/消极**：[ChnSentiCorp](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)（谭松波等，中文酒店评论情感语料，Parquet 格式）
- **中性**：脚本内补充的客观陈述句，用于三分类平衡（详见 `DATA_SOURCES.md`）

生成训练/评估 CSV（需先安装 `datasets`：`pip install datasets`）：

```bash
python scripts/download_dataset.py
```

将得到 `data/labeled_texts.csv`（训练）与 `data/labeled_texts_eval.csv`（评估）。**数据来源与引用格式见 [DATA_SOURCES.md](DATA_SOURCES.md)。**

## 完整流程：训练 → 评估 → 推理

按顺序执行即可复现实验（需已安装依赖并准备好 `bert_pretrain/`）：

```bash
# 1. 下载数据并生成 CSV
python scripts/download_dataset.py

# 2. 训练（保存最优模型到 saved_dict/bert.ckpt）
python train_sentiment.py

# 3. 在评估集上计算准确率与混淆矩阵
python eval_sentiment.py
```

## 实验结果（ChnSentiCorp + 中性补充）

在本次设定下（训练 9,745 / 验证 1,082 / 测试 1,203 条，3 个 epoch）：

| 项目 | 数值 |
|------|------|
| 最佳验证准确率 | **91.40%** |
| 测试集准确率 | **89.11%** |
| 最优 checkpoint | `saved_dict/bert.ckpt`（Epoch 3） |

**训练日志示例：**

```
Epoch 1: train_loss=0.4406, train_acc=0.8041, val_loss=0.3213, val_acc=0.8762
Epoch 2: train_loss=0.2612, train_acc=0.9003, val_loss=0.2941, val_acc=0.8900
Epoch 3: train_loss=0.1546, train_acc=0.9447, val_loss=0.2276, val_acc=0.9140
Best validation accuracy: 0.9140
```

**混淆矩阵（测试集）**：主要错误集中在积极↔消极之间的混淆，与数据中中性样本极少、偏二分类一致。

## 快速体验：对少量句子做情感分类

脚本 `run_predict.py` 会调用 `src.sentiment_model.predict` 对几条示例句子进行预测：

```bash
python run_predict.py
```

示例输出：

```text
text:这部电影真的很好看，我非常喜欢  label:积极
text:剧情拖沓，浪费时间          label:消极
text:演员演技很好，值得推荐      label:积极
```

这是一个最小可运行 demo，适合作为功能展示。

## 科研/论文使用：批量 CSV 分析 + 概率输出

对于科研/论文场景，你通常会有一批已经爬取或收集好的文本数据，希望：

- 自动打上情绪/情感标签
- 获得每个情绪类别的概率
- 做整体分布统计与可视化

本项目为此提供了脚本 `analyze_csv.py`。

### 1. 准备你的文本数据（CSV）

在 `data/` 目录下创建一个 `input_texts.csv`，至少包含一列 `text`，例如：

```csv
text
这部电影真的很好看
剧情拖沓，浪费时间
演员演技很好，值得推荐
```

> 你可以将爬取/标注好的任意中文文本放在这一列中，用于大规模情绪分析。

### 2. 运行批量分析脚本

在项目根目录下执行：

```bash
python analyze_csv.py
```

脚本会完成如下工作：

- 从 `data/input_texts.csv` 读取 `text` 列；
- 使用 BERT 模型进行情感预测；
- 对每条文本输出：
  - 原文：`text`
  - 预测标签：`label`（中性/积极/消极）
  - 各类别概率：`prob_neutral`、`prob_positive`、`prob_negative`
- 将结果保存到 `results/sentiment_results.csv`；
- 在终端打印各情感标签的数量和占比，方便写入论文或做图。

输出 CSV 示例：

```csv
text,label,prob_neutral,prob_positive,prob_negative
这部电影真的很好看,积极,0.05,0.92,0.03
剧情拖沓，浪费时间,消极,0.03,0.08,0.89
演员演技很好，值得推荐,积极,0.04,0.90,0.06
```

你可以使用 Excel、pandas、seaborn 或其它可视化工具，对该结果做：

- 情绪分布柱状图/饼图
- 不同时间/主题下的情绪趋势曲线
- 与其它心理学量表或变量的相关分析

## 模型与方法说明（可写进论文/申请材料）

- **模型架构**：`BERT-base` + 单层线性分类器。
- **输入处理**：
  - 使用 `BertTokenizer` 对中文文本进行分词；
  - 文本统一截断/补长到固定长度 `pad_size`；
  - 使用 attention mask 标记 padding 部分。
- **输出**：
  - 通过 `softmax` 得到每个类别的概率分布；
  - 使用 `argmax` 得到最终情感标签。

你可以在论文/申请中这样描述该部分工作：

> “我们基于中文 BERT 预训练模型构建了三分类情感分析器（积极/中性/消极），并在自建文本语料上进行微调。模型对每条文本输出各类别的 posterior probability，进而支持情绪分布统计及与其他心理学变量的相关分析。”

## 心理学/社会科学应用场景示例

- **微博/社交媒体文本情绪分布分析**：分析某一事件前后情绪变化趋势。
- **问卷开放题的情绪倾向**：将开放式回答转化为可量化的情绪标签和概率。
- **论文语料辅助标注**：为后续人工精标提供候选情绪标签和置信度，降低标注成本。

在 GitHub 或个人陈述中，你可以强调：

- 你熟悉 **NLP + 深度学习（BERT）** 技术路线；
- 你能搭建 **完整 pipeline**：数据预处理 → 模型推理 → 结果统计与可视化；
- 你能将技术工具与 **心理学/社会科学研究问题** 紧密结合。

## 上传到 GitHub

1. 在项目根目录初始化并提交（若尚未初始化）：
   ```bash
   git init
   git add .
   git commit -m "BERT 中文情感分析：训练、评估、批量分析完整流程"
   ```
2. 在 GitHub 新建仓库（如 `bert-sentiment-demo`），按页面提示关联并推送：
   ```bash
   git remote add origin https://github.com/<你的用户名>/bert-sentiment-demo.git
   git branch -M main
   git push -u origin main
   ```
3. 若 `saved_dict/bert.ckpt` 体积较大，可选择性加入 `.gitignore` 后由读者自行训练；当前 `.gitignore` 已排除缓存与 IDE 配置。

## 后续可扩展方向

- 在 `notebooks/` 中用 Jupyter Notebook 做可视化（训练曲线、混淆矩阵图）。
- 对比不同模型（LSTM、RoBERTa 等）在同一数据集上的表现。
- 增加 F1/宏 F1 等指标，或接入 W&B / MLflow 做实验跟踪。

这些扩展可进一步提升项目作为**科研作品集**和**硕士申请材料**的说服力。
#   b e r t - s e n t i m e n t - d e m o  
 