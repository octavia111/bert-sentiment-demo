# 数据来源与引用 (Data Sources & Citation)

本仓库的情感分类数据来源如下，便于论文/申请材料中规范引用。

## 积极 / 消极 类别

- **数据集名称**：ChnSentiCorp（中文情感分析语料，酒店评论子集）
- **来源**：谭松波等整理，广泛用于中文情感分析研究。
- **获取方式**：Hugging Face [lansinuote/ChnSentiCorp](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)（Parquet 格式，无需 loading script）
- **规模**：约 7000+ 条酒店评论，二分类（正/负）。
- **许可**：学术与教育使用常见；具体以 Hugging Face 页面及原始发布方为准。
- **在本项目中的使用**：通过 `scripts/download_dataset.py` 下载后，将标签映射为「积极」「消极」，并与中性类合并为三分类训练/评估 CSV。

**建议引用 (BibTeX)**：

```bibtex
@misc{chnsenticorp,
  title        = {ChnSentiCorp},
  author       = {Tan, Songbo and others},
  year         = {2008},
  howpublished = {Chinese sentiment analysis corpus, hotel reviews},
  note         = {Available via Hugging Face: lansinuote/ChnSentiCorp}
}
```

## 中性 类别

- **说明**：ChnSentiCorp 仅含积极/消极，为支持「积极 / 中性 / 消极」三分类，本项目在 `scripts/download_dataset.py` 中补充了少量**客观陈述句**（如“会议安排在明天下午三点”“保修期为一年”等），仅用于平衡类别与实验设定。
- **规模**：约 30 条，与 ChnSentiCorp 合并后做 train/val 划分。
- **引用建议**：在论文/报告中可写为“中性类为自建客观陈述句，用于三分类设定”。

## 生成的文件

- `data/labeled_texts.csv`：训练用 CSV（列：`text`, `label`）。
- `data/labeled_texts_eval.csv`：评估用 CSV，格式同上。

运行一次即可生成：

```bash
pip install datasets   # 若未安装
python scripts/download_dataset.py
```

## 可复现性

- 脚本内使用固定随机种子（默认 42）划分 train/eval，便于复现。
- 若无法访问 Hugging Face，脚本会退化为使用内置的少量示例样本，仅用于验证流程；正式实验建议在可访问 HF 的环境下重新运行以使用完整 ChnSentiCorp。
