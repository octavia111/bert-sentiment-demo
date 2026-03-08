import csv
import os
from collections import Counter

from src.sentiment_model import predict_with_proba


def read_texts_from_csv(input_path: str, text_column: str = "text"):
    texts = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_column not in reader.fieldnames:
            raise ValueError(f"CSV 中未找到列: {text_column}")
        for row in reader:
            value = (row.get(text_column) or "").strip()
            if value:
                texts.append(value)
    return texts


def save_results_to_csv(output_path: str, results):
    if not results:
        print("没有任何文本可供分析。")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 概率顺序与 class_list 对应：['中性', '积极', '消极']
    fieldnames = ["text", "label", "prob_neutral", "prob_positive", "prob_negative"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            probs = r["probabilities"]
            writer.writerow(
                {
                    "text": r["text"],
                    "label": r["label"],
                    "prob_neutral": probs[0],
                    "prob_positive": probs[1],
                    "prob_negative": probs[2],
                }
            )


def print_label_statistics(results):
    labels = [r["label"] for r in results]
    counter = Counter(labels)
    total = len(labels)
    print("\n情感标签统计：")
    for label, count in counter.items():
        print(f"{label}: {count} ({count / total:.2%})")


def main():
    input_path = "data/input_texts.csv"  # 你可以根据需要修改
    output_path = "results/sentiment_results.csv"

    print(f"从 {input_path} 读取文本...")
    texts = read_texts_from_csv(input_path, text_column="text")
    print(f"共读取 {len(texts)} 条文本。")

    print("调用 BERT 模型进行情感分析（包含概率分数）...")
    results = predict_with_proba(texts)

    print(f"将结果保存到 {output_path} ...")
    save_results_to_csv(output_path, results)

    print_label_statistics(results)
    print("\n分析完成。你可以在 Excel / pandas 中继续做可视化和深入分析。")


if __name__ == "__main__":
    main()

