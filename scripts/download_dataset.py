"""
从正规来源下载中文情感数据，生成 data/labeled_texts.csv 与 data/labeled_texts_eval.csv。

数据来源：
- 积极/消极：ChnSentiCorp（谭松波等，酒店评论），Hugging Face: lansinuote/ChnSentiCorp（Parquet 格式）
- 中性：为三分类平衡而补充的客观陈述句（见 DATA_SOURCES.md）
"""
import csv
import os
import random

# 可选：从 Hugging Face 拉取 ChnSentiCorp（需安装 pip install datasets）
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# 当 ChnSentiCorp 无法下载时使用的备用正/负样本（仅用于脚本测试）
FALLBACK_POSITIVE = [
    "酒店环境很好，服务态度也不错。",
    "房间干净整洁，下次还会入住。",
    "性价比高，推荐。",
    "设施齐全，住得很满意。",
]
FALLBACK_NEGATIVE = [
    "房间隔音太差，睡不好。",
    "卫生一般，不推荐。",
    "服务态度不好，体验差。",
]

# 中性类补充样本（客观陈述、无明确情感，用于三分类设定）
NEUTRAL_SENTENCES = [
    "会议安排在明天下午三点。",
    "这本书共有三百页。",
    "火车票已售罄。",
    "今天气温二十五度。",
    "该产品支持七天无理由退货。",
    "快递预计三天内送达。",
    "房间面积为二十平方米。",
    "营业时间为早九点到晚六点。",
    "本次航班准点起飞。",
    "订单号已通过短信发送。",
    "该功能需要联网使用。",
    "说明书在包装盒内。",
    "保修期为一年。",
    "请在工作日联系客服。",
    "系统将于今晚进行维护。",
    "本次更新修复了若干问题。",
    "该路段限速六十公里。",
    "发票可于次月开具。",
    "退房时间为中午十二点。",
    "入住需出示有效证件。",
    "停车场位于地下一层。",
    "早餐供应时间为七点到九点。",
    "WiFi 密码写在前台。",
    "电梯在走廊尽头。",
    "空调遥控器在床头柜里。",
    "浴室提供一次性用品。",
    "退订需提前二十四小时。",
    "该条款以合同为准。",
    "数据仅供参考。",
    "具体以实际为准。",
]


def fetch_chnsenticorp():
    """从 Hugging Face 加载 ChnSentiCorp，映射为 积极/消极。"""
    if not HAS_DATASETS:
        raise RuntimeError("请先安装: pip install datasets")
    ds = load_dataset("lansinuote/ChnSentiCorp")
    # 常见列名: text, label；label 0=负 1=正
    label_map = {0: "消极", 1: "积极"}
    rows = []
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for ex in ds[split]:
            text = (ex.get("text") or ex.get("sentence") or "").strip()
            if not text:
                continue
            lab = ex.get("label", 0)
            rows.append((text, label_map.get(lab, "消极")))
    return rows


def build_three_class(train_ratio=0.9, seed=42):
    """生成三分类数据：ChnSentiCorp（积极/消极） + 中性补充，并划分 train/eval。"""
    random.seed(seed)
    # 1) 积极/消极
    try:
        chn = fetch_chnsenticorp()
    except Exception as e:
        print("从 Hugging Face 加载 ChnSentiCorp 失败:", e)
        print("将使用内置示例样本（积极/消极/中性）生成 CSV；完整数据请安装 datasets 后重试。")
        chn = [(t, "积极") for t in FALLBACK_POSITIVE] + [(t, "消极") for t in FALLBACK_NEGATIVE]
    # 2) 中性
    neutral = [(s, "中性") for s in NEUTRAL_SENTENCES]
    # 合并并打乱
    all_rows = chn + neutral
    random.shuffle(all_rows)
    # 划分
    n = len(all_rows)
    n_train = max(1, int(n * train_ratio))
    train_rows = all_rows[:n_train]
    eval_rows = all_rows[n_train:]
    return train_rows, eval_rows


def main():
    # 获取项目根目录（当前文件在 scripts/，向上退一级）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_rows, eval_rows = build_three_class(train_ratio=0.9, seed=42)

    train_path = os.path.join(data_dir, "labeled_texts.csv")
    eval_path = os.path.join(data_dir, "labeled_texts_eval.csv")

    for path, rows in [(train_path, train_rows), (eval_path, eval_rows)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(rows)
        print(f"已写入 {path}，共 {len(rows)} 条。")

    print("数据来源与引用见 DATA_SOURCES.md。")
'''
def main():
    os.makedirs("data", exist_ok=True)
    train_rows, eval_rows = build_three_class(train_ratio=0.9, seed=42)

    train_path = "data/labeled_texts.csv"
    eval_path = "data/labeled_texts_eval.csv"

    for path, rows in [(train_path, train_rows), (eval_path, eval_rows)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(rows)
        print(f"已写入 {path}，共 {len(rows)} 条。")

    print("数据来源与引用见 DATA_SOURCES.md。")
'''

if __name__ == "__main__":
    main()
