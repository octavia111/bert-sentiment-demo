import csv
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from src.sentiment_model import Config, Model, clean, CLS


class EvalDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], config: Config):
        self.config = config
        self.samples = samples
        self.pad_size = config.pad_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label_idx = self.samples[idx]
        lin = clean(text)
        token = self.config.tokenizer.tokenize(lin)
        token = [CLS] + token
        seq_len = len(token)
        token_ids = self.config.tokenizer.convert_tokens_to_ids(token)

        if self.pad_size:
            if len(token_ids) < self.pad_size:
                mask = [1] * len(token_ids) + [0] * (self.pad_size - len(token_ids))
                token_ids = token_ids + [0] * (self.pad_size - len(token_ids))
            else:
                mask = [1] * self.pad_size
                token_ids = token_ids[: self.pad_size]
                seq_len = self.pad_size
        else:
            mask = [1] * len(token_ids)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(label_idx, dtype=torch.long),
        )


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    seq_lens = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    labels = torch.stack([b[3] for b in batch])
    return (input_ids, seq_lens, masks), labels


def load_labeled_csv(path: str, config: Config) -> List[Tuple[str, int]]:
    label2idx = {label: i for i, label in enumerate(config.class_list)}
    samples: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV 需要包含列: text, label")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or label not in label2idx:
                continue
            samples.append((text, label2idx[label]))
    return samples


def main():
    config = Config()
    device = config.device

    data_path = "data/labeled_texts_eval.csv"  # 或者复用训练集，按需修改
    print(f"Loading labeled eval data from {data_path} ...")
    samples = load_labeled_csv(data_path, config)
    print(f"Eval samples: {len(samples)}")

    dataset = EvalDataset(samples, config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = Model(config).to(device)
    model.load_state_dict(torch.load(config.save_path, map_location=device))
    model.eval()

    correct, total = 0, 0
    confusion = Counter()

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = tuple(t.to(device) for t in batch_x)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            for t_label, p_label in zip(batch_y.cpu().tolist(), preds.cpu().tolist()):
                confusion[(t_label, p_label)] += 1

    acc = correct / max(1, total)
    print(f"Eval accuracy: {acc:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    labels = list(range(len(config.class_list)))
    header = "     " + " ".join(f"{config.class_list[j]:^6}" for j in labels)
    print(header)
    for i in labels:
        row = [config.class_list[i].ljust(4)]
        for j in labels:
            row.append(str(confusion[(i, j)]).rjust(6))
        print(" ".join(row))


if __name__ == "__main__":
    main()

