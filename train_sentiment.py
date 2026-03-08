import csv
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.sentiment_model import Config, Model, clean, CLS


class SentimentDataset(Dataset):
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


def split_train_val(samples: List[Tuple[str, int]], val_ratio: float = 0.1, seed: int = 42):
    random.Random(seed).shuffle(samples)
    n_total = len(samples)
    n_val = max(1, int(n_total * val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    return train_samples, val_samples


def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = tuple(t.to(device) for t in batch_x)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item() * batch_y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train():
    config = Config()
    device = config.device


    data_path = "data/labeled_texts.csv"
    print(f"Loading labeled data from {data_path} ...")
    samples = load_labeled_csv(data_path, config)
    print(f"Total labeled samples: {len(samples)}")

    train_samples, val_samples = split_train_val(samples, val_ratio=0.1, seed=42)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_dataset = SentimentDataset(train_samples, config)
    val_dataset = SentimentDataset(val_samples, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 2
    bad_epochs = 0

    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        for batch_x, batch_y in train_loader:
            batch_x = tuple(t.to(device) for t in batch_x)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = epoch_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), config.save_path)
            print(f"New best model saved to {config.save_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()

