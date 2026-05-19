from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sa_text_pipeline import DEFAULT_CAPTION_PROMPT, build_sa_inputs_from_dataframe
from sa_dataset import load_sa_csv


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_sa_bertweet_model_dir() -> Path:
    """
    Return the directory where BERTweet SA artifacts will be stored.

    Returns:
        Path: Absolute path to models/sa_bertweet.
    """
    return get_project_root() / "models" / "sa_bertweet"


class SABertweetDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Dataset for BERTweet-based damage classification.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 160,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def evaluate_model(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, str]:
    """
    Evaluate the model on one split.

    Args:
        model: Classification model.
        data_loader: Split dataloader.
        criterion: Weighted loss function.
        device: Torch device.

    Returns:
        tuple[float, float, float, str]: Loss, accuracy, macro F1, report.
    """
    model.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    return avg_loss, accuracy, macro_f1, report


def main() -> None:
    """
    Train and evaluate a BERTweet SA model with weighted loss.
    """
    set_seed(42)

    model_name = "vinai/bertweet-base"
    batch_size = 8
    learning_rate = 1.5e-5
    num_epochs = 4
    max_length = 160
    patience = 2
    use_captions = True
    caption_prompt = DEFAULT_CAPTION_PROMPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model:", model_name)

    train_df = load_sa_csv("train")
    dev_df = load_sa_csv("dev")
    test_df = load_sa_csv("test")

    train_texts = build_sa_inputs_from_dataframe(
        train_df,
        use_captions=use_captions,
        caption_prompt=caption_prompt,
    )
    dev_texts = build_sa_inputs_from_dataframe(
        dev_df,
        use_captions=use_captions,
        caption_prompt=caption_prompt,
    )
    test_texts = build_sa_inputs_from_dataframe(
        test_df,
        use_captions=use_captions,
        caption_prompt=caption_prompt,
    )

    train_labels = train_df["label"].astype(int).tolist()
    dev_labels = dev_df["label"].astype(int).tolist()
    test_labels = test_df["label"].astype(int).tolist()

    print("Train samples:", len(train_texts))
    print("Dev samples:", len(dev_texts))
    print("Test samples:", len(test_texts))
    print("Caption-enriched inputs:", use_captions)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        normalization=True,
    )

    train_dataset = SABertweetDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dev_dataset = SABertweetDataset(
        texts=dev_texts,
        labels=dev_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    test_dataset = SABertweetDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model_dir = get_sa_bertweet_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    best_dev_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        dev_loss, dev_acc, dev_macro_f1, _ = evaluate_model(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss = {avg_train_loss:.4f} | "
            f"dev_loss = {dev_loss:.4f} | "
            f"dev_acc = {dev_acc:.4f} | "
            f"dev_macro_f1 = {dev_macro_f1:.4f}"
        )

        if dev_macro_f1 > best_dev_macro_f1:
            best_dev_macro_f1 = dev_macro_f1
            epochs_without_improvement = 0
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print("\nBest dev macro F1:", f"{best_dev_macro_f1:.4f}")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    test_loss, test_acc, test_macro_f1, test_report = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nTest results:")
    print(f"test_loss = {test_loss:.4f}")
    print(f"test_acc = {test_acc:.4f}")
    print(f"test_macro_f1 = {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(test_report)

    training_config = {
        "model_name": model_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "patience": patience,
        "use_captions": use_captions,
        "caption_prompt": caption_prompt,
        "best_dev_macro_f1": best_dev_macro_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
    }

    with open(model_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
