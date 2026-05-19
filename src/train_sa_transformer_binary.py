from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sa_text_pipeline import DEFAULT_CAPTION_PROMPT, build_sa_inputs_from_dataframe
from train_sa_transformer import SATransformerDataset, evaluate_model, load_sa_csv


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_binary_model_dir() -> Path:
    return get_project_root() / "models" / "sa_transformer_binary"


def to_binary_labels(labels: pd.Series) -> list[int]:
    return (labels.astype(int) > 0).astype(int).tolist()


def main() -> None:
    set_seed(42)

    model_name = "distilbert-base-uncased"
    batch_size = 8
    learning_rate = 2e-5
    num_epochs = 4
    max_length = 128
    patience = 2
    use_captions = True
    caption_prompt = DEFAULT_CAPTION_PROMPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model:", model_name)
    print("Task: binary damage classification")
    print("Labels: 0 = no_damage, 1 = damage")

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

    train_labels = to_binary_labels(train_df["label"])
    dev_labels = to_binary_labels(dev_df["label"])
    test_labels = to_binary_labels(test_df["label"])

    print("Train samples:", len(train_texts))
    print("Dev samples:", len(dev_texts))
    print("Test samples:", len(test_texts))
    print("Train label counts:", {label: train_labels.count(label) for label in sorted(set(train_labels))})
    print("Dev label counts:", {label: dev_labels.count(label) for label in sorted(set(dev_labels))})
    print("Test label counts:", {label: test_labels.count(label) for label in sorted(set(test_labels))})
    print("Caption-enriched inputs:", use_captions)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    train_dataset = SATransformerDataset(train_texts, train_labels, tokenizer, max_length)
    dev_dataset = SATransformerDataset(dev_texts, dev_labels, tokenizer, max_length)
    test_dataset = SATransformerDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        local_files_only=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model_dir = get_binary_model_dir()
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
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        dev_loss, dev_acc, dev_macro_f1, _ = evaluate_model(model, dev_loader, device)

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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)

    test_loss, test_acc, test_macro_f1, _ = evaluate_model(model, test_loader, device)

    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print("\nTest results:")
    print(f"test_loss = {test_loss:.4f}")
    print(f"test_acc = {test_acc:.4f}")
    print(f"test_macro_f1 = {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["no_damage", "damage"],
            digits=4,
            zero_division=0,
        )
    )

    training_config = {
        "model_name": model_name,
        "task": "binary_damage_classification",
        "label_map": {"0": "no_damage", "1": "damage"},
        "original_label_mapping": {
            "0": "no_damage",
            "1": "damage",
            "2": "damage",
        },
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
