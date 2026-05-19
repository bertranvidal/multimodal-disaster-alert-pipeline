from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.sa_bilstm import SABiLSTM
from src.sa_dataset import (
    PAD_TOKEN,
    SADataset,
    build_word_vocab,
    get_project_root,
    load_sa_splits,
    make_sa_collate_fn,
)


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


def get_sa_model_dir() -> Path:
    """
    Return the directory where SA artifacts will be stored.

    Returns:
        Path: Absolute path to models/sa.
    """
    return get_project_root() / "models" / "sa"


def evaluate_model(
    model: SABiLSTM,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, str]:
    """
    Evaluate the model on a dataset split.

    Args:
        model: SA model.
        data_loader: DataLoader for the split.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple[float, float, float, str]:
            loss, accuracy, macro_f1, classification_report string.
    """
    model.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, digits=4)

    return avg_loss, accuracy, macro_f1, report


def main() -> None:
    """
    Train and evaluate the SA model.
    """
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = load_sa_splits()

    print("Train samples:", len(train_texts))
    print("Dev samples:", len(dev_texts))
    print("Test samples:", len(test_texts))

    word2idx = build_word_vocab(train_texts)
    pad_idx = word2idx[PAD_TOKEN]

    train_dataset = SADataset(train_texts, train_labels, word2idx)
    dev_dataset = SADataset(dev_texts, dev_labels, word2idx)
    test_dataset = SADataset(test_texts, test_labels, word2idx)

    collate_fn = make_sa_collate_fn(word2idx)

    batch_size = 16
    embedding_dim = 100
    hidden_dim = 64
    num_classes = 3
    dropout = 0.5
    learning_rate = 1e-3
    num_epochs = 20
    patience = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SABiLSTM(
        vocab_size=len(word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pad_idx=pad_idx,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    model_dir = get_sa_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    best_dev_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
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
            torch.save(model.state_dict(), model_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print("\nBest dev macro F1:", f"{best_dev_macro_f1:.4f}")

    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))

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

    with open(model_dir / "word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    config = {
        "vocab_size": len(word2idx),
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_classes": num_classes,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "patience": patience,
        "best_dev_macro_f1": best_dev_macro_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
    }

    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()