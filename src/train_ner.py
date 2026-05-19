from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from ner_char_bilstm import CharCNNBiLSTMNER
from ner_dataset import (
    PAD_CHAR,
    PAD_TAG,
    PAD_TOKEN,
    UNK_TOKEN,
    NERDataset,
    build_char_vocab,
    build_sentences,
    build_tag_vocab,
    build_word_vocab,
    load_ner_csv,
    load_ner_splits,
    make_collate_fn,
)


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_model_dir() -> Path:
    """
    Return the directory where trained NER files will be stored.

    The directory is created if it does not already exist.

    Returns:
        Path: Absolute path to models/ner.
    """
    suffix = os.getenv("NER_MODEL_DIR_SUFFIX", "").strip()
    directory_name = "ner" if not suffix else f"ner_{suffix}"
    path = get_project_root() / "models" / directory_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate(
    model: CharCNNBiLSTMNER,
    loader: DataLoader,
    device: torch.device,
    pad_tag_idx: int,
) -> tuple[float, float, list[int], list[int]]:
    """
    Evaluate the model on a dataset loader.

    Args:
        model: Trained NER model.
        loader: DataLoader containing the evaluation split.
        device: Torch device used for computation.
        pad_tag_idx: Tag index corresponding to padding.

    Returns:
        tuple[float, float, list[int], list[int]]:
            - macro F1 score
            - micro F1 score
            - flattened true labels without padding
            - flattened predicted labels without padding
    """
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            label_ids = batch["label_ids"].to(device)

            logits = model(input_ids, char_ids)
            preds = torch.argmax(logits, dim=-1)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = int(batch["lengths"][i].item())
                true_seq = label_ids[i][:length].cpu().tolist()
                pred_seq = preds[i][:length].cpu().tolist()

                all_true.extend(true_seq)
                all_pred.extend(pred_seq)

    filtered_true: list[int] = []
    filtered_pred: list[int] = []

    for true_label, pred_label in zip(all_true, all_pred):
        if true_label != pad_tag_idx:
            filtered_true.append(true_label)
            filtered_pred.append(pred_label)

    macro_f1 = float(f1_score(filtered_true, filtered_pred, average="macro"))
    micro_f1 = float(f1_score(filtered_true, filtered_pred, average="micro"))

    return macro_f1, micro_f1, filtered_true, filtered_pred


def bio_entity_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """
    Convert a BIO tag sequence into strict entity spans.

    Returns:
        set[tuple[int, int, str]]: Spans as inclusive start, exclusive end, and label.
    """
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    current_label: str | None = None

    for index, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            if start is not None and current_label is not None:
                spans.add((start, index, current_label))
            start = None
            current_label = None
            continue

        prefix, label = tag.split("-", maxsplit=1)
        if prefix == "B" or current_label != label:
            if start is not None and current_label is not None:
                spans.add((start, index, current_label))
            start = index
            current_label = label
        elif prefix != "I":
            if start is not None and current_label is not None:
                spans.add((start, index, current_label))
            start = None
            current_label = None

    if start is not None and current_label is not None:
        spans.add((start, len(tags), current_label))

    return spans


def evaluate_entity_level(
    model: CharCNNBiLSTMNER,
    loader: DataLoader,
    device: torch.device,
    idx2tag: dict[int, str],
) -> tuple[float, float, float, int, int, int]:
    """
    Evaluate strict entity-level precision, recall, and F1.

    A predicted entity is correct only when its span boundaries and label both match.
    """
    model.eval()
    true_count = 0
    pred_count = 0
    correct_count = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            label_ids = batch["label_ids"].to(device)

            logits = model(input_ids, char_ids)
            preds = torch.argmax(logits, dim=-1)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = int(batch["lengths"][i].item())
                true_tags = [idx2tag[idx] for idx in label_ids[i][:length].cpu().tolist()]
                pred_tags = [idx2tag[idx] for idx in preds[i][:length].cpu().tolist()]

                true_spans = bio_entity_spans(true_tags)
                pred_spans = bio_entity_spans(pred_tags)

                true_count += len(true_spans)
                pred_count += len(pred_spans)
                correct_count += len(true_spans & pred_spans)

    precision = correct_count / pred_count if pred_count else 0.0
    recall = correct_count / true_count if true_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1, correct_count, pred_count, true_count


def build_loader(
    sentences: list[list[str]],
    tags: list[list[str]],
    word2idx: dict[str, int],
    char2idx: dict[str, int],
    tag2idx: dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Build a NER DataLoader using the project collate function.
    """
    dataset = NERDataset(sentences, tags, word2idx, tag2idx, char2idx=char2idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(word2idx, tag2idx, char2idx=char2idx),
    )


def load_optional_split(split: str) -> tuple[list[list[str]], list[list[str]]] | None:
    """
    Load an extra NER split if its CSV file exists.
    """
    path = get_project_root() / "data" / "NER_data" / f"{split}.csv"
    if not path.exists():
        return None

    return build_sentences(load_ner_csv(split))


def load_training_data() -> tuple[list[list[str]], list[list[str]]]:
    """
    Load original train data plus optional synthetic augmentation.
    """
    train_sentences, train_tags, _, _, _, _ = load_ner_splits()
    extra_splits: list[str] = ["train_augmented"]
    if os.getenv("NER_USE_SA_TWITTER_WEAK", "1") == "1":
        extra_splits.append("sa_twitter_weak_train")

    all_sentences = list(train_sentences)
    all_tags = list(train_tags)
    for split in extra_splits:
        extra_split = load_optional_split(split)
        if extra_split is None:
            continue
        extra_sentences, extra_tags = extra_split
        all_sentences.extend(extra_sentences)
        all_tags.extend(extra_tags)

    return all_sentences, all_tags


def build_loss_weights(train_tags: list[list[str]], tag2idx: dict[str, int], device: torch.device) -> torch.Tensor:
    """
    Build smoothed inverse-frequency class weights for token-level cross entropy.
    """
    counts = torch.ones(len(tag2idx), dtype=torch.float)
    for tags in train_tags:
        for tag in tags:
            counts[tag2idx[tag]] += 1.0

    total = counts.sum()
    weights = torch.sqrt(total / (len(tag2idx) * counts))
    weights[tag2idx["O"]] *= 0.5
    weights[tag2idx[PAD_TAG]] = 0.0
    weights = weights / weights[weights > 0].mean()
    return weights.to(device)


def print_split_report(
    title: str,
    prefix: str,
    y_true: list[int],
    y_pred: list[int],
    label_ids_eval: list[int],
    target_names: list[str],
    entity_metrics: tuple[float, float, float, int, int, int],
) -> None:
    """
    Print token-level and entity-level metrics for one split.
    """
    token_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    token_micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    entity_precision, entity_recall, entity_f1, correct, predicted, gold = entity_metrics

    print(f"\n{title}")
    print(f"{prefix}_token_macro_f1={token_macro_f1:.4f}")
    print(f"{prefix}_token_micro_f1={token_micro_f1:.4f}")
    print(f"{prefix}_entity_precision={entity_precision:.4f}")
    print(f"{prefix}_entity_recall={entity_recall:.4f}")
    print(f"{prefix}_entity_f1={entity_f1:.4f}")
    print(f"{prefix}_entity_counts=correct:{correct} predicted:{predicted} gold:{gold}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_ids_eval,
            target_names=target_names,
            zero_division=0,
        )
    )


def train_one_epoch(
    model: CharCNNBiLSTMNER,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: NER model to train.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Torch device used for computation.

    Returns:
        float: Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        char_ids = batch["char_ids"].to(device)
        label_ids = batch["label_ids"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, char_ids)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            label_ids.view(-1),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / len(loader)


def save_json(obj: dict[str, int] | dict[int, str] | dict[str, float | int], path: Path) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        obj: Dictionary to save.
        path: Destination JSON path.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def main() -> None:
    """
    Train, evaluate, and save the NER baseline model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        original_train_sentences,
        original_train_tags,
        dev_sentences,
        dev_tags,
        test_sentences,
        test_tags,
    ) = load_ner_splits()

    train_sentences, train_tags = load_training_data()
    word2idx = build_word_vocab(train_sentences)
    char2idx = build_char_vocab(train_sentences)
    tag2idx, idx2tag = build_tag_vocab(train_tags)

    batch_size = 16

    train_loader = build_loader(
        train_sentences,
        train_tags,
        word2idx,
        char2idx,
        tag2idx,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = build_loader(
        dev_sentences,
        dev_tags,
        word2idx,
        char2idx,
        tag2idx,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = build_loader(
        test_sentences,
        test_tags,
        word2idx,
        char2idx,
        tag2idx,
        batch_size=batch_size,
        shuffle=False,
    )

    hard_test_split = load_optional_split("hard_test")
    hard_test_loader = None
    if hard_test_split is not None:
        hard_test_sentences, hard_test_tags = hard_test_split
        hard_test_loader = build_loader(
            hard_test_sentences,
            hard_test_tags,
            word2idx,
            char2idx,
            tag2idx,
            batch_size=batch_size,
            shuffle=False,
        )

    model = CharCNNBiLSTMNER(
        vocab_size=len(word2idx),
        char_vocab_size=len(char2idx),
        embedding_dim=128,
        char_embedding_dim=32,
        char_num_filters=32,
        hidden_dim=160,
        num_tags=len(tag2idx),
        pad_idx=word2idx[PAD_TOKEN],
        char_pad_idx=char2idx[PAD_CHAR],
        unk_idx=word2idx[UNK_TOKEN],
        dropout=0.35,
        word_dropout=0.08,
    ).to(device)

    class_weights = build_loss_weights(train_tags, tag2idx, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=tag2idx[PAD_TAG])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)

    num_epochs = 18
    best_dev_macro_f1 = -1.0
    best_dev_entity_f1 = -1.0
    patience = 4
    epochs_without_improvement = 0
    model_dir = get_model_dir()
    best_model_path = model_dir / "best_model.pt"

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_macro_f1, dev_micro_f1, _, _ = evaluate(model, dev_loader, device, tag2idx[PAD_TAG])
        dev_entity_f1 = evaluate_entity_level(model, dev_loader, device, idx2tag)[2]

        print(
            f"Epoch {epoch + 1} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_macro_f1={dev_macro_f1:.4f} | "
            f"dev_micro_f1={dev_micro_f1:.4f} | "
            f"dev_entity_f1={dev_entity_f1:.4f}"
        )

        if dev_entity_f1 > best_dev_entity_f1:
            best_dev_macro_f1 = dev_macro_f1
            best_dev_entity_f1 = dev_entity_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))

    dev_macro_f1, dev_micro_f1, y_true_dev, y_pred_dev = evaluate(
        model,
        dev_loader,
        device,
        tag2idx[PAD_TAG],
    )
    # Current synthetic in-domain test evaluation. Keep this as the easy baseline.
    test_macro_f1, test_micro_f1, y_true_test, y_pred_test = evaluate(
        model,
        test_loader,
        device,
        tag2idx[PAD_TAG],
    )
    hard_test_macro_f1 = None
    hard_test_micro_f1 = None
    hard_test_entity_f1 = None
    y_true_hard_test: list[int] = []
    y_pred_hard_test: list[int] = []
    if hard_test_loader is not None:
        hard_test_macro_f1, hard_test_micro_f1, y_true_hard_test, y_pred_hard_test = evaluate(
            model,
            hard_test_loader,
            device,
            tag2idx[PAD_TAG],
        )

    label_ids_eval = [idx for tag, idx in tag2idx.items() if tag != PAD_TAG]
    target_names = [idx2tag[idx] for idx in label_ids_eval]

    dev_entity_metrics = evaluate_entity_level(model, dev_loader, device, idx2tag)
    test_entity_metrics = evaluate_entity_level(model, test_loader, device, idx2tag)

    print_split_report(
        "DEV RESULTS",
        "dev",
        y_true_dev,
        y_pred_dev,
        label_ids_eval,
        target_names,
        dev_entity_metrics,
    )

    print_split_report(
        "TEST RESULTS",
        "test",
        y_true_test,
        y_pred_test,
        label_ids_eval,
        target_names,
        test_entity_metrics,
    )

    config: dict[str, float | int] = {
        "embedding_dim": 128,
        "char_embedding_dim": 32,
        "char_num_filters": 32,
        "hidden_dim": 160,
        "dropout": 0.35,
        "word_dropout": 0.08,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "original_train_sentences": len(original_train_sentences),
        "augmented_train_sentences": len(train_sentences) - len(original_train_sentences),
        "train_sentences": len(train_sentences),
        "best_dev_macro_f1": best_dev_macro_f1,
        "best_dev_entity_f1": best_dev_entity_f1,
        "test_macro_f1": test_macro_f1,
        "test_micro_f1": test_micro_f1,
        "test_entity_f1": test_entity_metrics[2],
    }

    if hard_test_loader is not None and hard_test_macro_f1 is not None and hard_test_micro_f1 is not None:
        hard_test_entity_metrics = evaluate_entity_level(model, hard_test_loader, device, idx2tag)
        hard_test_entity_f1 = hard_test_entity_metrics[2]
        print_split_report(
            "HARD TEST RESULTS",
            "hard_test",
            y_true_hard_test,
            y_pred_hard_test,
            label_ids_eval,
            target_names,
            hard_test_entity_metrics,
        )
        config["hard_test_macro_f1"] = hard_test_macro_f1
        config["hard_test_micro_f1"] = hard_test_micro_f1
        config["hard_test_entity_f1"] = hard_test_entity_f1
    else:
        print("\nHARD TEST RESULTS")
        print("hard_test.csv not found. Run: python3 src/build_hard_ner_test.py")

    save_json(word2idx, model_dir / "word2idx.json")
    save_json(char2idx, model_dir / "char2idx.json")
    save_json(tag2idx, model_dir / "tag2idx.json")
    save_json(idx2tag, model_dir / "idx2tag.json")
    save_json(config, model_dir / "config.json")

    print(f"\nSaved model files to: {model_dir}")


if __name__ == "__main__":
    main()
