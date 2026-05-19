from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sa_dataset import load_sa_csv
from sa_text_pipeline import DEFAULT_CAPTION_PROMPT, build_sa_inputs_from_dataframe
from train_sa_bertweet import SABertweetDataset


LABEL_NAMES = ["little_or_no_damage", "mild_damage", "severe_damage"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_sa_bertweet_model_dir() -> Path:
    return get_project_root() / "models" / "sa_bertweet"


def print_metrics(y_true: list[int], y_pred: list[int]) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )

    print(f"accuracy = {accuracy:.4f}")
    print(f"macro_f1 = {macro_f1:.4f}")
    print("confusion_matrix =")
    for row in matrix.tolist():
        print(row)
    print("\nclassification_report =")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained SA BERTweet model.")
    parser.add_argument("--split", default="test", help="Split to evaluate. Default: test.")
    args = parser.parse_args()

    if args.split != "test":
        raise ValueError("Este script está preparado para evaluar el modelo final en el split test.")

    model_dir = get_sa_bertweet_model_dir()
    if not model_dir.exists():
        raise FileNotFoundError("No se ha encontrado models/sa_bertweet.")

    test_df = load_sa_csv("test")
    test_texts = build_sa_inputs_from_dataframe(
        test_df,
        use_captions=True,
        caption_prompt=DEFAULT_CAPTION_PROMPT,
    )
    test_labels = test_df["label"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    dataset = SABertweetDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_length=160,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)

    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print_metrics(y_true, y_pred)


if __name__ == "__main__":
    main()
