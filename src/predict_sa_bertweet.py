from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sa_text_pipeline import combine_text_and_caption


LABEL_MAP = {
    0: "little_or_no_damage",
    1: "mild_damage",
    2: "severe_damage",
}


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_sa_bertweet_model_dir() -> Path:
    """
    Return the directory where BERTweet SA artifacts are stored.

    Returns:
        Path: Absolute path to models/sa_bertweet.
    """
    return get_project_root() / "models" / "sa_bertweet"


def predict_text(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 160,
) -> tuple[int, str, list[float]]:
    """
    Predict the damage class for an input text.

    Args:
        text: Input text.
        model: Loaded classifier.
        tokenizer: Loaded tokenizer.
        device: Torch device.
        max_length: Maximum sequence length.

    Returns:
        tuple[int, str, list[float]]: Predicted id, label, and probabilities.
    """
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    pred_label = LABEL_MAP[pred_id]
    probabilities = probs.cpu().tolist()

    return pred_id, pred_label, probabilities


def predict_sa(text: str, caption: str | None = None) -> tuple[int, str, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_sa_bertweet_model_dir()
    model_path = str(model_dir.resolve())

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    ).to(device)

    return predict_text(
        text=combine_text_and_caption(text, caption),
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict SA class for a text using the trained BERTweet model."
    )
    parser.add_argument("text", type=str, help="Input text to classify.")
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="Optional caption to combine with the input text.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_sa_bertweet_model_dir()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    pred_id, pred_label, probabilities = predict_text(
        text=combine_text_and_caption(args.text, args.caption or None),
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    print("Input text:")
    print(args.text)
    print()

    if args.caption:
        print("Caption:")
        print(args.caption)
        print()

    print(f"Predicted class id: {pred_id}")
    print(f"Predicted label: {pred_label}")
    print()

    print("Class probabilities:")
    for class_id, prob in enumerate(probabilities):
        label_name = LABEL_MAP[class_id]
        print(f"  {class_id} ({label_name}): {prob:.4f}")


if __name__ == "__main__":
    main()
