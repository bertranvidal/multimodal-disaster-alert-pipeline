from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sa_text_pipeline import combine_text_and_caption


LABEL_MAP = {
    0: "no_damage",
    1: "damage",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_binary_model_dir() -> Path:
    return get_project_root() / "models" / "sa_transformer_binary"


def predict_text(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 128,
    damage_threshold: float = 0.55,
) -> tuple[int, str, list[float]]:
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    probabilities = probs.cpu().tolist()
    pred_id = int(probabilities[1] >= damage_threshold)
    pred_label = LABEL_MAP[pred_id]

    return pred_id, pred_label, probabilities


def predict_sa_binary(
    text: str,
    caption: str | None = None,
    damage_threshold: float = 0.55,
) -> tuple[int, str, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_binary_model_dir()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)

    return predict_text(
        text=combine_text_and_caption(text, caption),
        model=model,
        tokenizer=tokenizer,
        device=device,
        damage_threshold=damage_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict binary damage class for a text using the trained SA model."
    )
    parser.add_argument("text", type=str, help="Input text to classify.")
    parser.add_argument("--caption", type=str, default="", help="Optional image caption.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Decision threshold for the damage class.",
    )
    args = parser.parse_args()

    pred_id, pred_label, probabilities = predict_sa_binary(
        text=args.text,
        caption=args.caption or None,
        damage_threshold=args.threshold,
    )

    print("Input text:")
    print(args.text)
    print()
    print(f"Predicted class id: {pred_id}")
    print(f"Predicted label: {pred_label}")
    print(f"Damage threshold: {args.threshold:.2f}")
    print()
    print("Class probabilities:")
    for class_id, prob in enumerate(probabilities):
        print(f"  {class_id} ({LABEL_MAP[class_id]}): {prob:.4f}")


if __name__ == "__main__":
    main()
