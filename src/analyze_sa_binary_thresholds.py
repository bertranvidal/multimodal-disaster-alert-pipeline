from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sa_dataset import load_sa_csv
from sa_text_pipeline import DEFAULT_CAPTION_PROMPT, build_sa_inputs_from_dataframe
from train_sa_transformer import SATransformerDataset


def get_probabilities(split: str, model, tokenizer, device) -> tuple[np.ndarray, np.ndarray]:
    df = load_sa_csv(split)
    texts = build_sa_inputs_from_dataframe(
        df,
        use_captions=True,
        caption_prompt=DEFAULT_CAPTION_PROMPT,
    )
    labels = (df["label"].astype(int) > 0).astype(int).to_numpy()
    dataset = SATransformerDataset(texts, labels.tolist(), tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            probabilities.extend(probs.cpu().tolist())

    return labels, np.array(probabilities)


def main() -> None:
    model_dir = "models/sa_transformer_binary"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)

    dev_labels, dev_probs = get_probabilities("dev", model, tokenizer, device)
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 19):
        dev_pred = (dev_probs >= threshold).astype(int)
        candidates.append(
            (
                f1_score(dev_labels, dev_pred, average="macro"),
                accuracy_score(dev_labels, dev_pred),
                threshold,
            )
        )

    best_f1, best_acc, best_threshold = max(candidates)
    print("Best dev threshold")
    print(f"threshold = {best_threshold:.2f}")
    print(f"dev_accuracy = {best_acc:.4f}")
    print(f"dev_macro_f1 = {best_f1:.4f}")

    test_labels, test_probs = get_probabilities("test", model, tokenizer, device)
    test_pred = (test_probs >= best_threshold).astype(int)
    print("\nTest metrics at best dev threshold")
    print(f"accuracy = {accuracy_score(test_labels, test_pred):.4f}")
    print(f"macro_f1 = {f1_score(test_labels, test_pred, average='macro'):.4f}")
    print("confusion_matrix =")
    for row in confusion_matrix(test_labels, test_pred, labels=[0, 1]).tolist():
        print(row)
    print("\nclassification_report =")
    print(
        classification_report(
            test_labels,
            test_pred,
            target_names=["no_damage", "damage"],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
