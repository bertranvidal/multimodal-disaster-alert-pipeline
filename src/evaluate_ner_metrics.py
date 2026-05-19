from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from ner_char_bilstm import CharCNNBiLSTMNER
from ner_dataset import PAD_CHAR, PAD_TOKEN, build_sentences, load_ner_csv
from train_ner import bio_entity_spans, build_loader, get_model_dir


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_model(
    model_dir: Path,
    word2idx: dict[str, int],
    char2idx: dict[str, int],
    tag2idx: dict[str, int],
    config: dict[str, float | int],
    device: torch.device,
) -> CharCNNBiLSTMNER:
    model = CharCNNBiLSTMNER(
        vocab_size=len(word2idx),
        char_vocab_size=len(char2idx),
        embedding_dim=int(config["embedding_dim"]),
        char_embedding_dim=int(config["char_embedding_dim"]),
        char_num_filters=int(config["char_num_filters"]),
        hidden_dim=int(config["hidden_dim"]),
        num_tags=len(tag2idx),
        pad_idx=word2idx[PAD_TOKEN],
        char_pad_idx=char2idx[PAD_CHAR],
        unk_idx=word2idx["<UNK>"],
        dropout=float(config["dropout"]),
        word_dropout=float(config["word_dropout"]),
    ).to(device)
    model.load_state_dict(
        torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    )
    return model


def evaluate_entity_metrics_by_type(
    model: CharCNNBiLSTMNER,
    split: str,
    word2idx: dict[str, int],
    char2idx: dict[str, int],
    tag2idx: dict[str, int],
    idx2tag: dict[int, str],
    device: torch.device,
    batch_size: int = 16,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    sentences, tags = build_sentences(load_ner_csv(split))
    loader = build_loader(
        sentences,
        tags,
        word2idx,
        char2idx,
        tag2idx,
        batch_size=batch_size,
        shuffle=False,
    )

    total_gold = 0
    total_pred = 0
    total_correct = 0

    gold_by_type: dict[str, int] = defaultdict(int)
    pred_by_type: dict[str, int] = defaultdict(int)
    correct_by_type: dict[str, int] = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            label_ids = batch["label_ids"].to(device)

            logits = model(input_ids, char_ids)
            preds = torch.argmax(logits, dim=-1)

            batch_size_current = input_ids.size(0)
            for i in range(batch_size_current):
                length = int(batch["lengths"][i].item())
                true_tags = [idx2tag[idx] for idx in label_ids[i][:length].cpu().tolist()]
                pred_tags = [idx2tag[idx] for idx in preds[i][:length].cpu().tolist()]

                gold_spans = bio_entity_spans(true_tags)
                pred_spans = bio_entity_spans(pred_tags)
                correct_spans = gold_spans & pred_spans

                total_gold += len(gold_spans)
                total_pred += len(pred_spans)
                total_correct += len(correct_spans)

                for _, _, label in gold_spans:
                    gold_by_type[label] += 1
                for _, _, label in pred_spans:
                    pred_by_type[label] += 1
                for _, _, label in correct_spans:
                    correct_by_type[label] += 1

    global_precision = total_correct / total_pred if total_pred else 0.0
    global_recall = total_correct / total_gold if total_gold else 0.0
    global_f1 = (
        2 * global_precision * global_recall / (global_precision + global_recall)
        if global_precision + global_recall > 0
        else 0.0
    )

    all_labels = sorted(set(gold_by_type) | set(pred_by_type))
    per_type: dict[str, dict[str, float | int]] = {}
    for label in all_labels:
        label_precision = (
            correct_by_type[label] / pred_by_type[label]
            if pred_by_type[label]
            else 0.0
        )
        label_recall = (
            correct_by_type[label] / gold_by_type[label]
            if gold_by_type[label]
            else 0.0
        )
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall > 0
            else 0.0
        )
        per_type[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "gold": gold_by_type[label],
            "predicted": pred_by_type[label],
            "correct": correct_by_type[label],
        }

    global_metrics = {
        "precision": global_precision,
        "recall": global_recall,
        "f1": global_f1,
        "gold": total_gold,
        "predicted": total_pred,
        "correct": total_correct,
    }

    return global_metrics, per_type


def print_markdown_table(global_metrics: dict[str, float], per_type: dict[str, dict[str, float | int]]) -> None:
    print("Global entity-level metrics")
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Precision | {global_metrics['precision']:.4f} |")
    print(f"| Recall | {global_metrics['recall']:.4f} |")
    print(f"| F1 | {global_metrics['f1']:.4f} |")
    print()
    print("Per-entity metrics")
    print("| Entity | Precision | Recall | F1 | Gold | Predicted | Correct |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, metrics in per_type.items():
        print(
            f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['gold']} | {metrics['predicted']} | "
            f"{metrics['correct']} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved NER model and print report tables.")
    parser.add_argument("--split", default="test", help="Split to evaluate: dev, test, or hard_test.")
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional output path to save metrics as JSON.",
    )
    args = parser.parse_args()

    model_dir = get_model_dir()
    word2idx = load_json(model_dir / "word2idx.json")
    char2idx = load_json(model_dir / "char2idx.json")
    tag2idx = load_json(model_dir / "tag2idx.json")
    idx2tag_raw = load_json(model_dir / "idx2tag.json")
    config = load_json(model_dir / "config.json")
    idx2tag = {int(idx): tag for idx, tag in idx2tag_raw.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_dir, word2idx, char2idx, tag2idx, config, device)

    global_metrics, per_type = evaluate_entity_metrics_by_type(
        model=model,
        split=args.split,
        word2idx=word2idx,
        char2idx=char2idx,
        tag2idx=tag2idx,
        idx2tag=idx2tag,
        device=device,
        batch_size=int(config.get("batch_size", 16)),
    )

    print(f"Split: {args.split}")
    print_markdown_table(global_metrics, per_type)

    if args.json_out:
        output = {
            "split": args.split,
            "global": global_metrics,
            "per_entity": per_type,
        }
        with open(args.json_out, "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON metrics to: {args.json_out}")


if __name__ == "__main__":
    main()
