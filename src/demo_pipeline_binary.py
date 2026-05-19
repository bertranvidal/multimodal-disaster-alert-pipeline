from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from alert_generator import generate_alert
from captioning import generate_caption
from predict_ner import predict_ner
from predict_sa_transformer_binary import predict_sa_binary
from sa_text_pipeline import combine_text_and_caption


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_image_path(image_path_from_csv: str) -> Path:
    image_path = Path(str(image_path_from_csv))
    if image_path.exists():
        return image_path

    project_candidate = get_project_root() / image_path
    if project_candidate.exists():
        return project_candidate

    filename = image_path.name
    images_root = get_project_root() / "images"
    for candidate in images_root.rglob(filename):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"No se encontro la imagen: {image_path_from_csv}")


def load_sample_from_csv(split: str, row_index: int) -> tuple[str, Path]:
    csv_path = get_project_root() / "data" / "SA_data" / f"crisismmd_damage_{split}.csv"
    df = pd.read_csv(csv_path)
    row = df.iloc[row_index]
    text = str(row["tweet_text"])
    image_path = resolve_image_path(str(row["image_path"]))
    return text, image_path


def format_entities(entities: list[dict[str, str]]) -> str:
    if not entities:
        return "No entities detected"
    return "\n".join(f"- {entity['text']} ({entity['label']})" for entity in entities)


def binary_alert(
    original_text: str,
    caption: str,
    entities: list[dict[str, str]],
    sa_label: str,
) -> str:
    severity = 2 if sa_label == "damage" else 0
    alert_label = "damage" if sa_label == "damage" else "no_damage"
    return generate_alert(
        original_text=original_text,
        caption=caption,
        entities=entities,
        severity=severity,
        sa_label=alert_label,
    )


def save_demo_figure(
    image_path: Path,
    text: str,
    caption: str,
    entities: list[dict[str, str]],
    sa_label: str,
    probabilities: list[float],
    alert: str,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input image")

    result_text = (
        "Tweet\n"
        f"{textwrap.fill(text, 70)}\n\n"
        "Caption\n"
        f"{textwrap.fill(caption, 70)}\n\n"
        "NER\n"
        f"{format_entities(entities)}\n\n"
        "SA binary\n"
        f"label: {sa_label}\n"
        f"no_damage: {probabilities[0]:.4f}\n"
        f"damage: {probabilities[1]:.4f}\n\n"
        "Final alert\n"
        f"{textwrap.fill(alert, 70)}"
    )

    axes[1].axis("off")
    axes[1].text(
        0.0,
        1.0,
        result_text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        wrap=True,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one binary multimodal alert demo.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--row-index", type=int, default=30)
    parser.add_argument("--text", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--caption-prompt", default="a disaster scene of")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", default="outputs/demo_pipeline_binary.png")
    args = parser.parse_args()

    if args.text and args.image_path:
        text = args.text
        image_path = resolve_image_path(args.image_path)
    else:
        text, image_path = load_sample_from_csv(args.split, args.row_index)

    caption = generate_caption(str(image_path), prompt=args.caption_prompt)
    combined_text = combine_text_and_caption(text, caption)
    entities = predict_ner(combined_text)
    _, sa_label, probabilities = predict_sa_binary(
        text=text,
        caption=caption,
        damage_threshold=args.threshold,
    )
    alert = binary_alert(text, caption, entities, sa_label)

    output_path = get_project_root() / args.output
    save_demo_figure(
        image_path=image_path,
        text=text,
        caption=caption,
        entities=entities,
        sa_label=sa_label,
        probabilities=probabilities,
        alert=alert,
        output_path=output_path,
    )

    print("\n--- INPUT ---")
    print("Text:", text)
    print("Image path:", image_path)
    print()
    print("--- CAPTION ---")
    print(caption)
    print()
    print("--- NER ---")
    print(format_entities(entities))
    print()
    print("--- SA BINARY ---")
    print("Label:", sa_label)
    print(f"no_damage: {probabilities[0]:.4f}")
    print(f"damage: {probabilities[1]:.4f}")
    print(f"threshold: {args.threshold:.2f}")
    print()
    print("--- FINAL ALERT ---")
    print(alert)
    print()
    print("--- IMAGE REPORT ---")
    print(os.path.relpath(output_path, get_project_root()))


if __name__ == "__main__":
    main()
