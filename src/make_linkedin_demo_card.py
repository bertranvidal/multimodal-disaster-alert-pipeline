from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from alert_generator import generate_alert
from captioning import generate_caption
from predict_ner import predict_ner
from predict_sa_transformer_binary import predict_sa_binary
from sa_text_pipeline import combine_text_and_caption


WIDTH = 1200
HEIGHT = 1850
MARGIN = 58


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def resolve_image_path(image_path_from_csv: str) -> Path:
    image_path = Path(str(image_path_from_csv))
    if image_path.exists():
        return image_path

    project_candidate = get_project_root() / image_path
    if project_candidate.exists():
        return project_candidate

    filename = image_path.name
    for candidate in (get_project_root() / "images").rglob(filename):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"No se encontro la imagen: {image_path_from_csv}")


def load_sample(split: str, row_index: int) -> tuple[str, Path]:
    csv_path = get_project_root() / "data" / "SA_data" / f"crisismmd_damage_{split}.csv"
    row = pd.read_csv(csv_path).iloc[row_index]
    return str(row["tweet_text"]), resolve_image_path(str(row["image_path"]))


def fit_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((int(src_w * scale), int(src_h * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def draw_round_rect(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str | tuple[int, int, int, int] | None,
    outline: str | None = None,
    radius: int = 18,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    max_chars: int,
    fill: str,
    text_font: ImageFont.FreeTypeFont,
    line_spacing: int = 6,
) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=max_chars) or [""])

    for line in lines:
        draw.text((x, y), line, fill=fill, font=text_font)
        bbox = draw.textbbox((x, y), line or "Ag", font=text_font)
        y += bbox[3] - bbox[1] + line_spacing
    return y


def draw_section(
    draw: ImageDraw.ImageDraw,
    title: str,
    body: str,
    x: int,
    y: int,
    w: int,
    max_chars: int,
    title_color: str = "#0f172a",
) -> int:
    title_font = font(24, bold=True)
    body_font = font(22)
    draw.text((x, y), title.upper(), fill=title_color, font=title_font)
    y += 36
    y = draw_wrapped(draw, body, (x, y), max_chars=max_chars, fill="#26323f", text_font=body_font)
    return y + 20


def format_entities(entities: list[dict[str, str]]) -> str:
    if not entities:
        return "No entities detected"
    return "   ".join(f"{entity['text']} [{entity['label']}]" for entity in entities)


def make_alert(text: str, caption: str, entities: list[dict[str, str]], sa_label: str) -> str:
    generate_alert(
        original_text=text,
        caption=caption,
        entities=entities,
        severity=2 if sa_label == "damage" else 0,
        sa_label=sa_label,
    )

    location = None
    disaster = None
    for entity in entities:
        entity_text = entity["text"].strip()
        if entity["label"] in {"LOC", "GPE", "LOCATION"} and location is None and not entity_text.isdigit():
            location = entity["text"]
        elif entity["label"] == "DISASTER" and disaster is None:
            disaster = entity["text"]

    if sa_label == "damage":
        alert = "Severe damage detected"
    else:
        alert = "No severe damage detected"

    if location:
        alert += f" in {location}"
    if disaster:
        alert += f" caused by {disaster}"

    return alert + "."


def build_pipeline_summary(
    caption: str,
    entities: list[dict[str, str]],
    sa_label: str,
    probabilities: list[float],
) -> str:
    relevant_entities = [
        entity
        for entity in entities
        if entity["label"] in {"LOCATION", "DISASTER"} and not entity["text"].strip().isdigit()
    ]
    entity_text = format_entities(relevant_entities or entities)
    return (
        f"Image captioning generated: \"{caption}\". "
        f"The NER module extracted: {entity_text}. "
        f"The binary damage classifier predicted {sa_label} "
        f"(damage={probabilities[1]:.3f}, no_damage={probabilities[0]:.3f})."
    )


def create_card(
    text: str,
    image_path: Path,
    output_path: Path,
    threshold: float,
) -> None:
    caption = generate_caption(str(image_path), prompt="a disaster scene of")
    combined_text = combine_text_and_caption(text, caption)
    entities = predict_ner(combined_text)
    _, sa_label, probabilities = predict_sa_binary(text=text, caption=caption, damage_threshold=threshold)
    alert = make_alert(text, caption, entities, sa_label)
    pipeline_summary = build_pipeline_summary(caption, entities, sa_label, probabilities)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#f3f6f8")
    draw = ImageDraw.Draw(canvas)

    draw.text((MARGIN, 48), "Multimodal Disaster Alert", fill="#0f172a", font=font(46, bold=True))
    draw.text(
        (MARGIN, 108),
        "Image + tweet processed into a final emergency alert",
        fill="#475467",
        font=font(26),
    )

    image_box = (MARGIN, 178, WIDTH - MARGIN, 810)
    photo = fit_cover(
        Image.open(image_path).convert("RGB"),
        (image_box[2] - image_box[0], image_box[3] - image_box[1]),
    )
    canvas.paste(photo, image_box[:2])
    draw_round_rect(draw, image_box, fill=None, outline="#cfd8e3", radius=34, width=3)

    tweet_panel = (MARGIN, 852, WIDTH - MARGIN, 1076)
    draw_round_rect(draw, tweet_panel, fill="#ffffff", outline="#d8dee7", radius=30, width=2)
    draw_round_rect(draw, (MARGIN + 42, 890, MARGIN + 202, 942), fill="#fef3c7", radius=20)
    draw.text((MARGIN + 68, 903), "TWEET", fill="#92400e", font=font(22, bold=True))
    draw_wrapped(
        draw,
        text,
        (MARGIN + 42, 974),
        max_chars=73,
        fill="#26323f",
        text_font=font(27),
        line_spacing=10,
    )

    summary_panel = (MARGIN, 1116, WIDTH - MARGIN, 1428)
    draw_round_rect(draw, summary_panel, fill="#ffffff", outline="#d8dee7", radius=30, width=2)
    draw_round_rect(draw, (MARGIN + 42, 1154, MARGIN + 330, 1206), fill="#e0f2fe", radius=20)
    draw.text((MARGIN + 68, 1167), "PIPELINE OUTPUT", fill="#075985", font=font(22, bold=True))
    draw_wrapped(
        draw,
        pipeline_summary,
        (MARGIN + 42, 1240),
        max_chars=70,
        fill="#26323f",
        text_font=font(26),
        line_spacing=10,
    )

    final_panel = (MARGIN, 1468, WIDTH - MARGIN, 1758)
    draw_round_rect(draw, final_panel, fill="#ffffff", outline="#d8dee7", radius=34, width=2)

    badge_fill = "#dcfce7" if sa_label == "damage" else "#dbeafe"
    badge_text = "#14532d" if sa_label == "damage" else "#1e3a8a"
    draw_round_rect(draw, (MARGIN + 42, 1508, MARGIN + 300, 1564), fill=badge_fill, radius=22)
    draw.text((MARGIN + 68, 1521), "FINAL ALERT", fill=badge_text, font=font(24, bold=True))

    draw_wrapped(
        draw,
        alert,
        (MARGIN + 42, 1610),
        max_chars=27,
        fill="#101828",
        text_font=font(50, bold=True),
        line_spacing=14,
    )

    draw.text(
        (MARGIN, 1782),
        "Generated locally from a multimodal ML pipeline",
        fill="#667085",
        font=font(22),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    print("--- LINKEDIN CARD CREATED ---")
    print(output_path)
    print()
    print("--- SUMMARY ---")
    print("Caption:", caption)
    print("NER:", format_entities(entities))
    print("SA:", sa_label, f"(damage={probabilities[1]:.4f}, no_damage={probabilities[0]:.4f})")
    print("Alert:", alert)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a polished one-image LinkedIn demo card.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--row-index", type=int, default=41)
    parser.add_argument("--text", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output", default="outputs/linkedin_final_alert_card.png")
    args = parser.parse_args()

    if args.text and args.image_path:
        text = args.text
        image_path = resolve_image_path(args.image_path)
    else:
        text, image_path = load_sample(args.split, args.row_index)

    create_card(
        text=text,
        image_path=image_path,
        output_path=get_project_root() / args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
