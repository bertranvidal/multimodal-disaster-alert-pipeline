from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from captioning import generate_caption
from predict_ner import predict_ner
from predict_sa_transformer_binary import predict_sa_binary
from sa_text_pipeline import combine_text_and_caption


WIDTH = 1600
HEIGHT = 2100
BG = "#ffffff"
TEXT = "#111827"
MUTED = "#4b5563"
GREEN = "#d6f7d6"
ORANGE = "#fde7cf"
PURPLE = "#d9d6ff"
GRAY = "#f3f4f6"
YELLOW = "#fff3c4"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def resolve_image_path(image_path_from_csv: str) -> Path:
    path = Path(str(image_path_from_csv))
    if path.exists():
        return path

    project_path = root() / path
    if project_path.exists():
        return project_path

    for candidate in (root() / "images").rglob(path.name):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(image_path_from_csv)


def load_example(split: str = "test", row_index: int = 6) -> tuple[str, Path]:
    df = pd.read_csv(root() / "data" / "SA_data" / f"crisismmd_damage_{split}.csv")
    row = df.iloc[row_index]
    tweet = str(row["tweet_text"]).replace("·ΩÑ9", "").strip()
    return tweet, resolve_image_path(str(row["image_path"]))


def fit_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def wrapped(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=max_chars) or [""])
    return lines


def draw_dashed_border(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = xy
    dash = 14
    gap = 10
    for x in range(x1 + 14, x2 - 14, dash + gap):
        draw.line((x, y1, min(x + dash, x2 - 14), y1), fill=TEXT, width=3)
        draw.line((x, y2, min(x + dash, x2 - 14), y2), fill=TEXT, width=3)
    for y in range(y1 + 14, y2 - 14, dash + gap):
        draw.line((x1, y, x1, min(y + dash, y2 - 14)), fill=TEXT, width=3)
        draw.line((x2, y, x2, min(y + dash, y2 - 14)), fill=TEXT, width=3)


def draw_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str = "",
    fill: str = GRAY,
    dashed: bool = False,
    title_bold: bool = False,
    title_size: int = 31,
    body_size: int = 25,
    max_chars: int = 30,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=15, fill=fill, outline=TEXT, width=3)
    if dashed:
        draw_dashed_border(draw, xy)

    cx = (x1 + x2) // 2
    y = y1 + 18
    title_font = font(title_size, bold=title_bold)
    body_font = font(body_size)

    if title:
        for line in wrapped(title, max_chars):
            bbox = draw.textbbox((0, 0), line, font=title_font)
            draw.text((cx - (bbox[2] - bbox[0]) / 2, y), line, fill=TEXT, font=title_font)
            y += bbox[3] - bbox[1] + 5

    if body:
        y += 4
        for line in wrapped(body, max_chars):
            bbox = draw.textbbox((0, 0), line, font=body_font)
            draw.text((cx - (bbox[2] - bbox[0]) / 2, y), line, fill=TEXT, font=body_font)
            y += bbox[3] - bbox[1] + 4


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((*start, *end), fill=TEXT, width=4)
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    size = 17
    p1 = (ex, ey)
    p2 = (ex - ux * size + px * size * 0.55, ey - uy * size + py * size * 0.55)
    p3 = (ex - ux * size - px * size * 0.55, ey - uy * size - py * size * 0.55)
    draw.polygon([p1, p2, p3], fill=TEXT)


def entities_for_display(entities: list[dict[str, str]]) -> str:
    selected = []
    seen = set()
    has_specific_hurricane = any("hurricane " in entity["text"].lower() for entity in entities)
    temporal_words = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    for entity in entities:
        entity_text = entity["text"].strip()
        entity_key = (entity_text.lower(), entity["label"])
        if entity["label"] not in {"LOCATION", "DISASTER"} or entity_text.isdigit():
            continue
        if entity_text.lower() in temporal_words:
            continue
        if has_specific_hurricane and entity_text.lower() == "hurricane":
            continue
        if entity_key in seen:
            continue
        selected.append(entity)
        seen.add(entity_key)

    return "\n".join(f"{entity['text']} [{entity['label']}]" for entity in selected[:4])


def default_alert(split: str, row_index: int) -> str:
    alerts = {
        ("test", 6): "Severe damage detected in Naples caused by Hurricane Irma.",
        ("test", 39): "Severe damage detected in Houston caused by Hurricane Harvey.",
        ("train", 3): "Severe damage detected in Crestline caused by tornado.",
    }
    return alerts.get((split, row_index), "Severe damage detected.")


def create_diagram(output: Path, split: str = "test", row_index: int = 6, final_alert: str | None = None) -> None:
    tweet, image_path = load_example(split=split, row_index=row_index)
    caption = generate_caption(str(image_path), prompt="a disaster scene of")
    combined = combine_text_and_caption(tweet, caption)
    entities = predict_ner(combined)
    _, sa_label, probs = predict_sa_binary(tweet, caption)
    final_alert = final_alert or default_alert(split, row_index)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((90, 58), "Multimodal Disaster Alert Pipeline", fill=TEXT, font=font(54, bold=True))
    draw.text((90, 126), "Example flow using a tweet and disaster image from the dataset", fill=MUTED, font=font(30))

    # Main boxes. This is intentionally close to the original guide layout,
    # but with extra vertical spacing and smaller output boxes at the top.
    final_box = (520, 210, 1080, 320)
    alert_box = (610, 420, 990, 525)
    ner_out = (170, 555, 565, 680)
    sa_out = (1035, 555, 1430, 680)
    ner_box = (190, 790, 590, 900)
    sa_box = (1010, 790, 1410, 900)
    combined_box = (590, 990, 1010, 1115)
    caption_box = (560, 1205, 1040, 1345)
    captioning_box = (610, 1460, 990, 1570)
    image_box = (560, 1665, 1040, 2060)
    tweet_box = (60, 1040, 475, 1285)

    draw_box(draw, final_box, "Final Alert", final_alert, fill=GRAY, dashed=True, max_chars=38, body_size=22)
    draw_box(draw, alert_box, "Alert", "Generation", fill=PURPLE, title_size=32, body_size=27)
    draw_box(draw, ner_out, "", entities_for_display(entities), fill=GRAY, dashed=True, body_size=26, max_chars=26)
    draw_box(draw, sa_out, "Binary output", f"{sa_label}\nP(damage) = {probs[1]:.3f}", fill=GRAY, dashed=True, body_size=25, max_chars=24)
    draw_box(draw, ner_box, "Named Entity", "Recognition (NER)", fill=GREEN, title_size=31, body_size=25)
    draw_box(draw, sa_box, "Damage", "Classification (SA)", fill=ORANGE, title_size=31, body_size=25)
    draw_box(draw, combined_box, "Combined Text", "(Tweet + Caption)", fill=GRAY, dashed=True, title_size=31, body_size=25)
    draw_box(draw, caption_box, "Caption Text", caption, fill=GRAY, dashed=True, title_size=31, body_size=24, max_chars=34)
    draw_box(draw, captioning_box, "Image", "Captioning", fill=PURPLE, title_size=31, body_size=25)
    draw_box(draw, tweet_box, "Text Input:", f"\"{tweet}\"", fill=YELLOW, title_bold=True, title_size=30, body_size=24, max_chars=27)

    draw.rounded_rectangle(image_box, radius=16, fill=GRAY, outline=TEXT, width=3)
    draw.text((image_box[0] + 88, image_box[1] + 28), "Image Input", fill=TEXT, font=font(29, bold=True))
    img = fit_cover(Image.open(image_path).convert("RGB"), (410, 270))
    canvas.paste(img, (image_box[0] + 35, image_box[1] + 105))

    # Vertical image/caption/combined/alert flow.
    arrow(draw, (800, image_box[1]), (800, captioning_box[3]))
    arrow(draw, (800, captioning_box[1]), (800, caption_box[3]))
    arrow(draw, (800, caption_box[1]), (800, combined_box[3]))
    arrow(draw, (800, combined_box[1]), (800, alert_box[3]))
    arrow(draw, (800, alert_box[1]), (800, final_box[3]))

    # Text input to combined text.
    arrow(draw, (475, 1160), (590, 1055))

    # Combined text to NER and SA, kept clear of boxes.
    arrow(draw, (650, 990), (500, 900))
    arrow(draw, (950, 990), (1100, 900))

    # NER/SA modules to their outputs.
    arrow(draw, (390, 790), (390, 680))
    arrow(draw, (1210, 790), (1210, 680))

    # Outputs into alert generation. These terminate at side/top edges,
    # not inside label text.
    arrow(draw, (565, 610), (650, 525))
    arrow(draw, (1035, 610), (950, 525))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)
    print("Tweet:", tweet)
    print("Caption:", caption)
    print("Entities:", entities_for_display(entities))
    print("SA:", sa_label, f"damage={probs[1]:.3f}")
    print("Final alert:", final_alert)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LinkedIn-ready multimodal pipeline flow diagram.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--row-index", type=int, default=6)
    parser.add_argument("--output", default="outputs/linkedin_pipeline_flow_diagram.png")
    parser.add_argument("--final-alert", default="")
    args = parser.parse_args()

    create_diagram(
        root() / args.output,
        split=args.split,
        row_index=args.row_index,
        final_alert=args.final_alert or None,
    )


if __name__ == "__main__":
    main()
