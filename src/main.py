import os
import pandas as pd

from captioning import generate_caption
from predict_ner import predict_ner
from predict_sa_transformer_binary import predict_sa_binary
from sa_text_pipeline import combine_text_and_caption
from alert_generator import generate_alert


def resolve_image_path(image_path_from_csv: str, images_root: str) -> str:
    """
    Convierte el image_path del CSV en una ruta real dentro de tu carpeta local.

    Ejemplo:
    image_path_from_csv = 'images/train/923466082358263808_0.jpg'
    -> devuelve 'images/train/923466082358263808_0.jpg' si existe
    o también prueba con basename dentro de images_root
    """

    # 1. Intentar usar la ruta tal cual viene en el CSV
    if os.path.exists(image_path_from_csv):
        return image_path_from_csv

    # 2. Quedarse solo con el nombre del archivo
    filename = os.path.basename(image_path_from_csv)

    # 3. Buscar ese archivo dentro de la carpeta raíz de imágenes
    candidate = os.path.join(images_root, filename)
    if os.path.exists(candidate):
        return candidate

    # 4. Buscar recursivamente por si está en subcarpetas como dev/train/test
    for root, _, files in os.walk(images_root):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(
        f"No se encontró la imagen '{filename}' a partir de '{image_path_from_csv}'"
    )


def run_pipeline(
    text: str,
    image_path: str | None = None,
    caption_prompt: str | None = None,
):
    print("\n--- INPUT ---")
    print("Text:", text)
    print("Image path:", image_path)

    # 1. Captioning
    if image_path and os.path.exists(image_path):
        caption = generate_caption(image_path, prompt=caption_prompt)
        print("Caption:", caption)
    else:
        caption = None
        print("No valid image found.")

    combined_text = combine_text_and_caption(text, caption)

    # 2. NER
    entities = predict_ner(combined_text)
    print("\nEntities:", entities)

    # 3. SA
    _, label, probs = predict_sa_binary(text=text, caption=caption)
    severity = 2 if label == "damage" else 0
    print("Damage class:", label)
    print(f"Probabilities: no_damage={probs[0]:.4f}, damage={probs[1]:.4f}")

    # 4. Alert final
    alert = generate_alert(
        original_text=text,
        caption=caption,
        entities=entities,
        severity=severity,
        sa_label=label,
    )

    print("\n--- FINAL ALERT ---")
    print(alert)

    return {
        "text": text,
        "image_path": image_path,
        "caption": caption,
        "entities": entities,
        "severity": severity,
        "label": label,
        "alert": alert,
    }


if __name__ == "__main__":
    csv_path = "data/SA_data/crisismmd_damage_test.csv"

    # carpeta raíz donde tienes tus imágenes
    images_root = "images/test"

    # fila que quieres probar
    row_index = 30

    # prompt opcional
    caption_prompt = "a disaster scene of"

    df = pd.read_csv(csv_path)
    row = df.iloc[row_index]

    text = row["tweet_text"]
    image_path_from_csv = row["image_path"]

    image_path = resolve_image_path(
        image_path_from_csv=image_path_from_csv,
        images_root=images_root,
    )

    print("Image path from CSV:", image_path_from_csv)
    print("Resolved image path:", image_path)

    run_pipeline(
        text=text,
        image_path=image_path,
        caption_prompt=caption_prompt,
    )
