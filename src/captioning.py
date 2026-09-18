from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_NAME = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def load_captioning_model() -> tuple[
    BlipProcessor,
    BlipForConditionalGeneration,
    torch.device,
]:
    """Load BLIP once and reuse it across predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model, device


def generate_caption(image_path: str, prompt: str | None = None) -> str:
    """Generate a caption for an image, optionally conditioned on a prompt."""
    processor, model, device = load_captioning_model()
    image = Image.open(image_path).convert("RGB")

    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)

    return processor.decode(output[0], skip_special_tokens=True)
