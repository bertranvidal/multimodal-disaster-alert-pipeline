from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# puedes usar base o large
MODEL_NAME = "Salesforce/blip-image-captioning-base"
# MODEL_NAME = "Salesforce/blip-image-captioning-large"  # más lento pero mejor

processor = BlipProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, local_files_only=True)


def generate_caption(image_path: str, prompt: str | None = None) -> str:
    """
    Generate image caption, optionally using a prompt.

    Args:
        image_path: path to image
        prompt: optional text prompt (e.g. "a disaster scene of")

    Returns:
        caption (str)
    """

    image = Image.open(image_path).convert("RGB")

    if prompt:
        inputs = processor(image, prompt, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")

    out = model.generate(
        **inputs,
        max_new_tokens=30  # evita captions cortados
    )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
