from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

DEFAULT_CAPTION_PROMPT = "a disaster scene of"
GENERIC_CAPTION_PATTERNS = (
    re.compile(r"^a disaster scene of\s*", re.IGNORECASE),
    re.compile(r"^the aftermath of\s*", re.IGNORECASE),
)
LOW_SIGNAL_CAPTION_PATTERNS = (
    re.compile(r"^a man\b", re.IGNORECASE),
    re.compile(r"^a woman\b", re.IGNORECASE),
    re.compile(r"^a person\b", re.IGNORECASE),
    re.compile(r"^people\b", re.IGNORECASE),
    re.compile(r"^a group of people\b", re.IGNORECASE),
    re.compile(r"^a boat in the ocean\b", re.IGNORECASE),
)
DISASTER_KEYWORDS = {
    "damage",
    "damaged",
    "debris",
    "destroyed",
    "flood",
    "flooded",
    "flooding",
    "fire",
    "burning",
    "wildfire",
    "hurricane",
    "storm",
    "tornado",
    "earthquake",
    "collapse",
    "collapsed",
    "wreckage",
    "ruins",
    "smoke",
    "ash",
    "mud",
    "landslide",
    "rescue",
}


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_sa_data_dir() -> Path:
    """
    Return the directory containing SA data files.

    Returns:
        Path: Absolute path to data/SA_data.
    """
    return get_project_root() / "data" / "SA_data"


def get_caption_cache_path() -> Path:
    """
    Return the caption cache path used by SA.

    Returns:
        Path: Absolute path to the caption cache JSON file.
    """
    return get_sa_data_dir() / "caption_cache.json"


def resolve_image_path(image_path_from_csv: str) -> Path | None:
    """
    Resolve an image path stored in a CSV row to a local file.

    Args:
        image_path_from_csv: Relative or absolute image path from the dataset.

    Returns:
        Path | None: Resolved local path if found, otherwise None.
    """
    if not image_path_from_csv or pd.isna(image_path_from_csv):
        return None

    image_path = Path(str(image_path_from_csv))
    if image_path.exists():
        return image_path

    project_candidate = get_project_root() / image_path
    if project_candidate.exists():
        return project_candidate

    filename = image_path.name
    images_root = get_project_root() / "images"
    if not images_root.exists():
        return None

    for candidate in images_root.rglob(filename):
        if candidate.is_file():
            return candidate

    return None


def combine_text_and_caption(text: str, caption: str | None = None) -> str:
    """
    Combine tweet text and caption in a stable format for SA.

    Args:
        text: Original tweet text.
        caption: Optional generated caption.

    Returns:
        str: Combined input string for the SA model.
    """
    clean_text = str(text).strip()
    clean_caption = str(caption).strip() if caption else ""

    if not clean_caption:
        return clean_text

    return f"Tweet: {clean_text}\nCaption: {clean_caption}"


def normalize_caption_text(caption: str) -> str:
    """
    Normalize a generated caption before deciding whether to use it for SA.

    Args:
        caption: Raw generated caption.

    Returns:
        str: Cleaned caption text.
    """
    clean_caption = str(caption).strip()
    for pattern in GENERIC_CAPTION_PATTERNS:
        clean_caption = pattern.sub("", clean_caption).strip()

    clean_caption = re.sub(r"\s+", " ", clean_caption)
    clean_caption = clean_caption.strip(" ,.;:-")
    return clean_caption


def has_repeated_bigram(caption: str) -> bool:
    """
    Detect obvious repetition artifacts in generated captions.

    Args:
        caption: Caption text.

    Returns:
        bool: True when repetition artifacts are detected.
    """
    tokens = caption.lower().split()
    if len(tokens) < 4:
        return False

    bigrams = list(zip(tokens, tokens[1:]))
    for index in range(len(bigrams) - 1):
        if bigrams[index] == bigrams[index + 1]:
            return True

    return False


def is_caption_useful_for_sa(caption: str | None) -> bool:
    """
    Decide whether a generated caption is informative enough for SA.

    Args:
        caption: Raw or normalized caption.

    Returns:
        bool: True when the caption is worth including.
    """
    if not caption:
        return False

    clean_caption = normalize_caption_text(caption)
    if not clean_caption:
        return False

    if len(clean_caption.split()) < 3:
        return False

    if has_repeated_bigram(clean_caption):
        return False

    lower_caption = clean_caption.lower()
    if any(pattern.match(lower_caption) for pattern in LOW_SIGNAL_CAPTION_PATTERNS):
        return False

    caption_tokens = set(re.findall(r"[a-z]+", lower_caption))
    if not (caption_tokens & DISASTER_KEYWORDS):
        return False

    return True


def prepare_caption_for_sa(caption: str | None) -> str | None:
    """
    Clean and filter a generated caption before it is concatenated for SA.

    Args:
        caption: Raw generated caption.

    Returns:
        str | None: Cleaned caption if useful, otherwise None.
    """
    if not caption:
        return None

    clean_caption = normalize_caption_text(caption)
    if not is_caption_useful_for_sa(clean_caption):
        return None

    return clean_caption


def load_caption_cache(cache_path: Path | None = None) -> dict[str, str]:
    """
    Load the caption cache if it exists.

    Args:
        cache_path: Optional cache path override.

    Returns:
        dict[str, str]: Cached captions indexed by image path.
    """
    path = cache_path or get_caption_cache_path()
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_caption_cache(cache: dict[str, str], cache_path: Path | None = None) -> None:
    """
    Persist the caption cache to disk.

    Args:
        cache: Cache content to store.
        cache_path: Optional cache path override.
    """
    path = cache_path or get_caption_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def generate_caption_for_sa(
    image_path: Path,
    prompt: str = DEFAULT_CAPTION_PROMPT,
) -> str | None:
    """
    Generate a caption for one image, returning None on failure.

    Args:
        image_path: Local image path.
        prompt: Prompt sent to the captioning model.

    Returns:
        str | None: Generated caption or None if generation fails.
    """
    try:
        from captioning import generate_caption

        return generate_caption(str(image_path), prompt=prompt)
    except Exception as error:
        print(f"[SA] Warning: caption generation failed for {image_path}: {error}")
        return None


def build_sa_inputs_from_dataframe(
    df: pd.DataFrame,
    use_captions: bool = True,
    caption_prompt: str = DEFAULT_CAPTION_PROMPT,
    cache_path: Path | None = None,
) -> list[str]:
    """
    Build SA input texts for a dataframe, optionally enriching them with captions.

    Args:
        df: Dataframe containing tweet_text and optionally image_path.
        use_captions: Whether to enrich tweets with generated captions.
        caption_prompt: Prompt sent to the captioning model.
        cache_path: Optional caption cache path.

    Returns:
        list[str]: Combined input strings aligned with the SA pipeline.
    """
    if not use_captions:
        return [combine_text_and_caption(text) for text in df["tweet_text"].astype(str).tolist()]

    cache = load_caption_cache(cache_path)
    cache_updated = False
    inputs: list[str] = []

    for row in df.itertuples(index=False):
        tweet_text = str(getattr(row, "tweet_text"))
        image_path_value = getattr(row, "image_path", None)
        caption: str | None = None

        resolved_path = resolve_image_path(str(image_path_value)) if image_path_value is not None else None
        if resolved_path is not None:
            cache_key = os.path.relpath(resolved_path, get_project_root())
            raw_caption = cache.get(cache_key)
            if raw_caption is None:
                raw_caption = generate_caption_for_sa(
                    image_path=resolved_path,
                    prompt=caption_prompt,
                )
                if raw_caption:
                    cache[cache_key] = raw_caption
                    cache_updated = True
            caption = prepare_caption_for_sa(raw_caption)

        inputs.append(combine_text_and_caption(tweet_text, caption))

    if cache_updated:
        save_caption_cache(cache, cache_path)

    return inputs
