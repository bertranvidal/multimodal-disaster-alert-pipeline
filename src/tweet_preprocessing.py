from __future__ import annotations

import re
import unicodedata


TOKEN_PATTERN = re.compile(r"\w+(?:-\w+)*|[^\w\s]")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+:?")
HASHTAG_PATTERN = re.compile(r"#([A-Za-z][A-Za-z0-9_]+)")
CAMEL_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])")
MULTISPACE_PATTERN = re.compile(r"\s+")
CONTRACTION_PATTERNS = [
    (re.compile(r"\b([Ii])t['’]s\b"), r"\1t is"),
    (re.compile(r"\b([Tt])hat['’]s\b"), r"\1hat is"),
    (re.compile(r"\b([Ww])hat['’]s\b"), r"\1hat is"),
    (re.compile(r"\b([Tt]here)['’]s\b"), r"\1 is"),
    (re.compile(r"\b([Ww]e)['’]re\b"), r"\1 are"),
    (re.compile(r"\b([Tt]hey)['’]re\b"), r"\1 are"),
    (re.compile(r"\b([Yy]ou)['’]re\b"), r"\1 are"),
    (re.compile(r"\b([Cc]an)['’]t\b"), r"\1 not"),
    (re.compile(r"\b([Ww])on['’]t\b"), r"\1ill not"),
    (re.compile(r"\b([Dd])on['’]t\b"), r"\1o not"),
]


def split_camel_case(text: str) -> str:
    """
    Split CamelCase words into space-separated tokens.
    """
    return CAMEL_BOUNDARY_PATTERN.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """
    Normalize quotes and other Unicode variants that commonly appear in tweets.
    """
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u00b4": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def remove_symbol_noise(text: str) -> str:
    """
    Remove emoji-like or decorative symbols that do not help the NER model.
    """
    kept_chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("S") and char not in {"#", "@", "$", "%", "&"}:
            kept_chars.append(" ")
        else:
            kept_chars.append(char)
    return "".join(kept_chars)


def expand_contractions(text: str) -> str:
    """
    Expand a small set of common English contractions found in tweets.
    """
    for pattern, replacement in CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def expand_hashtag(match: re.Match[str]) -> str:
    """
    Turn a hashtag into a token sequence closer to natural text.
    """
    hashtag_body = match.group(1).replace("_", " ")
    hashtag_body = split_camel_case(hashtag_body)
    return f" {hashtag_body} "


def normalize_tweet_for_ner(text: str) -> str:
    """
    Normalize noisy tweet text before NER inference.

    The goal is to keep semantically useful content while removing platform-specific noise.
    """
    text = normalize_unicode(text)
    text = remove_symbol_noise(text)
    text = expand_contractions(text)
    text = URL_PATTERN.sub(" ", text)
    text = re.sub(r"(?<!\w)RT\b", " ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(expand_hashtag, text)
    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize_for_ner(text: str) -> list[str]:
    """
    Tokenize text with a regex tokenizer that handles punctuation better than whitespace split.
    """
    return TOKEN_PATTERN.findall(text)
