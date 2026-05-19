from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ner_char_bilstm import CharCNNBiLSTMNER
from ner_dataset import MAX_WORD_LEN, PAD_CHAR, PAD_TOKEN, UNK_CHAR, UNK_TOKEN
from tweet_preprocessing import normalize_tweet_for_ner, tokenize_for_ner


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_model_dir() -> Path:
    """
    Return the directory containing trained NER files.

    Returns:
        Path: Absolute path to models/ner.
    """
    return get_project_root() / "models" / "ner"


def load_json(path: Path) -> dict:
    """
    Load a JSON file into a Python dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        dict: Loaded JSON content.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def decode_entities(tokens: list[str], tags: list[str]) -> list[dict[str, str]]:
    """
    Convert BIO tag predictions into entity spans.

    Args:
        tokens: Input tokens.
        tags: Predicted BIO tags aligned with the tokens.

    Returns:
        list[dict[str, str]]: List of extracted entities with text and label.
    """
    entities: list[dict[str, str]] = []
    current_tokens: list[str] = []
    current_label: str | None = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_tokens and current_label is not None:
                entities.append(
                    {
                        "text": " ".join(current_tokens),
                        "label": current_label,
                    }
                )
                current_tokens = []
                current_label = None
            continue

        prefix, label = tag.split("-", maxsplit=1)

        if prefix == "B":
            if current_tokens and current_label is not None:
                entities.append(
                    {
                        "text": " ".join(current_tokens),
                        "label": current_label,
                    }
                )
            current_tokens = [token]
            current_label = label
        elif prefix == "I" and current_label == label:
            current_tokens.append(token)
        else:
            if current_tokens and current_label is not None:
                entities.append(
                    {
                        "text": " ".join(current_tokens),
                        "label": current_label,
                    }
                )
            current_tokens = [token]
            current_label = label

    if current_tokens and current_label is not None:
        entities.append(
            {
                "text": " ".join(current_tokens),
                "label": current_label,
            }
        )

    return entities


def predict_tags(
    model: CharCNNBiLSTMNER,
    tokens: list[str],
    word2idx: dict[str, int],
    char2idx: dict[str, int],
    idx2tag: dict[int, str],
    device: torch.device,
) -> list[str]:
    """
    Predict BIO tags for a tokenized sentence.

    Args:
        model: Trained NER model.
        tokens: Tokenized input sentence.
        word2idx: Token-to-index mapping.
        idx2tag: Index-to-tag mapping.
        device: Torch device used for computation.

    Returns:
        list[str]: Predicted tag sequence.
    """
    input_ids = [word2idx.get(token, word2idx[UNK_TOKEN]) for token in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    char_ids = [
        [
            char2idx.get(char, char2idx[UNK_CHAR])
            for char in token[:MAX_WORD_LEN]
        ]
        + [char2idx[PAD_CHAR]] * max(0, MAX_WORD_LEN - len(token[:MAX_WORD_LEN]))
        for token in tokens
    ]
    char_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor, char_tensor)
        pred_ids = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()

    return [idx2tag[idx] for idx in pred_ids[: len(tokens)]]


def load_ner_model() -> tuple[
    CharCNNBiLSTMNER,
    dict[str, int],
    dict[str, int],
    dict[int, str],
    torch.device,
]:
    """
    Load the trained NER model and its vocabularies for inference.
    """
    model_dir = get_model_dir()

    word2idx = load_json(model_dir / "word2idx.json")
    char2idx = load_json(model_dir / "char2idx.json")
    tag2idx = load_json(model_dir / "tag2idx.json")
    idx2tag_raw = load_json(model_dir / "idx2tag.json")
    config = load_json(model_dir / "config.json")

    idx2tag = {int(idx): tag for idx, tag in idx2tag_raw.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        unk_idx=word2idx[UNK_TOKEN],
        dropout=float(config["dropout"]),
        word_dropout=float(config["word_dropout"]),
    ).to(device)

    model.load_state_dict(
        torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    )

    return model, word2idx, char2idx, idx2tag, device


# función añadida para combinar modelo ner y sa
def predict_ner(text: str, raw: bool = False) -> list[dict[str, str]]:
    """
    Run the current NER model on input text and return extracted entities.
    """
    model, word2idx, char2idx, idx2tag, device = load_ner_model()
    normalized_text = text if raw else normalize_tweet_for_ner(text)
    tokens = tokenize_for_ner(normalized_text)
    pred_tags = predict_tags(model, tokens, word2idx, char2idx, idx2tag, device)
    entities = decode_entities(tokens, pred_tags)

    return entities  # mantiene "label"


def main() -> None:
    """
    Load a trained NER model and run inference on a single input sentence.
    """
    parser = argparse.ArgumentParser(description="Run inference with the trained NER model.")
    parser.add_argument("--text", type=str, required=True, help="Input sentence for NER inference.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Disable tweet normalization and run the model on the raw text.",
    )
    parser.add_argument(
        "--show-normalized",
        action="store_true",
        help="Print the normalized text that is sent to the tokenizer.",
    )
    args = parser.parse_args()

    model, word2idx, char2idx, idx2tag, device = load_ner_model()

    normalized_text = args.text if args.raw else normalize_tweet_for_ner(args.text)
    tokens = tokenize_for_ner(normalized_text)
    pred_tags = predict_tags(model, tokens, word2idx, char2idx, idx2tag, device)
    entities = decode_entities(tokens, pred_tags)

    if args.show_normalized:
        print("NORMALIZED TEXT")
        print(normalized_text)
        print()

    print("TOKENS AND TAGS")
    for token, tag in zip(tokens, pred_tags):
        print(f"{token:20s} {tag}")

    print("\nENTITIES")
    for entity in entities:
        print(entity)


if __name__ == "__main__":
    main()
