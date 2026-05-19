from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "<PAD>"
PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"
MAX_WORD_LEN = 24


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_ner_data_dir() -> Path:
    """
    Return the directory containing the NER CSV files.

    Returns:
        Path: Absolute path to data/NER_data.
    """
    return get_project_root() / "data" / "NER_data"


def load_ner_csv(split: str) -> pd.DataFrame:
    """
    Load a NER CSV split.

    Args:
        split: Dataset split name. Expected values are "train", "dev", or "test".

    Returns:
        pd.DataFrame: Loaded CSV as a pandas DataFrame.
    """
    path = get_ner_data_dir() / f"{split}.csv"
    return pd.read_csv(path)


def build_sentences(df: pd.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    """
    Convert a token-level DataFrame into sentence-level token and tag sequences.

    The input DataFrame is expected to contain at least:
    - sentence_id
    - token_id
    - token
    - tag

    Args:
        df: Token-level NER DataFrame.

    Returns:
        tuple[list[list[str]], list[list[str]]]:
            A tuple containing:
            - list of tokenized sentences
            - list of tag sequences
    """
    df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)
    grouped = df.groupby("sentence_id", sort=True)

    sentences: list[list[str]] = []
    tags: list[list[str]] = []

    for _, group in grouped:
        sentences.append(group["token"].astype(str).tolist())
        tags.append(group["tag"].astype(str).tolist())

    return sentences, tags


def load_ner_splits() -> tuple[
    list[list[str]],
    list[list[str]],
    list[list[str]],
    list[list[str]],
    list[list[str]],
    list[list[str]],
]:
    """
    Load train, dev, and test splits and convert them into sentence-level sequences.

    Returns:
        tuple[
            list[list[str]],
            list[list[str]],
            list[list[str]],
            list[list[str]],
            list[list[str]],
            list[list[str]],
        ]:
            Train sentences, train tags, dev sentences, dev tags, test sentences, test tags.
    """
    train_df = load_ner_csv("train")
    dev_df = load_ner_csv("dev")
    test_df = load_ner_csv("test")

    train_sentences, train_tags = build_sentences(train_df)
    dev_sentences, dev_tags = build_sentences(dev_df)
    test_sentences, test_tags = build_sentences(test_df)

    return (
        train_sentences,
        train_tags,
        dev_sentences,
        dev_tags,
        test_sentences,
        test_tags,
    )


def build_word_vocab(train_sentences: list[list[str]]) -> dict[str, int]:
    """
    Build a token-to-index vocabulary from the training sentences.

    Args:
        train_sentences: Training token sequences.

    Returns:
        dict[str, int]: Mapping from token string to integer index.
    """
    word2idx: dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for sentence in train_sentences:
        for token in sentence:
            if token not in word2idx:
                word2idx[token] = len(word2idx)

    return word2idx


def build_char_vocab(train_sentences: list[list[str]]) -> dict[str, int]:
    """
    Build a character-to-index vocabulary from the training sentences.

    Args:
        train_sentences: Training token sequences.

    Returns:
        dict[str, int]: Mapping from character string to integer index.
    """
    char2idx: dict[str, int] = {
        PAD_CHAR: 0,
        UNK_CHAR: 1,
    }

    for sentence in train_sentences:
        for token in sentence:
            for char in token:
                if char not in char2idx:
                    char2idx[char] = len(char2idx)

    return char2idx


def build_tag_vocab(train_tags: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Build tag-to-index and index-to-tag mappings from the training tags.

    Args:
        train_tags: Training tag sequences.

    Returns:
        tuple[dict[str, int], dict[int, str]]:
            - tag2idx mapping
            - idx2tag mapping
    """
    unique_tags = sorted(set(tag for seq in train_tags for tag in seq))
    tag2idx: dict[str, int] = {PAD_TAG: 0}

    for tag in unique_tags:
        tag2idx[tag] = len(tag2idx)

    idx2tag: dict[int, str] = {idx: tag for tag, idx in tag2idx.items()}
    return tag2idx, idx2tag


class NERDataset(Dataset[dict[str, object]]):
    """
    PyTorch dataset for token-level NER sequences.

    Each item corresponds to one sentence and returns:
    - original tokens
    - input token ids
    - label ids
    - sequence length
    """

    def __init__(
        self,
        sentences: list[list[str]],
        tags: list[list[str]],
        word2idx: dict[str, int],
        tag2idx: dict[str, int],
        char2idx: dict[str, int] | None = None,
        max_word_len: int = MAX_WORD_LEN,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            sentences: Tokenized sentences.
            tags: Tag sequences aligned with the sentences.
            word2idx: Token-to-index mapping.
            tag2idx: Tag-to-index mapping.
        """
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.char2idx = char2idx
        self.max_word_len = max_word_len

    def __len__(self) -> int:
        """
        Return the number of sentences in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """
        Return a sentence sample.

        Args:
            idx: Sample index.

        Returns:
            dict[str, object]: Dictionary containing tokens, input_ids, label_ids, and length.
        """
        tokens = self.sentences[idx]
        labels = self.tags[idx]

        input_ids = [self.word2idx.get(token, self.word2idx[UNK_TOKEN]) for token in tokens]
        label_ids = [self.tag2idx[tag] for tag in labels]

        item: dict[str, object] = {
            "tokens": tokens,
            "input_ids": input_ids,
            "label_ids": label_ids,
            "length": len(input_ids),
        }
        if self.char2idx is not None:
            item["char_ids"] = [
                [
                    self.char2idx.get(char, self.char2idx[UNK_CHAR])
                    for char in token[: self.max_word_len]
                ]
                for token in tokens
            ]

        return item


def make_collate_fn(
    word2idx: dict[str, int],
    tag2idx: dict[str, int],
    char2idx: dict[str, int] | None = None,
    max_word_len: int = MAX_WORD_LEN,
) -> Callable[[list[dict[str, object]]], dict[str, object]]:
    """
    Create a collate function for variable-length NER batches.

    Args:
        word2idx: Token-to-index mapping.
        tag2idx: Tag-to-index mapping.

    Returns:
        Callable[[list[dict[str, object]]], dict[str, object]]:
            Collate function for a PyTorch DataLoader.
    """
    pad_token_id = word2idx[PAD_TOKEN]
    pad_tag_id = tag2idx[PAD_TAG]
    pad_char_id = char2idx[PAD_CHAR] if char2idx is not None else 0

    def ner_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
        """
        Pad and batch NER samples.

        Args:
            batch: List of dataset samples.

        Returns:
            dict[str, object]: Batched tokens, input_ids, label_ids, and lengths.
        """
        max_len = max(int(item["length"]) for item in batch)

        input_ids_batch: list[list[int]] = []
        label_ids_batch: list[list[int]] = []
        char_ids_batch: list[list[list[int]]] = []
        lengths: list[int] = []
        tokens_batch: list[list[str]] = []

        for item in batch:
            input_ids = list(item["input_ids"])
            label_ids = list(item["label_ids"])
            tokens = list(item["tokens"])
            length = int(item["length"])

            pad_len = max_len - length

            input_ids_batch.append(input_ids + [pad_token_id] * pad_len)
            label_ids_batch.append(label_ids + [pad_tag_id] * pad_len)
            if char2idx is not None:
                char_ids = list(item.get("char_ids", []))
                padded_char_ids: list[list[int]] = []
                for token_chars in char_ids:
                    chars = list(token_chars)[:max_word_len]
                    char_pad_len = max_word_len - len(chars)
                    padded_char_ids.append(chars + [pad_char_id] * char_pad_len)
                padded_char_ids.extend([[pad_char_id] * max_word_len for _ in range(pad_len)])
                char_ids_batch.append(padded_char_ids)
            lengths.append(length)
            tokens_batch.append(tokens)

        batch_dict: dict[str, object] = {
            "tokens": tokens_batch,
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "label_ids": torch.tensor(label_ids_batch, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }
        if char2idx is not None:
            batch_dict["char_ids"] = torch.tensor(char_ids_batch, dtype=torch.long)

        return batch_dict

    return ner_collate_fn


def describe_tag_distribution(tags: list[list[str]]) -> Counter[str]:
    """
    Count tag frequencies across a list of tag sequences.

    Args:
        tags: List of tag sequences.

    Returns:
        Counter[str]: Frequency counter for tags.
    """
    return Counter(tag for seq in tags for tag in seq)
