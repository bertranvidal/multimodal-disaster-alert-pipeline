from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def get_sa_data_dir() -> Path:
    """
    Return the directory containing the SA CSV files.

    Returns:
        Path: Absolute path to data/SA_data.
    """
    return get_project_root() / "data" / "SA_data"


def load_sa_csv(split: str) -> pd.DataFrame:
    """
    Load an SA CSV split.

    Args:
        split: Dataset split name. Expected values are "train", "dev", or "test".

    Returns:
        pd.DataFrame: Loaded CSV as a pandas DataFrame.
    """
    path = get_sa_data_dir() / f"crisismmd_damage_{split}.csv"
    return pd.read_csv(path)


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize and lightly normalize tweet text.

    Processing steps:
    - lowercase
    - remove URLs
    - remove @mentions
    - remove RT markers
    - convert hashtags "#word" -> "word"
    - remove most punctuation noise
    - normalize whitespace
    - split by whitespace

    Args:
        text: Input text.

    Returns:
        list[str]: Tokenized text.
    """
    text = str(text).lower()

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\brt\b", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    text = text.replace("\u00a0", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"[^a-z0-9\s\-$%']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return [UNK_TOKEN]

    return text.split()


def load_sa_splits() -> tuple[
    list[list[str]],
    list[int],
    list[list[str]],
    list[int],
    list[list[str]],
    list[int],
]:
    """
    Load train, dev, and test splits for SA.

    Returns:
        tuple[
            list[list[str]],
            list[int],
            list[list[str]],
            list[int],
            list[list[str]],
            list[int],
        ]:
            Train texts, train labels, dev texts, dev labels, test texts, test labels.
    """
    train_df = load_sa_csv("train")
    dev_df = load_sa_csv("dev")
    test_df = load_sa_csv("test")

    train_texts = [tokenize_text(text) for text in train_df["tweet_text"].astype(str).tolist()]
    dev_texts = [tokenize_text(text) for text in dev_df["tweet_text"].astype(str).tolist()]
    test_texts = [tokenize_text(text) for text in test_df["tweet_text"].astype(str).tolist()]

    train_labels = train_df["label"].astype(int).tolist()
    dev_labels = dev_df["label"].astype(int).tolist()
    test_labels = test_df["label"].astype(int).tolist()

    return (
        train_texts,
        train_labels,
        dev_texts,
        dev_labels,
        test_texts,
        test_labels,
    )


def build_word_vocab(train_texts: list[list[str]]) -> dict[str, int]:
    """
    Build a token-to-index vocabulary from the training texts.

    Args:
        train_texts: Training token sequences.

    Returns:
        dict[str, int]: Mapping from token string to integer index.
    """
    word2idx: dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for text in train_texts:
        for token in text:
            if token not in word2idx:
                word2idx[token] = len(word2idx)

    return word2idx


class SADataset(Dataset[dict[str, object]]):
    """
    PyTorch dataset for sentiment analysis sequences.

    Each item corresponds to one text and returns:
    - original tokens
    - input token ids
    - label
    - sequence length
    """

    def __init__(
        self,
        texts: list[list[str]],
        labels: list[int],
        word2idx: dict[str, int],
    ) -> None:
        """
        Initialize the dataset.

        Args:
            texts: Tokenized texts.
            labels: Integer labels for each text.
            word2idx: Token-to-index mapping.
        """
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self) -> int:
        """
        Return the number of texts in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """
        Return one SA sample.

        Args:
            idx: Sample index.

        Returns:
            dict[str, object]: Dictionary containing tokens, input_ids, label, and length.
        """
        tokens = self.texts[idx]
        label = self.labels[idx]

        input_ids = [self.word2idx.get(token, self.word2idx[UNK_TOKEN]) for token in tokens]

        return {
            "tokens": tokens,
            "input_ids": input_ids,
            "label": label,
            "length": len(input_ids),
        }


def make_sa_collate_fn(
    word2idx: dict[str, int],
) -> Callable[[list[dict[str, object]]], dict[str, object]]:
    """
    Create a collate function for variable-length SA batches.

    Args:
        word2idx: Token-to-index mapping.

    Returns:
        Callable[[list[dict[str, object]]], dict[str, object]]:
            Collate function for a PyTorch DataLoader.
    """
    pad_token_id = word2idx[PAD_TOKEN]

    def sa_collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
        """
        Pad and batch SA samples.

        Args:
            batch: List of dataset samples.

        Returns:
            dict[str, object]: Batched tokens, input_ids, labels, and lengths.
        """
        max_len = max(int(item["length"]) for item in batch)

        input_ids_batch: list[list[int]] = []
        labels: list[int] = []
        lengths: list[int] = []
        tokens_batch: list[list[str]] = []

        for item in batch:
            input_ids = list(item["input_ids"])
            tokens = list(item["tokens"])
            label = int(item["label"])
            length = int(item["length"])

            pad_len = max_len - length

            input_ids_batch.append(input_ids + [pad_token_id] * pad_len)
            labels.append(label)
            lengths.append(length)
            tokens_batch.append(tokens)

        return {
            "tokens": tokens_batch,
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }

    return sa_collate_fn