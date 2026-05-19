from src.sa_dataset import (
    SADataset,
    build_word_vocab,
    load_sa_splits,
    make_sa_collate_fn,
)
from torch.utils.data import DataLoader


def main() -> None:
    train_texts, train_labels, dev_texts, dev_labels, test_texts, test_labels = load_sa_splits()

    print("Train samples:", len(train_texts))
    print("Dev samples:", len(dev_texts))
    print("Test samples:", len(test_texts))

    word2idx = build_word_vocab(train_texts)
    print("Vocab size:", len(word2idx))

    train_dataset = SADataset(train_texts, train_labels, word2idx)
    print("First sample:", train_dataset[0])

    collate_fn = make_sa_collate_fn(word2idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["labels"].shape)
    print("lengths:", batch["lengths"])


if __name__ == "__main__":
    main()