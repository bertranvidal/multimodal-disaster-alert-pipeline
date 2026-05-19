from torch.utils.data import DataLoader

from src.sa_bilstm import SABiLSTM
from src.sa_dataset import (
    PAD_TOKEN,
    SADataset,
    build_word_vocab,
    load_sa_splits,
    make_sa_collate_fn,
)


def main() -> None:
    train_texts, train_labels, _, _, _, _ = load_sa_splits()

    word2idx = build_word_vocab(train_texts)
    pad_idx = word2idx[PAD_TOKEN]

    train_dataset = SADataset(train_texts, train_labels, word2idx)
    collate_fn = make_sa_collate_fn(word2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(train_loader))

    model = SABiLSTM(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_classes=3,
        pad_idx=pad_idx,
        dropout=0.3,
    )

    logits = model(batch["input_ids"])

    print("input_ids shape:", batch["input_ids"].shape)
    print("logits shape:", logits.shape)
    print("logits:", logits)


if __name__ == "__main__":
    main()