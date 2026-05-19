from __future__ import annotations

import torch
from torch import Tensor, nn


class SABiLSTM(nn.Module):
    """
    BiLSTM model for text classification.

    The model receives token ids for a batch of texts and outputs
    logits over the sentiment/damage classes.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the model.

        Args:
            vocab_size: Size of the input vocabulary.
            embedding_dim: Dimension of the word embeddings.
            hidden_dim: Hidden size of the LSTM.
            num_classes: Number of output classes.
            pad_idx: Padding token index.
            dropout: Dropout probability.
        """
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len].

        Returns:
            Tensor: Logits of shape [batch_size, num_classes].
        """
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)

        mask = (input_ids != self.pad_idx).unsqueeze(-1).float()
        masked_outputs = outputs * mask

        summed = masked_outputs.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        features = summed / lengths

        features = self.dropout(features)
        logits = self.classifier(features)

        return logits