from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CharCNNBiLSTMNER(nn.Module):
    """
    NER model trained from scratch with word embeddings, character CNN features, and BiLSTM context.
    """

    def __init__(
        self,
        vocab_size: int,
        char_vocab_size: int,
        embedding_dim: int,
        char_embedding_dim: int,
        char_num_filters: int,
        hidden_dim: int,
        num_tags: int,
        pad_idx: int,
        char_pad_idx: int,
        unk_idx: int,
        dropout: float = 0.35,
        word_dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.char_embedding = nn.Embedding(
            char_vocab_size,
            char_embedding_dim,
            padding_idx=char_pad_idx,
        )
        self.char_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=char_embedding_dim,
                    out_channels=char_num_filters,
                    kernel_size=kernel_size,
                    padding=0,
                )
                for kernel_size in (2, 3, 4)
            ]
        )
        char_feature_dim = char_num_filters * len(self.char_convs)

        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + char_feature_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)

    def apply_word_dropout(self, input_ids: Tensor) -> Tensor:
        if not self.training or self.word_dropout <= 0:
            return input_ids

        keep_mask = torch.rand_like(input_ids, dtype=torch.float) > self.word_dropout
        special_mask = input_ids <= self.unk_idx
        replacement = torch.full_like(input_ids, self.unk_idx)
        return torch.where(keep_mask | special_mask, input_ids, replacement)

    def encode_chars(self, char_ids: Tensor) -> Tensor:
        batch_size, seq_len, word_len = char_ids.shape
        flat_char_ids = char_ids.view(batch_size * seq_len, word_len)
        char_emb = self.char_embedding(flat_char_ids)
        char_emb = char_emb.transpose(1, 2)

        conv_outputs = []
        for conv in self.char_convs:
            conv_out = torch.relu(conv(char_emb))
            pooled = torch.max(conv_out, dim=-1).values
            conv_outputs.append(pooled)

        char_features = torch.cat(conv_outputs, dim=-1)
        return char_features.view(batch_size, seq_len, -1)

    def forward(self, input_ids: Tensor, char_ids: Tensor) -> Tensor:
        input_ids = self.apply_word_dropout(input_ids)
        word_features = self.word_embedding(input_ids)
        char_features = self.encode_chars(char_ids)

        features = torch.cat([word_features, char_features], dim=-1)
        features = self.input_dropout(features)
        output, _ = self.lstm(features)
        output = self.output_dropout(output)
        return self.classifier(output)
