# models/baselines/lstm.py
"""Bidirectional-LSTM baselines: raw per-frequency input, and a ΔE-input fairness variant."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_baseline(nn.Module):
    """
    Raw-input variant: one BiLSTM run per sensor-path over its own 256-length
    frequency sequence, then attention-pooled across the 66 resulting
    per-path embeddings.

    Uses `LayerNorm`, not `BatchNorm1d`, in the decoder head -- a real bug
    fix, not a hyperparameter tune. `BatchNorm1d` normalizes across the
    batch dimension; with ~66 training samples and batch size 8, its running
    statistics come from very few, noisy updates, which is a plausible
    concrete cause of this model's previously-observed instability (FPR up
    to 94% in an earlier version). `LayerNorm` normalizes each sample
    independently with no batch-composition dependence at all.
    """

    def __init__(self, num_freqs: int, feature_dim_per_freq: int, num_sensor_pairs: int,
                lstm_hidden_dim: int = 256, num_lstm_layers: int = 2, attention_dim: int = 256,
                decoder_hidden_dim: int = 256, output_dim: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.num_sensor_pairs = num_sensor_pairs
        self.num_freqs = num_freqs
        self.feature_dim_per_freq = feature_dim_per_freq

        self.lstm_encoder = nn.LSTM(
            input_size=feature_dim_per_freq, hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
        )
        lstm_output_dim = lstm_hidden_dim * 2
        self.attention_w = nn.Linear(lstm_output_dim, attention_dim)
        self.attention_v = nn.Linear(attention_dim, 1, bias=False)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, decoder_hidden_dim), nn.ReLU(),
            nn.LayerNorm(decoder_hidden_dim), nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2), nn.ReLU(),
            nn.LayerNorm(decoder_hidden_dim // 2), nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim // 2, output_dim),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        batch_size = pair_features.shape[0]
        lstm_input = pair_features.view(-1, self.num_freqs, self.feature_dim_per_freq)
        _, (h_n, _) = self.lstm_encoder(lstm_input)
        lstm_outputs = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        pair_embeddings = lstm_outputs.view(batch_size, self.num_sensor_pairs, -1)

        attention_scores = self.attention_v(torch.tanh(self.attention_w(pair_embeddings)))
        attention_weights = F.softmax(attention_scores, dim=1)
        graph_embedding = (attention_weights * pair_embeddings).sum(dim=1)
        return self.decoder_mlp(graph_embedding)


class LSTM_baseline_deltae(nn.Module):
    """
    ΔE-input fairness ablation: with only 1 scalar per path (no frequency
    sequence left), the per-pair sub-sequence structure `LSTM_baseline` uses
    is vacuous. Instead, the 66 ΔE values themselves become ONE sequence (66
    timesteps, 1 feature each), through a single BiLSTM, with the same
    attention-pooling + decoder machinery now attending over timesteps of
    that one sequence.
    """

    def __init__(self, num_paths: int = 66, lstm_hidden_dim: int = 256, num_lstm_layers: int = 2,
                attention_dim: int = 256, decoder_hidden_dim: int = 256, output_dim: int = 2,
                dropout_rate: float = 0.2):
        super().__init__()
        self.num_paths = num_paths
        self.lstm_encoder = nn.LSTM(
            input_size=1, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
        )
        lstm_output_dim = lstm_hidden_dim * 2
        self.attention_w = nn.Linear(lstm_output_dim, attention_dim)
        self.attention_v = nn.Linear(attention_dim, 1, bias=False)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, decoder_hidden_dim), nn.ReLU(),
            nn.LayerNorm(decoder_hidden_dim), nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2), nn.ReLU(),
            nn.LayerNorm(decoder_hidden_dim // 2), nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim // 2, output_dim),
        )

    def forward(self, delta_e_seq: torch.Tensor) -> torch.Tensor:
        # delta_e_seq: [B, 66, 1]
        timestep_outputs, _ = self.lstm_encoder(delta_e_seq)   # [B, 66, 2*hidden]
        attention_scores = self.attention_v(torch.tanh(self.attention_w(timestep_outputs)))
        attention_weights = F.softmax(attention_scores, dim=1)
        sequence_embedding = (attention_weights * timestep_outputs).sum(dim=1)
        return self.decoder_mlp(sequence_embedding)
