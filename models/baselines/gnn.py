# models/baselines/gnn.py
"""
GNN-MLP and GAT baselines. Both are the SAME `FlexibleGNN` architecture and
message-passing processor (`GNNProcessor_MLP`, built from `RichEdgeConv`
layers) -- they differ ONLY in the edge encoder:

  - "GNN-MLP" (`encoder_type="simple_mlp"`): a plain MLP over the raw
    per-path edge features.
  - "GAT"     (`encoder_type="attention"`):  an attention-weighted pool
    over the 256 frequency bins before the same MLP-based message passing.

There is no separate GAT *message-passing* processor wired into this
codebase -- if you need a literal GAT-conv processor, it does not exist
here; describe the baseline accordingly (attention is in the edge encoder
only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData

from models.layers import NodeEncoder, GraphDecoder, RichEdgeConv


class EdgeEncoderWithAttention(nn.Module):
    """Attention-weighted pool over per-frequency-bin features -- the "GAT" baseline's edge encoder."""

    def __init__(self, num_freqs, feature_dim_per_freq, static_feat_dim, final_embedding_dim,
                dropout_rate=0.2):
        super().__init__()
        self.static_feat_dim = static_feat_dim
        self.feature_dim_per_freq = feature_dim_per_freq
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim_per_freq, 64), nn.ReLU(), nn.Dropout(dropout_rate))
        self.attention_mlp = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.combiner_mlp = nn.Sequential(
            nn.Linear(64 + static_feat_dim, final_embedding_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(final_embedding_dim, final_embedding_dim),
        )

    def forward(self, x_edges):
        static_feats = x_edges[:, :self.static_feat_dim]
        freq_features = x_edges[:, self.static_feat_dim:].view(
            x_edges.shape[0], -1, self.feature_dim_per_freq)
        processed = self.feature_processor(freq_features)
        attn = F.softmax(self.attention_mlp(processed), dim=1)
        dynamic_embedding = (attn * processed).sum(dim=1)
        return self.combiner_mlp(torch.cat([dynamic_embedding, static_feats], dim=1))


class SimpleEdgeEncoder(nn.Module):
    """Plain MLP edge encoder -- the "GNN-MLP" baseline's edge encoder."""

    def __init__(self, raw_edge_feat_dim, embedding_dim, hidden_dim, num_layers=3, dropout_rate=0.2):
        super().__init__()
        layers = [nn.Linear(raw_edge_feat_dim, hidden_dim), nn.ReLU(),
                 nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                      nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.encoder_mlp = nn.Sequential(*layers)

    def forward(self, x_edges: torch.Tensor) -> torch.Tensor:
        return self.encoder_mlp(x_edges)


class GNNProcessor_MLP(nn.Module):
    """Stacked `RichEdgeConv` message-passing layers -- shared by both GNN-MLP and GAT."""

    def __init__(self, hidden_dim, num_gnn_layers, dropout_rate=0.2):
        super().__init__()
        self.convs = nn.ModuleList([
            RichEdgeConv(node_feat_dim=hidden_dim, edge_feat_dim=hidden_dim,
                        hidden_channels=hidden_dim, out_channels=hidden_dim,
                        dropout_rate=dropout_rate)
            for _ in range(num_gnn_layers)
        ])

    def forward(self, node_embeds, edge_index, edge_embeds):
        x_nodes = node_embeds
        for conv in self.convs:
            x_nodes = conv(x_nodes, edge_index, edge_embeds)
        return x_nodes


class FlexibleGNN(nn.Module):
    def __init__(self, encoder_type: str, processor_type: str, raw_node_feat_dim, raw_edge_feat_dim,
                num_attention_freqs, hidden_dim, num_gnn_proc_layers, gat_attention_heads,
                decoder_mlp_hidden_dim, final_output_dim, decoder_pooling_type,
                num_decoder_mlp_layers, decoder_dropout_rate):
        super().__init__()
        self.node_encoder = NodeEncoder(raw_node_feat_dim, hidden_dim)

        if encoder_type == "attention":
            self.edge_encoder = EdgeEncoderWithAttention(
                num_attention_freqs, 2, 3, hidden_dim, decoder_dropout_rate)
        elif encoder_type == "simple_mlp":
            self.edge_encoder = SimpleEdgeEncoder(
                raw_edge_feat_dim, hidden_dim, hidden_dim * 2, 4, decoder_dropout_rate)
        else:
            raise ValueError(f"Unknown encoder_type: '{encoder_type}'.")

        if processor_type != "mlp":
            raise ValueError(f"Unknown processor_type: '{processor_type}' (only 'mlp' is implemented).")
        self.gnn_processor = GNNProcessor_MLP(hidden_dim, num_gnn_proc_layers, decoder_dropout_rate)

        self.graph_decoder = GraphDecoder(
            hidden_dim, decoder_mlp_hidden_dim, final_output_dim,
            decoder_pooling_type, num_decoder_mlp_layers, decoder_dropout_rate)

    def forward(self, data: PyGData):
        node_emb = self.node_encoder(data.x)
        edge_emb = self.edge_encoder(data.edge_attr)
        node_emb = self.gnn_processor(node_emb, data.edge_index, edge_emb)
        return self.graph_decoder(node_emb, data.batch)
