# models/wavegraphnet_inverse.py
"""
WaveGraphNet's inverse localization branch: `DeltaEOnlyInverseGNN`.

Maps the measured path-wise energy-deviation vector ΔE ∈ R^66 (plus fixed
sensor geometry) directly to a predicted defect coordinate p_hat ∈ R^2. No
raw per-frequency spectral signal is used — ΔE alone is the input.

Design summary (see the paper's "Inverse localization module" section for
the full equations):
  1. A shared, order-invariant message-passing stack builds a final
     embedding for every sensor node and every one of the 66 measured
     paths (edges), via `SymmetricEdgeInteractionLayer`.
  2. Each path proposes a candidate point on its own sensor-to-sensor
     segment: a SHARED scoring function scores both endpoints, a 2-way
     softmax turns the two scores into a convex-combination weight.
  3. A second softmax, over all 66 paths' learned "importance," combines
     the 66 per-path proposals into one point `p_conv` (confined to the
     convex hull of the sensor positions — see the class docstring below
     for the accepted limitation this implies).
  4. A learned scalar gate interpolates between `p_conv` and the fixed
     out-of-domain "no damage" sentinel, so damaged and undamaged samples
     share one regression target space with no separate classification
     head or loss term.
"""

import torch
import torch.nn as nn


class SymmetricEdgeInteractionLayer(nn.Module):
    """
    One round of message passing over the fixed, undirected 66-path graph
    (each path represented once — canonical order `s_idx[k] < r_idx[k]`,
    no duplicated reverse edges, so there is no "which direction was this
    edge built in" ambiguity at all).

    Edge update combines the two endpoints via `h_s + h_r` — a SUM, hence
    order-invariant: swapping which sensor is called "s" vs "r" for a path
    provably cannot change the result. Node update aggregates every
    incident edge into both of its endpoints identically.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_n, h_e, s_idx, r_idx):
        # h_n: [B, N, H]   h_e: [B, 66, H]   s_idx/r_idx: [66] long
        h_s, h_r = h_n[:, s_idx, :], h_n[:, r_idx, :]
        h_e = h_e + self.edge_mlp(torch.cat([h_s + h_r, h_e], dim=-1))

        B, N, H = h_n.shape
        msg = h_n.new_zeros(B, N, H)
        msg.index_add_(1, s_idx, h_e)
        msg.index_add_(1, r_idx, h_e)
        h_n = h_n + self.node_mlp(torch.cat([h_n, msg], dim=-1))
        return h_n, h_e


class DeltaEOnlyInverseGNN(nn.Module):
    """
    The proposed inverse branch. Input is ΔE (66 scalars) plus each path's
    fixed length — 2 scalars per edge, no spectral descriptor at all.

    Known, accepted limitation: `p_conv` (the pre-gate localization
    estimate) is a convex combination of sensor coordinates, so it is
    confined to the convex hull of the sensor positions, not the full
    plate — the decoder structurally cannot place a prediction in the
    border strip between the sensor hull and the plate edge.

    p_ud default: the sentinel-gating formula is
    `p_hat = damage_prob * p_conv + (1 - damage_prob) * p_ud`, so the
    "damage_prob value at which p_hat crosses the [0,1]^2 boundary" is
    `d* = |p_ud| / (|p_conv| + |p_ud|)`. With the original `p_ud=(-0.001,
    -0.001)` and typical `|p_conv| ~ 0.7` (measured empirically), `d* ~
    0.0014` -- the model has to be almost machine-precision-confident
    "not damaged" before the FPR check reflects that. `(-0.5, -0.5)`
    moves `d*` to ~0.42, close to the semantically correct d*=0.5
    crossover, and was confirmed (retraining 3 seeds on the hardest
    split) to make FPR agree with the model's own damage_prob>0.5
    decision on every seed, where the old default did not on 2 of 3.
    """

    def __init__(self, hidden_dim: int = 256, num_interaction_layers: int = 3,
                p_ud=(-0.5, -0.5)):
        super().__init__()
        self.node_encoder = nn.Linear(2, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            SymmetricEdgeInteractionLayer(hidden_dim) for _ in range(num_interaction_layers)
        ])
        # Shared endpoint-scoring function g -- applied to (h_s, h_e) and
        # (h_r, h_e) with the SAME weights, so it cannot learn an
        # index-order artifact; its only lever is the two endpoints' own
        # learned content.
        self.g = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.damage_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("p_ud", torch.tensor(p_ud, dtype=torch.float32))

    def forward(self, coords, s_idx, r_idx, edge_length, delta_e):
        """
        coords      : [N, 2]   fixed sensor positions (N=12, or fewer for a
                                reduced-sensor split)
        s_idx,r_idx : [66]     long, canonical path endpoints (s_idx[k] < r_idx[k])
        edge_length : [66]     fixed ||coords[r]-coords[s]||
        delta_e     : [B, 66]  measured per-path energy deviation (input)
        returns     : [B, 2]   predicted coordinate (or the p_ud sentinel)
        """
        B = delta_e.shape[0]
        h_n = self.node_encoder(coords).unsqueeze(0).expand(B, -1, -1).clone()
        edge_in = torch.stack([delta_e, edge_length.unsqueeze(0).expand(B, -1)], dim=-1)  # [B,66,2]
        h_e = self.edge_encoder(edge_in)

        for layer in self.layers:
            h_n, h_e = layer(h_n, h_e, s_idx, r_idx)

        # --- per-path candidate point: shared scorer g applied to both endpoints ---
        h_s, h_r = h_n[:, s_idx, :], h_n[:, r_idx, :]
        z_s = self.g(torch.cat([h_s, h_e], dim=-1)).squeeze(-1)   # [B,66]
        z_r = self.g(torch.cat([h_r, h_e], dim=-1)).squeeze(-1)   # [B,66]
        w1, w2 = torch.softmax(torch.stack([z_s, z_r], dim=-1), dim=-1).unbind(-1)

        r_s, r_r = coords[s_idx], coords[r_idx]                    # [66,2]
        q = w1.unsqueeze(-1) * r_s.unsqueeze(0) + w2.unsqueeze(-1) * r_r.unsqueeze(0)  # [B,66,2]

        # --- pool all 66 candidate points via a second softmax over path importance ---
        W_e = torch.softmax(self.edge_weight_mlp(h_e).squeeze(-1), dim=-1)  # [B,66]
        p_conv = (W_e.unsqueeze(-1) * q).sum(dim=1)                          # [B,2]

        # --- damage / no-damage sentinel gate ---
        damage_prob = torch.sigmoid(self.damage_mlp(h_n.mean(dim=1)))        # [B,1]
        p_ud = self.p_ud.unsqueeze(0).expand(B, -1)
        return damage_prob * p_conv + (1 - damage_prob) * p_ud
