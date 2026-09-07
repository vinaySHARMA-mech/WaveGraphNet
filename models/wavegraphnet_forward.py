# models/wavegraphnet_forward.py
"""
Forward consistency branch: `ForwardGNN`.

Maps a candidate damage coordinate to the path-wise energy-deviation
pattern it would produce, and is used only at inference, inside test-time
refinement (see evaluate_refinement.py).

Each measured path is represented by BOTH of its directed edges, (i,j) and
(j,i). The raw forward edge features are not invariant to exchanging the
two transducers -- the direction vector changes sign and the two endpoint
distances swap -- so invariance is imposed at the encoding stage instead:
the two directional feature vectors are embedded with the same shared
encoder and averaged,

    h_ij = h_ji = 0.5 * ( phi(e_ij) + phi(e_ji) )

Averaging is invariant to the order of its inputs, so the edge embedding is
invariant to node interchange by construction rather than learned from
data. The subsequent message passing preserves that property: each edge
update depends on its two endpoints only through their sum. A runtime
assertion in `forward()` checks that h_(i,j) really does equal h_(j,i)
after message passing.
"""
import torch
import torch.nn as nn


class ForwardLayer(nn.Module):
    """
    One message-passing step over the forward branch's bidirectional graph
    (both (i,j) and (j,i) present -- 132 edges, not 66).

    Why h_(i,j) stays equal to h_(j,i) if it was equal before: the edge
    update reads its endpoints as h_row + h_col, i.e. h_i+h_j for edge (i,j)
    and h_j+h_i for edge (j,i) -- the same sum through the same weights, so
    the same output. The node update sends each edge's message to its own
    `col` node; because h_(i,j) == h_(j,i), node i receives that value from
    edge (j,i) and node j receives it from edge (i,j), so both endpoints of
    a path are updated symmetrically.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_n, x_e, row, col):
        # x_n: [N, H]   x_e: [E, H]   row, col: [E] node index for each edge
        h_row = x_n[row]
        h_col = x_n[col]
        x_e = x_e + self.edge_mlp(torch.cat([h_row + h_col, x_e], dim=-1))

        msg = x_n.new_zeros(x_n.shape[0], x_n.shape[-1])
        msg.index_add_(0, col, x_e)
        x_n = x_n + self.node_mlp(torch.cat([x_n, msg], dim=-1))
        return x_n, x_e


class ForwardGNN(nn.Module):
    """
    Forward branch: candidate coordinate -> path-wise energy deviation.

    The graph carries both directed edges of every measured path, (i,j) and
    (j,i) -- 132 edges over the 66 measured paths. The raw geometric feature
    is therefore computed TWICE, once per direction, passed through the SAME
    edge encoder, and averaged:

        h_ij = h_ji = 0.5 * ( edge_encoder(feat_sr) + edge_encoder(feat_rs) )

    Averaging gives the same answer regardless of which direction is treated
    as "first", so the edge embedding is invariant to node interchange by
    construction rather than by learning. Message passing preserves this:
    each edge update sees its two endpoints only through their sum. The
    assertion in `forward()` verifies that h_(i,j) and h_(j,i) are still
    equal after the interaction layers, and the decoder therefore only needs
    to be applied to one direction per path.
    """
    def __init__(
        self,
        raw_node_feat_dim: int = 2,
        physical_edge_feat_dim: int = 6,
        hidden_dim: int = 128,
        num_propagation_pairs: int = 66,
        num_interaction_layers: int = 3,
    ):
        super().__init__()
        self.num_propagation_pairs = num_propagation_pairs
        self.node_encoder = nn.Linear(raw_node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(physical_edge_feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.processor = nn.ModuleList(
            [ForwardLayer(hidden_dim) for _ in range(num_interaction_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, graph, damage_locs: torch.Tensor, e_pristine: torch.Tensor):
        batch_size = damage_locs.shape[0]
        eps = 1e-8

        row, col = graph.edge_index      # [132 per graph] -- both directions present
        keep = row < col                 # pick one row per pair, arbitrarily, to build the geometry from
        s_idx = row[keep]                # [batch*66] node index of one endpoint
        r_idx = col[keep]                # [batch*66] node index of the other endpoint

        s_coords = graph.x[s_idx]
        r_coords = graph.x[r_idx]
        damage_locs = damage_locs[graph.batch[s_idx]]   # broadcast candidate location to each edge

        # ---- view 1: treat s as the first endpoint, r as the second ----
        vec_sr = s_coords - r_coords
        edge_length = torch.sqrt(vec_sr.pow(2).sum(dim=-1, keepdim=True) + eps)  # same in both views
        t = torch.sum((damage_locs - s_coords) * (r_coords - s_coords), dim=-1, keepdim=True)
        t = (t / (r_coords - s_coords).pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)).clamp(0, 1)
        proj = s_coords + t * (r_coords - s_coords)
        dist_from_damage = torch.sqrt((damage_locs - proj).pow(2).sum(dim=-1, keepdim=True) + eps)  # same in both views
        dist_s = torch.sqrt((s_coords - damage_locs).pow(2).sum(dim=-1, keepdim=True) + eps)
        dist_r = torch.sqrt((r_coords - damage_locs).pow(2).sum(dim=-1, keepdim=True) + eps)
        feat_sr = torch.cat([vec_sr, edge_length, dist_from_damage, dist_s, dist_r], dim=1)

        # ---- view 2: the SAME physical path, treating r as the first endpoint, s as the second ----
        vec_rs = r_coords - s_coords   # = -vec_sr
        feat_rs = torch.cat([vec_rs, edge_length, dist_from_damage, dist_r, dist_s], dim=1)

        e_pristine = e_pristine.repeat(batch_size).unsqueeze(-1)   # [batch*66, 1] -- one value per path, same in both views
        h_sr = self.edge_encoder(torch.cat([feat_sr, e_pristine], dim=1))   # embedding from view 1
        h_rs = self.edge_encoder(torch.cat([feat_rs, e_pristine], dim=1))   # embedding from view 2
        h_pair = 0.5 * (h_sr + h_rs)   # average -- same result no matter which view is "first"

        h_n = self.node_encoder(graph.x)
        row_full = torch.cat([s_idx, r_idx])   # edge (s,r) for the first half, edge (r,s) for the second half
        col_full = torch.cat([r_idx, s_idx])
        h_e = torch.cat([h_pair, h_pair], dim=0)   # both halves start from the identical averaged embedding

        for layer in self.processor:
            h_n, h_e = layer(h_n, h_e, row_full, col_full)

        n = s_idx.shape[0]
        h_first_half, h_second_half = h_e[:n], h_e[n:]
        assert torch.allclose(h_first_half, h_second_half, atol=1e-5), \
            "h_(i,j) != h_(j,i) after message passing -- the bidirectional graph broke the invariance it was built to preserve"

        # Both halves are identical (just checked), so decoding either one gives the full answer.
        predicted_delta_e = self.decoder(h_first_half).squeeze(-1)   # [batch*66]
        return predicted_delta_e.view(batch_size, self.num_propagation_pairs)
