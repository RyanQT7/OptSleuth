import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class ImprovedDecoder(nn.Module):
    """
    MLP Decoder to reconstruct features from hidden representations.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        layers = []

        # Construct decoder network
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Activation and Norm for hidden layers
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.LayerNorm(dims[i + 1]))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class DeviceGAT(nn.Module):
    """
    GAT-based network for spatial correlation modeling.
    Graph Structure:
        - Local nodes: Fully connected.
        - Local <-> Remote: Bi-directional connection.
        - Remote nodes: No direct connection between them.
    """

    def __init__(self, input_feature_dim, mapped_dim=32,
                 hidden_dim=64, gat_heads=4, dropout=0.1,
                 decoder_hidden_dims=[128, 64]):
        super().__init__()

        self.input_feature_dim = input_feature_dim
        self.mapped_dim = mapped_dim
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads

        # Single Layer GAT
        self.gat = GATConv(
            in_channels=mapped_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=True,  # Output dim = hidden_dim * heads
            dropout=dropout
        )

        # Output dimension after GAT
        self.output_dim = hidden_dim * gat_heads

        # LayerNorm + Dropout for stability
        self.norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

        # Decoders for local and remote nodes
        self.decoder_local = ImprovedDecoder(
            input_dim=self.output_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_feature_dim,
            dropout=dropout
        )
        self.decoder_remote = ImprovedDecoder(
            input_dim=self.output_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_feature_dim,
            dropout=dropout
        )

    def build_graph(self, local_idx, remote_idx, device):
        """
        Constructs the edge index for the graph.
        """
        edges = []
        L = len(local_idx)

        # 1. Local Fully Connected
        for i in range(L):
            for j in range(L):
                if i != j:
                    edges.append([local_idx[i], local_idx[j]])

        # 2. Local <-> Remote (Pairwise)
        for i in range(L):
            edges.append([local_idx[i], remote_idx[i]])
            edges.append([remote_idx[i], local_idx[i]])

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index

    def forward(self, node_features, node_types):
        """
        Forward pass.
        Args:
            node_features: [2*L, mapped_dim] - Embeddings from ADA
            node_types:    [2*L]             - 1=local, 0=remote
        """
        device = node_features.device
        num_nodes = node_features.size(0)

        # Identify indices
        local_idx = torch.where(node_types == 1)[0]
        remote_idx = torch.where(node_types == 0)[0]

        # Build Graph
        edge_index = self.build_graph(local_idx, remote_idx, device)

        # GAT Convolution
        gat_out, (ei, ew) = self.gat(
            node_features,
            edge_index,
            return_attention_weights=True
        )

        # Normalization
        h = self.norm(gat_out)
        h = self.dropout(h)

        # Reconstruction
        reconstructed = torch.zeros(
            (num_nodes, self.input_feature_dim), device=device)

        reconstructed[local_idx] = self.decoder_local(h[local_idx])
        reconstructed[remote_idx] = self.decoder_remote(h[remote_idx])

        return reconstructed, ei, ew