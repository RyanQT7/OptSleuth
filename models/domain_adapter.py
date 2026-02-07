import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Dict

class GradientReversalLayer(torch.autograd.Function):
    """
    Implements a Gradient Reversal Layer (GRL).
    During the forward pass, it acts as an identity transform.
    During the backward pass, it reverses the gradient by multiplying it by a negative scalar alpha.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x) 

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        output = -ctx.alpha * grad_output
        return output, None 


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class EnhancedTransformerEncoder(nn.Module):
    """
    An enhanced Transformer Encoder designed to extract temporal features from optical device time-series data.
    It incorporates a learnable CLS token to aggregate global sequence information.
    """

    def __init__(self, d_model=64, nhead=8, num_layers=3, dropout=0.1, feature_dim=6, out_dim=64):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim

        self.input_projection = nn.Linear(feature_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, out_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [num_devices, T, feature_dim]
        Returns:
            final_features: Aggregated features of shape [num_devices, out_dim]
        """
        num_devices, T, _ = x.shape

        x_proj = self.input_projection(x) 

        cls = self.cls_token.expand(num_devices, -1, -1) 
        x_with_cls = torch.cat([cls, x_proj], dim=1) 

        x_encoded = self.positional_encoding(x_with_cls)

        # Transpose to [seq_len, batch_size, d_model] for Transformer
        x_trans = x_encoded.transpose(0, 1) 

        encoded = self.transformer(x_trans) 

        # Transpose back to [batch_size, seq_len, d_model]
        encoded = encoded.transpose(0, 1) 

        # Aggregate via CLS token (index 0)
        temporal_features = encoded[:, 0, :] 

        final_features = self.output_projection(temporal_features)

        return final_features


class UnifiedDomainAdapter(nn.Module):
    """
    Unified Domain Adaptation module that combines feature extraction,
    domain adversarial training, and signal reconstruction tasks.
    """

    def __init__(self, input_feature_dim=6, transformer_d_model=64, transformer_nhead=8,
                 transformer_num_layers=3, transformer_out_dim=64,
                 num_domains=3, mapped_dim=32, reconstruction_dim=6):
        super().__init__()
        self.mapped_dim = mapped_dim
        self.reconstruction_dim = reconstruction_dim

        self.feature_extractor = EnhancedTransformerEncoder(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            feature_dim=input_feature_dim,
            out_dim=transformer_out_dim
        )

        feature_dim = transformer_out_dim

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains)
        )

        self.reconstruction_network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, reconstruction_dim) 
        )

    def forward(self, x, domain_labels, node_types, alpha=1.0, return_reconstructed=False):
        """
        Forward pass for the unified model.

        Args:
            x: Input tensor [num_devices, T, input_feature_dim]
            domain_labels: Vendor labels [num_devices]
            node_types: Node type indicators (0=remote, 1=local) [num_devices]
            alpha: GRL scaling factor
            return_reconstructed: Boolean to return reconstructed values (for testing)

        Returns:
            Tuple containing base features, domain loss, and reconstruction loss.
            If return_reconstructed is True, also returns the predicted values.
        """
        device = x.device
        num_devices = x.size(0)
        T = x.size(1)

        base_features = self.feature_extractor(x) 

        reversed_features = GradientReversalLayer.apply(base_features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        domain_loss = F.cross_entropy(domain_pred, domain_labels)

        reconstruction_loss, reconstructed_points = self.compute_reconstruction_loss(
            x, base_features, T, return_reconstructed=True
        )

        if return_reconstructed:
            return base_features, domain_loss, reconstruction_loss, reconstructed_points
        else:
            return base_features, domain_loss, reconstruction_loss

    def compute_reconstruction_loss(self, x, base_features, seq_len, return_reconstructed=False):
        """
        Computes the reconstruction loss by predicting the last time point of the window.

        Args:
            x: Original input [num_devices, T, feature_dim]
            base_features: Extracted features [num_devices, feature_dim]
            seq_len: Sequence length T
            return_reconstructed: Boolean to return predictions

        Returns:
            MSE loss value, and optionally the predicted tensor.
        """
        last_timepoint_true = x[:, -1, :] 

        last_timepoint_pred = self.reconstruction_network(base_features) 

        reconstruction_loss = F.mse_loss(last_timepoint_pred, last_timepoint_true)

        if return_reconstructed:
            return reconstruction_loss, last_timepoint_pred
        else:
            return reconstruction_loss