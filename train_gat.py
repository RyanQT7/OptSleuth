import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom Imports
from models.domain_adapter import UnifiedDomainAdapter
from models.gat import DeviceGAT
from data.switch_dataset import SwitchDataset
from utils.rcl import root_cause_localization
from utils.pattern_matcher import PatternMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SwitchTrainer:
    """
    Trainer for a single switch using a frozen Pretrained ADA and a trainable GAT.
    Includes learnable alpha parameter for weight balancing.
    """
    def __init__(self, switch_name, config, pretrained_ada_path, device):
        self.switch_name = switch_name
        self.config = config
        self.device = device

        # 1. Load Pretrained ADA (Frozen)
        logger.info(f"[{switch_name}] Loading ADA model from {pretrained_ada_path}")
        checkpoint = torch.load(pretrained_ada_path, map_location=device)
        ada_config = checkpoint['model_config']
        
        self.domain_adapter = UnifiedDomainAdapter(
            input_feature_dim=ada_config['input_feature_dim'],
            transformer_d_model=config['transformer_d_model'],
            transformer_nhead=config['transformer_nhead'],
            transformer_num_layers=config['transformer_num_layers'],
            transformer_out_dim=config['transformer_out_dim'],
            num_domains=ada_config['num_domains'],
            mapped_dim=ada_config['mapped_dim'],
            reconstruction_dim=ada_config['input_feature_dim']
        ).to(device)
        
        self.domain_adapter.load_state_dict(checkpoint['ada_state_dict'])
        
        # Freeze ADA
        for param in self.domain_adapter.parameters():
            param.requires_grad = False
        self.domain_adapter.eval()

        # 2. Initialize GAT (Trainable)
        self.gat_model = DeviceGAT(
            input_feature_dim=config['input_feature_dim'],
            mapped_dim=config['transformer_out_dim'],
            hidden_dim=config['gat_hidden_dim'],
            gat_heads=config['gat_heads'],
            dropout=config['dropout'],
            decoder_hidden_dims=config['decoder_hidden_dims']
        ).to(device)

        # 3. Initialize Learnable Parameter Alpha
        # Use a tensor value (e.g. 0.5) which will be passed through sigmoid later
        self.alpha = nn.Parameter(torch.tensor(0.5, device=device))

        # 4. Optimizer: Include both GAT parameters and Alpha
        trainable_parameters = list(self.gat_model.parameters()) + [self.alpha]
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config['lr_gat'],
            weight_decay=config['weight_decay']
        )

    def train_epoch(self, dataloader, pattern_matcher=None):
        self.gat_model.train()
        total_loss, total_mse = 0, 0
        total_alpha = 0
        samples = 0

        for batch in dataloader:
            # Unpack (batch_size=1)
            window_data, _, node_types = batch
            window_data = window_data.to(self.device).squeeze(0) # [Num_Nodes, T, F]
            node_types = node_types.to(self.device).squeeze(0)
            
            self.optimizer.zero_grad()
            samples += 1

            # 1. Feature Extraction (ADA) - Frozen
            with torch.no_grad():
                node_features = self.domain_adapter.feature_extractor(window_data)
                # node_features shape: [Num_Nodes, transformer_out_dim]

            # 2. Pattern Matching (Optional) - Using Features
            pattern_scores = None
            if pattern_matcher:
                with torch.no_grad():
                    # CHANGE: Use match_patterns_from_features as per reference logic
                    pattern_scores, _ = pattern_matcher.match_patterns_from_features(node_features)
                    pattern_scores = pattern_scores.to(self.device) # [Num_Nodes]

            # 3. GAT Forward
            reconstructed, edge_index, edge_weights = self.gat_model(node_features, node_types)

            # 4. Loss Calculation (Target is last timestamp)
            target = window_data[:, -1, :] # [Num_Nodes, F]
            
            # Calculate MSE per node
            mse_per_node = F.mse_loss(reconstructed, target, reduction='none').mean(dim=1) # [Num_Nodes]

            # 5. Root Cause Localization Weight (W1)
            with torch.no_grad():
                # Placeholder labels (all 1 for unsupervised weighting calculation)
                dummy_labels = torch.ones_like(mse_per_node, dtype=torch.long)
                # Ensure root_cause_localization returns (final_score, score_pr)
                _, score_pr = root_cause_localization(
                    mse_per_node.detach(), 
                    dummy_labels, 
                    edge_index, 
                    edge_weights, 
                    device=self.device
                )
                w1 = score_pr.detach()

            # 6. Weight Combination with Learnable Alpha
            alpha_val = torch.sigmoid(self.alpha)
            
            if pattern_scores is not None:
                w2 = pattern_scores.detach()
                # Dynamic fusion: alpha * W1 + (1 - alpha) * W2
                w = alpha_val * w1 + (1 - alpha_val) * w2
            else:
                # Fallback if no pattern matching
                w = score_pr.detach()

            # 7. Final Weighted Loss
            loss = (w * mse_per_node).mean()

            # Backward
            loss.backward()
            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'], 
                    self.config['grad_clip']
                )
            self.optimizer.step()

            total_loss += loss.item()
            total_mse += mse_per_node.mean().item()
            total_alpha += alpha_val.item()

        return total_loss / samples, total_mse / samples, total_alpha / samples

    def save(self, save_dir):
        path = os.path.join(save_dir, self.switch_name, "model")
        os.makedirs(path, exist_ok=True)
        
        # Save GAT state dict, Alpha, and Config
        save_dict = {
            'gat_state_dict': self.gat_model.state_dict(),
            'alpha': self.alpha.detach().cpu().item(),
            'alpha_sigmoid': torch.sigmoid(self.alpha).detach().cpu().item()
        }
        
        torch.save(save_dict, os.path.join(path, 'gat_model.pt'))
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

# ... (Previous helper functions like load_templates remain unchanged) ...

def main():
    parser = argparse.ArgumentParser(description="GAT Training for Optical Networks")
    parser.add_argument('--data_root', type=str, default='./pod17_normalized')
    parser.add_argument('--ada_path', type=str, required=True, help='Path to pretrained UnifiedADA .pt file')
    parser.add_argument('--template_dir', type=str, default='./failure_templates')
    parser.add_argument('--save_root', type=str, default='./model_results/GAT_models')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Config
    config = {
        'input_feature_dim': 14,
        'transformer_d_model': 64,
        'transformer_nhead': 8,
        'transformer_num_layers': 3,
        'transformer_out_dim': 64,
        'mapped_dim': 32,
        'gat_hidden_dim': 64,
        'gat_heads': 4,
        'decoder_hidden_dims': [128, 64],
        'dropout': 0.1,
        'window_len': 288,
        'batch_size': args.batch_size,
        'lr_gat': args.lr,
        'gat_epochs': args.epochs,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'save_dir': args.save_root
    }

    # 1. Load Templates
    # Note: Ensure your pattern_matcher.py supports `build_template_library` and `match_patterns_from_features`
    pattern_templates = [] 
    # Logic to load templates into pattern_templates list...
    # (Assuming load_templates function is defined as before)
    # pattern_templates = load_templates(args.template_dir)
    
    # 2. Iterate Switches
    if not os.path.exists(args.data_root):
        logger.error(f"Data root {args.data_root} does not exist.")
        return

    switch_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    global_vendor_map = {}
    
    for switch in tqdm(switch_dirs, desc="Switches"):
        switch_path = os.path.join(args.data_root, switch)
        
        try:
            # Dataset
            dataset = SwitchDataset(
                switch_path, global_vendor_map, 
                window_len=config['window_len'], feature_dim=config['input_feature_dim']
            )
            # Batch size must be 1 for this graph logic
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

            # Trainer
            trainer = SwitchTrainer(switch, config, args.ada_path, device)
            
            # Pattern Matcher Init
            matcher = None
            if pattern_templates:
                # Use the modified PatternMatcher from your project
                matcher = PatternMatcher(trainer.domain_adapter, input_feature_dim=config['input_feature_dim'], device=device)
                matcher.build_template_library(pattern_templates)

            # Train Loop
            for epoch in range(args.epochs):
                loss, mse, alpha = trainer.train_epoch(dataloader, matcher)
                if (epoch + 1) % 5 == 0:
                    logger.info(f"[{switch}] Ep {epoch+1}: Loss={loss:.4f}, MSE={mse:.4f}, Alpha={alpha:.4f}")

            # Save
            trainer.save(args.save_root)

        except Exception as e:
            logger.error(f"Failed to train {switch}: {e}")
            import traceback
            traceback.print_exc()

    # Save Metadata
    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, 'global_info.json'), 'w') as f:
        json.dump({'vendor_map': global_vendor_map, 'config': config}, f, indent=2)

if __name__ == "__main__":
    main()