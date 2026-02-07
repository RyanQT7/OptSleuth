import os
import argparse
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.optical_dataset import OpticalDataset
from models.domain_adapter import UnifiedDomainAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Domain Adaptation Model for Optical Networks")
    
    # Data & Paths
    parser.add_argument('--data_root', type=str, default='./processed_dataset', help='Path to the normalized dataset root')
    parser.add_argument('--save_dir', type=str, default='./experiments/checkpoints', help='Directory to save model checkpoints')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # Model Architecture
    parser.add_argument('--input_dim', type=int, default=14, help='Input feature dimension per timestamp')
    parser.add_argument('--mapped_dim', type=int, default=64, help='Dimension of the mapped feature space')
    parser.add_argument('--lambda_recon', type=float, default=0.5, help='Weight for reconstruction loss')
    parser.add_argument('--num_domains_guess', type=int, default=20, help='Initial guess for number of domains (if dynamic detection fails)')

    return parser.parse_args()

def train(args):
    """
    Main training loop for Unified Domain Adaptation.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device set to: {device}")

    # 1. Initialize Dataset
    logger.info(f"Loading data from {args.data_root}...")
    vendor_id_map = {} # This will be populated by the dataset
    
    try:
        dataset = OpticalDataset(args.data_root, vendor_id_map)
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        return

    if len(dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return

    # Determine actual number of domains found
    num_domains = len(vendor_id_map) if len(vendor_id_map) > 0 else args.num_domains_guess
    logger.info(f"Detected {num_domains} distinct vendor domains.")
    logger.info(f"Vendor Map: {vendor_id_map}")

    # 2. Initialize Model
    model = UnifiedDomainAdapter(
        input_feature_dim=args.input_dim,
        transformer_d_model=64,
        transformer_nhead=8,
        transformer_num_layers=3,
        transformer_out_dim=64,
        num_domains=num_domains,
        mapped_dim=args.mapped_dim,
        reconstruction_dim=args.input_dim
    ).to(device)

    # 3. Setup Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    os.makedirs(args.save_dir, exist_ok=True)
    logger.info("Starting training loop...")

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_domain_loss = 0.0
        total_recon_loss = 0.0
        total_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")
        
        for batch in pbar:
            # Unpack batch
            sequences, domain_labels, node_types = batch
            
            # Move to device
            sequences = sequences.to(device)       # [B, T, F]
            domain_labels = domain_labels.to(device) # [B]
            node_types = node_types.to(device)       # [B]

            current_batch_size = sequences.size(0)
            total_samples += current_batch_size

            # Forward pass
            # Alpha controls the strength of the gradient reversal. 
            # It can be dynamic (schedule) or static. Using static 1.0 here.
            alpha = 1.0 
            
            _, domain_loss, reconstruction_loss = model(
                sequences, domain_labels, node_types, alpha=alpha
            )

            # Combined loss
            total_loss = domain_loss + args.lambda_recon * reconstruction_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress bar
            pbar.set_postfix({
                'd_loss': f'{domain_loss.item():.4f}',
                'r_loss': f'{reconstruction_loss.item():.4f}',
                'ttl': f'{total_loss.item():.4f}'
            })

            # Accumulate metrics
            total_domain_loss += domain_loss.item() * current_batch_size
            total_recon_loss += reconstruction_loss.item() * current_batch_size

        # Epoch Summary
        avg_domain_loss = total_domain_loss / total_samples
        avg_recon_loss = total_recon_loss / total_samples
        logger.info(f"Epoch {epoch + 1} Summary - Domain Loss: {avg_domain_loss:.5f}, Recon Loss: {avg_recon_loss:.5f}")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            checkpoint_name = f"unified_ada_epoch_{epoch + 1}.pt"
            save_path = os.path.join(args.save_dir, checkpoint_name)
            
            torch.save({
                'epoch': epoch + 1,
                'ada_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vendor_id_map': vendor_id_map,
                'model_config': {
                    'input_feature_dim': args.input_dim,
                    'mapped_dim': args.mapped_dim,
                    'num_domains': num_domains
                },
                'losses': {
                    'domain_loss': avg_domain_loss,
                    'recon_loss': avg_recon_loss
                }
            }, save_path)
            logger.info(f"Checkpoint saved: {save_path}")

    logger.info(f"Training complete. Models saved to {args.save_dir}")

if __name__ == "__main__":
    args = parse_args()
    train(args)