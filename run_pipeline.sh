#!/bin/bash

# ==============================================================================
# Optical Network Anomaly Detection Pipeline Execution Script
# ==============================================================================
# This script executes the complete workflow:
# 1. Data Setup (Move existing data)
# 2. Preprocessing (Normalization)
# 3. Domain Adaptation (ADA) Training
# 4. Graph Attention Network (GAT) Training
# 5. Final Testing & Root Cause Analysis
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---

# Project Root Directories
DATA_ROOT="./data_workspace"
RAW_DATA_DIR="${DATA_ROOT}/raw_data"
PROCESSED_DATA_DIR="${DATA_ROOT}/processed_data"
RESULTS_DIR="./experiment_results"

# Model Checkpoint Directories
ADA_SAVE_DIR="${RESULTS_DIR}/checkpoints/ada"
GAT_SAVE_DIR="${RESULTS_DIR}/checkpoints/gat"
FINAL_OUTPUT_DIR="${RESULTS_DIR}/final_inference"

# Scripts Paths
SCRIPT_PREPROCESS="data/preprocess.py"
SCRIPT_TRAIN_ADA="train_ada.py"
SCRIPT_TRAIN_GAT="train_gat.py"
SCRIPT_TEST_GAT="test_gat.py"

# Hyperparameters
EPOCHS_ADA=50
EPOCHS_GAT=30
BATCH_SIZE=64

# Device Configuration
# Default to "cpu" for compatibility. 
# Change to "cuda" if you have an NVIDIA GPU.
DEVICE="cpu"

# Colors for logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Helper Functions ---

log_info() {
    echo -e "${BLUE}[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
    fi
}

# --- Pipeline Execution ---

echo "=========================================================="
echo "   Starting Optical Network Anomaly Detection Pipeline    "
echo "=========================================================="

# 0. Setup Directories
log_info "Setting up directories..."
ensure_dir "$DATA_ROOT"
ensure_dir "$RESULTS_DIR"
ensure_dir "$ADA_SAVE_DIR"
ensure_dir "$GAT_SAVE_DIR"
ensure_dir "$FINAL_OUTPUT_DIR"

# 1. Data Setup (Skipping Generation)
log_info "Step 1: Checking for existing data..."

# Logic: If user placed data in 'mock_dataset', move it to the workspace raw dir.
if [ -d "mock_dataset" ]; then
    log_info "Found 'mock_dataset' folder, moving to workspace..."
    # Clean target if exists to avoid conflicts
    if [ -d "$RAW_DATA_DIR" ]; then
        rm -rf "$RAW_DATA_DIR"
    fi
    mv "mock_dataset" "$RAW_DATA_DIR"
    log_success "Data moved to $RAW_DATA_DIR"
fi

# Logic: Verify that data actually exists where preprocessing expects it
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo ""
    log_error "Data Missing!"
    echo "Please place your CSV files in a folder named 'mock_dataset' in the root directory,"
    echo "OR directly into '$RAW_DATA_DIR'."
    exit 1
fi

# 2. Preprocessing
log_info "Step 2: Preprocessing Data (Normalization)..."

python "$SCRIPT_PREPROCESS"

# Move output if necessary (Based on preprocess.py default output "./processed_dataset")
if [ -d "./processed_dataset" ]; then
    log_info "Moving processed data to workspace..."
    if [ -d "$PROCESSED_DATA_DIR" ]; then
        rm -rf "$PROCESSED_DATA_DIR"
    fi
    mv "./processed_dataset" "$PROCESSED_DATA_DIR"
fi
log_success "Preprocessing complete. Data at: $PROCESSED_DATA_DIR"


# 3. ADA Training
log_info "Step 3: Training Unified Domain Adapter (ADA) on $DEVICE..."
python "$SCRIPT_TRAIN_ADA" \
    --data_root "$PROCESSED_DATA_DIR" \
    --save_dir "$ADA_SAVE_DIR" \
    --epochs "$EPOCHS_ADA" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

ADA_BEST_MODEL="${ADA_SAVE_DIR}/unified_ada_epoch_${EPOCHS_ADA}.pt"
if [ ! -f "$ADA_BEST_MODEL" ]; then
    log_error "ADA Model not found at $ADA_BEST_MODEL"
    exit 1
fi
log_success "ADA Training complete."


# 4. GAT Training
log_info "Step 4: Training Graph Attention Network (GAT) on $DEVICE..."
python "$SCRIPT_TRAIN_GAT" \
    --data_root "$PROCESSED_DATA_DIR" \
    --ada_path "$ADA_BEST_MODEL" \
    --save_root "$GAT_SAVE_DIR" \
    --epochs "$EPOCHS_GAT" \
    --device "$DEVICE" \
    --batch_size 1

log_success "GAT Training complete."


# 5. GAT Testing (Inference)
log_info "Step 5: Running Final Testing & Anomaly Detection..."
python "$SCRIPT_TEST_GAT" \
    --data_root "$PROCESSED_DATA_DIR" \
    --model_dir "$GAT_SAVE_DIR" \
    --ada_path "$ADA_BEST_MODEL" \
    --output_dir "$FINAL_OUTPUT_DIR" \
    --spot_q 0.00001 \
    --template_dir "./failure_templates"

log_success "Pipeline finished successfully!"
echo "=========================================================="
echo "Results available in: $FINAL_OUTPUT_DIR"
echo "=========================================================="