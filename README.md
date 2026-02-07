# OptSleuth: Unified and Interpretable Incident Management for Optical Networks in AI Clusters

The explosive growth of Artificial Intelligence (AI) has made large-scale clusters with optical network backbones critical infrastructure, where any optical device failure can result in catastrophic job interruptions and substantial economic losses. Effective incident management requires accurate anomaly detection, precise root cause localization, and reliable failure diagnosis. However, existing approaches are hindered by a lack of generalization across multi-vendor devices, an inability to model complex spatiotemporal failure propagation, and poor interpretability for operational trust. In this paper, we present OptSleuth, a unified incident management framework. By learning vendor-agnostic representations and then forming spatio-temporal failure representations, OptSleuth performs interpretable reasoning for multiple tasks, which are jointly optimized through a unified loss function.

## Project Structure

```text
.
├── data/
│   ├── preprocess.py          # Data normalization and cleaning script
│   ├── optical_dataset.py     # PyTorch Dataset for ADA training
│   └── switch_dataset.py      # Dataset for GAT training (graph-based)
├── models/
│   ├── domain_adapter.py      # ADA model (Transformer Encoder + GRL)
│   └── gat.py                 # GAT model (Graph Attention + Reconstruction Decoder)
├── utils/
│   ├── spot.py                # SPOT algorithm for thresholding
│   ├── pattern_matcher.py     # DTW-based pattern matching module
│   └── rca.py                 # Root cause localization logic (PageRank-like)
├── train_ada.py               # Script for Stage 1: Domain Adaptation Training
├── train_gat.py               # Script for Stage 2: GAT Training
├── test_gat.py                # Script for Stage 3: Inference & Evaluation
├── run_pipeline.bat           # Execution script for Windows
├── run_pipeline.sh            # Execution script for Linux/Bash
├── requirements.txt           # Dependency lock file
└── README.md                  # Project documentation

```

---

## Dataset & Data Structure

### Data Privacy & Synthetic Generation Disclaimer

> **Note:** Due to strict Non-Disclosure Agreements (NDA) regarding production network topology and vendor specifications, the dataset provided in this repository (`mock_dataset/`) consists of **high-fidelity synthetic data**.
> This data is algorithmically generated to demonstrate the input format, feature engineering, and pipeline functionality of OptSleuth.

### Directory Hierarchy

The data is organized hierarchically by network switches. Each switch folder contains time-series CSV files representing individual optical device.

```text
mock_dataset/
├── switch_01/                 # Switch Identifier
│   ├── link_data_01.csv       # Time-series data for a specific optical device
│   ├── link_data_02.csv
│   └── ...
├── switch_02/
│   └── ...
└── ...

```

### Feature Description (DDM Metrics)

Each CSV file contains **Digital Diagnostics Monitoring Interface (DDM)** data collected at high frequency. The dataset captures the state of a bi-directional optical device, including both the **Local** side (the reporting switch) and the **Remote** side (the connected peer device).

The features support multi-lane **optical devices**, detailed as follows:

| Feature Category | Column Names (Pattern) | Description |
| --- | --- | --- |
| **Metadata** | `timestamp` | Sampling time (Precision: Microseconds). |
|  | `local/remote_vendorSn` | Unique Serial Number of the optical device. |
|  | `local/remote_optical_id` | Unique Logical ID of the optical device. |
| **Electrical Health** | `local/remote_voltage` | Supply voltage (V). |
|  | `local/remote_current` | Supply current (mA). |
| **Bias Current** | `*_currentMultiBias[1-4]` | Laser bias current for each of the 4 parallel lanes (mA). Critical for detecting laser aging. |
| **Optical Power (TX)** | `*_currentTXPower` | Aggregate Transmit Optical Power (dBm/mW). |
|  | `*_currentMultiTXPower[1-4]` | Transmit Optical Power for individual lanes 1-4. |
| **Optical Power (RX)** | `*_currentRXPower` | Aggregate Receive Optical Power (dBm/mW). |
|  | `*_currentMultiRXPower[1-4]` | Receive Optical Power for individual lanes 1-4. |
| **Environment** | `local/remote_temperature` | **Device** case temperature (°C). |

### Data Sample

A standardized input row (CSV format) representing a single timestamp snapshot:

```csv
timestamp,local_vendorSn,remote_vendorSn,local_optical_id,remote_optical_id,local_voltage,local_current,local_currentTXPower,local_currentRXPower,local_currentMultiBias1,local_currentMultiBias2,local_currentMultiBias3,local_currentMultiBias4,local_currentMultiRXPower1,local_currentMultiRXPower2,local_currentMultiRXPower3,local_currentMultiRXPower4,local_currentMultiTXPower1,local_currentMultiTXPower2,local_currentMultiTXPower3,local_currentMultiTXPower4,local_temperature,remote_voltage,remote_current,remote_currentTXPower,remote_currentRXPower,remote_currentMultiBias1,remote_currentMultiBias2,remote_currentMultiBias3,remote_currentMultiBias4,remote_currentMultiRXPower1,remote_currentMultiRXPower2,remote_currentMultiRXPower3,remote_currentMultiRXPower4,remote_currentMultiTXPower1,remote_currentMultiTXPower2,remote_currentMultiTXPower3,remote_currentMultiTXPower4,remote_temperature
2026-01-24 20:17:49.994703,Vendor_C-SN,Vendor_A-SN,OPT-LOC-773,OPT-REM-162,4.37,10.28,10.27,10.73,10.27,10.18,10.62,11.86,10.81,10.18,10.37,10.83,10.47,10.62,10.66,11.05,45.83,3.16,11.29,11.11,10.90,10.54,9.46,10.16,11.11,10.71,10.19,9.94,10.47,11.43,10.09,10.52,10.21,48.23

```

---

## Installation

### Prerequisites

* **Python 3.8+**

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/OptSleuth.git
cd OptSleuth

```

2. Install dependencies:

* **Standard Install**:

```bash
pip install -r requirements.txt

```

---

## Usage Pipeline

You can run the entire pipeline automatically using the provided scripts, or execute each step manually.

### Quick Start (Automated)

The script handles directory setup, data moving, training, and testing in one go.

* **Windows**: `.\run_pipeline.bat`
* **Linux/Mac**: `./run_pipeline.sh`

---

### Manual Execution

If you prefer to run step-by-step, follow the commands below.
*(Note: These examples use `--device cpu` by default. Change to `cuda` if you have an NVIDIA GPU.)*

### Step 1: Data Preprocessing

Standardizes the raw DDM data, handles missing values, and aligns timestamps.
*(Ensure your data is in `./mock_dataset` or `./data_workspace/raw_data` before running)*

```bash
python data/preprocess.py
# The script automatically processes data into ./data_workspace/processed_data

```

### Step 2: Train Domain Adapter (ADA)

Trains the Transformer-based Domain Adapter to learn vendor-invariant feature representations.

```bash
python train_ada.py \
    --data_root "./data_workspace/processed_data" \
    --save_dir "./experiment_results/checkpoints/ada" \
    --epochs 50 \
    --batch_size 64 \
    --device cpu

```

### Step 3: Train Graph Network (GAT)

Trains the Spatio-Temporal Graph Attention Network using the frozen features from Step 2.

> **Note:** We use the specific checkpoint `unified_ada_epoch_50.pt` generated in Step 2.

```bash
python train_gat.py \
    --data_root "./data_workspace/processed_data" \
    --ada_path "./experiment_results/checkpoints/ada/unified_ada_epoch_50.pt" \
    --save_root "./experiment_results/checkpoints/gat" \
    --epochs 30 \
    --batch_size 1 \
    --device cpu

```

### Step 4: Incident management

Runs anomaly detection, root cause localization and pattern matching on the test set.

```bash
python test_gat.py \
    --data_root "./data_workspace/processed_data" \
    --model_dir "./experiment_results/checkpoints/gat" \
    --ada_path "./experiment_results/checkpoints/ada/unified_ada_epoch_50.pt" \
    --output_dir "./experiment_results/final_inference" \
    --template_dir "./failure_templates" \
    --spot_q 0.00001

```