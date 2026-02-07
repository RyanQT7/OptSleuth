@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM   Optical Network Anomaly Detection Pipeline (Windows Version)
REM ==============================================================================

echo ==========================================================
echo    Starting Optical Network Anomaly Detection Pipeline    
echo ==========================================================

REM --- Configuration ---

REM Project Root Directories
set "DATA_ROOT=.\data_workspace"
set "RAW_DATA_DIR=%DATA_ROOT%\raw_data"
set "PROCESSED_DATA_DIR=%DATA_ROOT%\processed_data"
set "RESULTS_DIR=.\experiment_results"

REM Model Checkpoint Directories
set "ADA_SAVE_DIR=%RESULTS_DIR%\checkpoints\ada"
set "GAT_SAVE_DIR=%RESULTS_DIR%\checkpoints\gat"
set "FINAL_OUTPUT_DIR=%RESULTS_DIR%\final_inference"

REM Hyperparameters
set EPOCHS_ADA=50
set EPOCHS_GAT=30
set BATCH_SIZE=64

REM --- Device Configuration ---
REM Default to CPU for compatibility/reproducibility
set DEVICE=cpu
REM TIP: If you have an NVIDIA GPU, change the above line to: set DEVICE=cuda

REM --- 0. Setup Directories ---
echo [INFO] Setting up directories...
if not exist "%DATA_ROOT%" mkdir "%DATA_ROOT%"
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%ADA_SAVE_DIR%" mkdir "%ADA_SAVE_DIR%"
if not exist "%GAT_SAVE_DIR%" mkdir "%GAT_SAVE_DIR%"
if not exist "%FINAL_OUTPUT_DIR%" mkdir "%FINAL_OUTPUT_DIR%"

REM --- 1. Data Setup (Skipping Generation) ---
echo.
echo [INFO] Step 1: Checking for existing data...

REM Logic: If user placed data in 'mock_dataset', move it to the workspace raw dir.
if exist "mock_dataset" (
    echo [INFO] Found 'mock_dataset' folder, moving to workspace...
    if exist "%RAW_DATA_DIR%" rmdir /s /q "%RAW_DATA_DIR%"
    move "mock_dataset" "%RAW_DATA_DIR%" >nul
    echo [SUCCESS] Data moved to %RAW_DATA_DIR%
)

REM Logic: Verify that data actually exists where preprocessing expects it
if not exist "%RAW_DATA_DIR%" (
    echo.
    echo [ERROR] Data Missing! 
    echo Please place your CSV files in a folder named 'mock_dataset' in the root directory,
    echo OR directly into '%RAW_DATA_DIR%'.
    goto :error
)

REM --- 2. Preprocessing ---
echo.
echo [INFO] Step 2: Preprocessing Data (Normalization)...
python data\preprocess.py
if !errorlevel! neq 0 goto :error

REM Move output if necessary (Assuming script outputs to ./processed_dataset)
if exist "processed_dataset" (
    echo [INFO] Moving processed data to workspace...
    if exist "%PROCESSED_DATA_DIR%" rmdir /s /q "%PROCESSED_DATA_DIR%"
    move "processed_dataset" "%PROCESSED_DATA_DIR%" >nul
)
echo [SUCCESS] Preprocessing complete. Data at: %PROCESSED_DATA_DIR%

REM --- 3. ADA Training ---
echo.
echo [INFO] Step 3: Training Unified Domain Adapter (ADA) on %DEVICE%...
python train_ada.py ^
    --data_root "%PROCESSED_DATA_DIR%" ^
    --save_dir "%ADA_SAVE_DIR%" ^
    --epochs %EPOCHS_ADA% ^
    --batch_size %BATCH_SIZE% ^
    --device %DEVICE%

if !errorlevel! neq 0 goto :error

set "ADA_BEST_MODEL=%ADA_SAVE_DIR%\unified_ada_epoch_%EPOCHS_ADA%.pt"
if not exist "%ADA_BEST_MODEL%" (
    echo [ERROR] ADA Model not found at %ADA_BEST_MODEL%
    goto :error
)
echo [SUCCESS] ADA Training complete.

REM --- 4. GAT Training ---
echo.
echo [INFO] Step 4: Training Graph Attention Network (GAT) on %DEVICE%...
python train_gat.py ^
    --data_root "%PROCESSED_DATA_DIR%" ^
    --ada_path "%ADA_BEST_MODEL%" ^
    --save_root "%GAT_SAVE_DIR%" ^
    --epochs %EPOCHS_GAT% ^
    --device %DEVICE% ^
    --batch_size 1

if !errorlevel! neq 0 goto :error
echo [SUCCESS] GAT Training complete.

REM --- 5. GAT Testing ---
echo.
echo [INFO] Step 5: Running Final Testing and Anomaly Detection...
python test_gat.py ^
    --data_root "%PROCESSED_DATA_DIR%" ^
    --model_dir "%GAT_SAVE_DIR%" ^
    --ada_path "%ADA_BEST_MODEL%" ^
    --output_dir "%FINAL_OUTPUT_DIR%" ^
    --spot_q 0.00001 ^
    --template_dir ".\failure_templates"

if !errorlevel! neq 0 goto :error

echo.
echo ==========================================================
echo [SUCCESS] Pipeline finished successfully!
echo Results available in: %FINAL_OUTPUT_DIR%
echo ==========================================================
pause
exit /b 0

:error
echo.
echo ==========================================================
echo [ERROR] Pipeline failed! Please check the logs above.
echo ==========================================================
pause
exit /b 1