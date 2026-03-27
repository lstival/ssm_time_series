#!/bin/bash

# --- CONFIGURATION ---
# Replace 'your_server_address' with the actual hostname or IP of the server.
REMOTE_USER="stiva001"
REMOTE_HOST="login201.internal.anunna.wur.nl" 
REMOTE_PATH="/home/WUR/stiva001/WUR/ssm_time_series"
LOCAL_PATH="." # Downloads to the current directory

# List of specific checkpoint files identified from the YAML configs
FILES=(
    "checkpoints/multi_horizon_forecast_dual_frozen_20251209_1049/all_datasets/best_model.pt"
    "checkpoints/multi_horizon_forecast_chronos_20251209_0946/all_datasets/best_model.pt"
    "checkpoints/multi_horizon_forecast_chronos_20251209_0936/all_datasets/best_model.pt"
    "checkpoints/ts_encoder_20251126_1750_ep10/time_series_best.pt"
    "checkpoints/ts_encoder_20251126_1750_ep10/visual_encoder_best.pt"
)

echo "Starting download from $REMOTE_HOST..."

# Create local checkpoints directory if it doesn't exist
mkdir -p "$LOCAL_PATH/checkpoints"

for FILE in "${FILES[@]}"; do
    # Extract the directory path to create it locally
    REL_DIR=$(dirname "$FILE")
    mkdir -p "$LOCAL_PATH/$REL_DIR"
    
    echo "Downloading: $FILE"
    rsync -avzP "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/$FILE" "$LOCAL_PATH/$FILE"
done

echo "-----------------------------------"
echo "Download complete."
echo "Checkpoints are located in: $(realpath $LOCAL_PATH/checkpoints)"
