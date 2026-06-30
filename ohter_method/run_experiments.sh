#!/bin/bash

# ==============================================================================
# AI-Generated Text Detection Benchmark Script
# This script runs four different detection evaluation methods:
# 1. Baselines (Likelihood, Rank, Logrank, Entropy)
# 2. DetectGPT (Perturbation-based)
# 3. Fast-DetectGPT (Sampling discrepancy-based)
# 4. RoBERTa-based classification
# ==============================================================================

# Define global variables for experiment configuration
DATASET="xsum"
DATA_FILE="./data/my_dataset"        # Path prefix for your raw data
OUTPUT_DIR="./results"               # Directory to save experimental results
CACHE_DIR="./models"                 # Directory for cached model weights
DEVICE="cuda"                        # Execution device: 'cuda' or 'cpu'
SCORING_MODEL="gpt2"                 # The LLM to be evaluated

echo "Starting AI-Generated Text Detection Experiments..."

# 1. Run Baseline Experiments
echo "--- Running Baselines ---"
python baselines.py \
    --dataset "$DATASET" \
    --dataset_file "$DATA_FILE" \
    --scoring_model_name "$SCORING_MODEL" \
    --output_file "$OUTPUT_DIR/baseline" \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR"

# 2. Run DetectGPT (requires mask filling model)
echo "--- Running DetectGPT ---"
python detect_gpt.py \
    --dataset "$DATASET" \
    --dataset_file "$DATA_FILE" \
    --scoring_model_name "$SCORING_MODEL" \
    --output_file "$OUTPUT_DIR/detect_gpt" \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR"

# 3. Run Fast-DetectGPT (efficient sampling discrepancy estimation)
echo "--- Running Fast-DetectGPT ---"
python fast_detect_gpt.py \
    --dataset "$DATASET" \
    --dataset_file "$DATA_FILE" \
    --scoring_model_name "$SCORING_MODEL" \
    --reference_model_name "$SCORING_MODEL" \
    --output_file "$OUTPUT_DIR/fast_detect" \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR"

# 4. Run RoBERTa Detection
# Note: Ensure the raw_data.json exists at the specified location
echo "--- Running RoBERTa Detection ---"
python roberta.py \
    --dataset "$DATASET" \
    --dataset_file "${DATA_FILE}.raw_data.json" \
    --roberta_model_name "roberta-base" \
    --output_file "$OUTPUT_DIR/roberta" \
    --device "$DEVICE" \
    --cache_dir "$CACHE_DIR"

echo "All experiments finished! Results saved in $OUTPUT_DIR."