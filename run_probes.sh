#!/bin/bash
DEVICE="cuda:6"
TRAIN_DEST_DIR="probing/probing-chess-mamba-4-train"
TEST_DEST_DIR="probing/probing-chess-mamba-4-test"
MODEL="/mnt/ssd-1/gpaulo/mambaChess/chess-mamba-4"

# Generate and record model states/activations and board states
python -m probing.generate_probing_data --device $DEVICE --dataset train_dataset_lichess_200k_elo_bins.zip --model $MODEL --output-dir $TRAIN_DEST_DIR --max-iters 100
python -m probing.generate_probing_data --device $DEVICE --dataset test_dataset_lichess_200k_elo_bins.zip --model $MODEL --output-dir $TEST_DEST_DIR --max-iters 100

# Train probes and print results on the testing set
python -m probing.train_probe --device $DEVICE --train-data $TRAIN_DEST_DIR --test-data $TEST_DEST_DIR