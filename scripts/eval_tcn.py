import os
import sys
import argparse
import glob
import yaml

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.append(SRC)

import lightning as L
import torch
from torch.utils.data import DataLoader

import mirdata
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import random

from pre.data_loader import BeatData
from model.tcn import MultiTracker
from model.lightning_module import PLTCN
from utils.split_utils import carn_split_keys

# changed path of the model in get_params_path, hardcoded seed to test individually in each run

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TCN models with different modes')
    parser.add_argument('--data_home', type=str, default='../data/',
                    help='Path to the mounted datasets directory')
    parser.add_argument('--mode', type=str, choices=['BL', 'FT', 'FS'], required=True,
                        help='Evaluation mode: BL (baseline), FT (fine-tuned), or FS (from scratch)')
    return parser.parse_args()

# * 1
def get_params_paths(mode):
    """Get all checkpoint paths for the given mode."""
    if mode == 'BL':
        with open('../config/train_BL.yaml', 'r') as f:
            config = yaml.safe_load(f)
        with open('../config/model.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        config['model'] = model_config['model']
        PARAMS = get_params(config)
        ckpt_name = config['experiment']['ckpt_name']
        ckpt_dir = os.path.join(ROOT, 'pretrained', ckpt_name)
        pattern = f'{ckpt_name}-*.ckpt'


    elif mode == 'FT':
        with open('../config/train_FT.yaml', 'r') as f:
            config = yaml.safe_load(f)
        with open('../config/model.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        config['model'] = model_config['model']
        PARAMS = get_params(config)
        ckpt_name = config['experiment']['ckpt_name']
        ckpt_dir = os.path.join(ROOT, 'pretrained', ckpt_name)
        pattern = f'{ckpt_name}-*.ckpt'

    elif mode == 'FS':
        with open('../config/train_FS.yaml', 'r') as f:
            config = yaml.safe_load(f)
        with open('../config/model.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        config['model'] = model_config['model']
        PARAMS = get_params(config)
        ckpt_name = config['experiment']['ckpt_name']
        ckpt_dir = os.path.join(ROOT, 'output', 'checkpoints', '202512091016')
        pattern = f'{ckpt_name}-*.ckpt'

    ckpt_paths = glob.glob(os.path.join(ckpt_dir, pattern))
    return PARAMS, ckpt_paths

def get_params(config):
    """Get model parameters for the given mode from config file"""
    PARAMS = {
            # Model parameters
            "N_FILTERS": config['model']['n_filters'],
            "KERNEL_SIZE": config['model']['kernel_size'],
            "DROPOUT": config['model']['dropout'],
            "N_DILATIONS": config['model']['n_dilations'],

            # Training configuration
            "LEARNING_RATE": config['training']['learning_rate'],
            "N_EPOCHS": config['training']['n_epochs'],
            "LOSS": config['training']['loss'],
            "POST_PROCESSOR": config['training']['post_processor'],
            "BATCH_SIZE": 1,
            "NUM_WORKERS": config['training']['num_workers'],
            "EARLY_STOP_PATIENCE": config['training']['early_stop_patience'],
            "EARLY_STOP_MIN_DELTA": config['training']['early_stop_min_delta'],
            "SCHEDULER_FACTOR": config['training']['scheduler_factor'],
            "SCHEDULER_PATIENCE": config['training']['scheduler_patience'],
            "TEST_SIZE": config['training']['test_size'],

            # Experiment tracking
            "CKPT_NAME": config['experiment']['ckpt_name']
            }

    return PARAMS


def evaluate_baseline(PARAMS, ckpt_path, carn_tracks, carn_keys, accelerator):
    """Evaluate baseline model on complete dataset."""

    # Extract fold and run information from checkpoint name
    ckpt_name = os.path.basename(ckpt_path)

    # Format: tcn_bl-runX.ckpt
    parts = ckpt_name.replace('.ckpt', '').split('-')
    run = int(parts[1].replace('run', ''))

    # Determine seed based on run number
    seeds = {1: 42, 2: 52, 3: 62}
    seed = 62

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)

    # Test on complete dataset
    test_data = BeatData(carn_tracks, carn_keys, widen=True)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=0, pin_memory=False, persistent_workers=False)

    # Initialize model
    tcn = MultiTracker(
        n_filters=PARAMS["N_FILTERS"],
        n_dilations=PARAMS["N_DILATIONS"],
        kernel_size=PARAMS["KERNEL_SIZE"],
        dropout_rate=PARAMS["DROPOUT"]
    )

    model = PLTCN.load_from_checkpoint(
        ckpt_path,
        model=tcn,
        params=PARAMS
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=PARAMS["N_EPOCHS"],
        accelerator=accelerator,
    )

    # Run test
    trainer.test(model, test_loader, verbose=True)

    return process_and_save_results(ckpt_path, parts[0], carn_tracks)


def evaluate_ft_fs(PARAMS, ckpt_path, carn_tracks, accelerator):
    """Evaluate FT/FS models using fold-based evaluation."""

    # Extract fold and run information from checkpoint name
    ckpt_name = os.path.basename(ckpt_path)

    # Format: tcn_carnatic_xx-trainfoldX-runY.ckpt
    parts = ckpt_name.replace('.ckpt', '').split('-')
    train_fold = int(parts[1].replace('trainfold', ''))
    run = int(parts[2].replace('run', ''))
    test_fold = 3 - train_fold

    # Determine seed based on run number
    seeds = {1: 42, 2: 52, 3: 62}
    seed = 62

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)

    # Load splits
    csv_path = os.path.join(ROOT, 'data', 'cmr_splits.csv')
    carn_train_keys, carn_val_keys, carn_test_keys = carn_split_keys(
        csv_path=csv_path,
        split_col='Taala',
        test_size=PARAMS["TEST_SIZE"],
        seed=seed,
        reorder=True
    )

    # Use only test split for evaluation
    test_data = BeatData(carn_tracks, carn_test_keys, widen=True)
    test_loader = DataLoader(test_data, batch_size=PARAMS["BATCH_SIZE"], num_workers=PARAMS["NUM_WORKERS"])

    # Initialize model
    tcn = MultiTracker(
        n_filters=PARAMS["N_FILTERS"],
        n_dilations=PARAMS["N_DILATIONS"],
        kernel_size=PARAMS["KERNEL_SIZE"],
        dropout_rate=PARAMS["DROPOUT"]
    )

    model = PLTCN.load_from_checkpoint(
        ckpt_path,
        model=tcn,
        params=PARAMS
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=PARAMS["N_EPOCHS"],
        accelerator=accelerator,
        devices=1
    )

    # Run test
    trainer.test(model, test_loader, verbose=True)

    return process_and_save_results(ckpt_path, parts[0],carn_tracks)


def process_and_save_results(ckpt_path, folder, carn_tracks):
    """Process results and save to CSV."""

    tmp_results_path = os.path.join(os.getcwd(), 'temp', 'results.pkl')
    if os.path.exists(tmp_results_path):
        results_pkl = pd.read_pickle(tmp_results_path)
    else:
        print(f"Results file {tmp_results_path} not found.")
        return

    # Create main DataFrame
    df = pd.DataFrame(results_pkl)
    # Add taala column using the track_id
    df['taala'] = df['track_id'].apply(lambda x: carn_tracks[x].taala if x in carn_tracks else '')
    # Rearrange columns to have 'track_id' and 'taala' first
    df = df[['track_id', 'taala'] + [col for col in df.columns if col not in ['track_id', 'taala']]]

    # Compute overall average (excluding non-numeric columns)
    avg_metrics = df.drop(columns=['track_id', 'taala']).mean()
    avg_metrics['track_id'] = 'average'
    avg_metrics['taala'] = ''

    # Compute per-Taala averages
    taala_averages = []
    for taala, group in df.groupby('taala'):
        taala_avg = group.drop(columns=['track_id', 'taala']).mean()
        taala_avg['track_id'] = ''
        taala_avg['taala'] = taala
        taala_averages.append(taala_avg)

    # Convert list of Series to DataFrame
    taala_avg_df = pd.DataFrame(taala_averages)

    # Concatenate everything: original data + taala averages + overall average
    df_with_avg = pd.concat([df, pd.DataFrame([avg_metrics]), taala_avg_df], ignore_index=True)

    # Save results
    ckpt_name = os.path.basename(ckpt_path)

    output_dir = os.path.join(ROOT, 'output', 'results', folder)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, ckpt_name.replace('.ckpt', '.csv'))
    df_with_avg.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Delete pkl file
    if os.path.exists(tmp_results_path):
        os.remove(tmp_results_path)
        print(f"Deleted temporary file: {tmp_results_path}")


def main():
    args = parse_args()

    # Load Dataset
    data_home = args.data_home

    carn = mirdata.initialize('compmusic_carnatic_rhythm', version='full_dataset_1.0', data_home=data_home)
    #carn.download(['index'])
    carn_tracks = carn.load_tracks()
    carn_keys = list(carn_tracks.keys())

    # Get parameters and checkpoint paths for the specified mode
    PARAMS, ckpt_paths = get_params_paths(args.mode)

    if not ckpt_paths:
        print(f"No checkpoints found for mode {args.mode}")
        return

    print(f"Found {len(ckpt_paths)} checkpoints for mode {args.mode}")

        # Device Setup
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"


    # Evaluate each checkpoint
    for ckpt_path in sorted(ckpt_paths):
        print(f"\nEvaluating checkpoint: {os.path.basename(ckpt_path)}")
        try:
            if args.mode == 'BL':
                # BL: Evaluate on complete dataset
                evaluate_baseline(PARAMS, ckpt_path, carn_tracks, carn_keys, accelerator)
            else:
                # FT/FS: Evaluate using fold-based splits
                evaluate_ft_fs(PARAMS, ckpt_path, carn_tracks, accelerator)
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {str(e)}")
            continue

    print(f"\nCompleted evaluation for mode {args.mode}")


if __name__ == "__main__":
    main()
