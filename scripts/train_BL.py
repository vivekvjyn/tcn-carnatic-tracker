import os
import sys
import shutil
from datetime import datetime
import time
import argparse
import yaml

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.append(SRC)

import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

import wandb

from pre.dataset_config import DatasetManager
from pre.data_loader import MultiBeatData
from model.tcn import MultiTracker
from model.lightning_module import PLTCN
from utils.split_utils import get_split_keys

import numpy as np
import random


# ----- Argument Parsing -----

parser = argparse.ArgumentParser(description='Train TCN model on multiple datasets')
parser.add_argument('--data_home', type=str, default='../data',
                    help='Path to the mounted datasets directory')
parser.add_argument('--datasets', nargs='+',
                    default=['gtzan_genre', 'beatles', 'ballroom',
                            'rwc_popular', 'rwc_jazz', 'rwc_classical'],
                    help='Datasets to include in training')
parser.add_argument('--disable-wandb', action='store_true',
                    help='Disable WandB logging for trial run')
args = parser.parse_args()

data_home = args.data_home

# Verify the data_home exists
if not os.path.exists(data_home):
    raise FileNotFoundError(f"Data home directory not found: {data_home}")

# Load configuration
with open('../config/train_BL.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('../config/model.yaml', 'r') as f:
    model_config = yaml.safe_load(f)
config['model'] = model_config['model']

print(f"Using data home: {data_home}")
print(f"Using config file: config/train_BL.yaml")

# Set WandB mode based on the flag
if args.disable_wandb:
    os.environ["WANDB_MODE"] = "disabled"
    print("WandB logging disabled")
else:
    print("WandB logging enabled")

# ----- Set Training Parameters -----

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
    "BATCH_SIZE": config['training']['batch_size'],
    "NUM_WORKERS": config['training']['num_workers'],
    "EARLY_STOP_PATIENCE": config['training']['early_stop_patience'],
    "EARLY_STOP_MIN_DELTA": config['training']['early_stop_min_delta'],
    "SCHEDULER_FACTOR": config['training']['scheduler_factor'],
    "SCHEDULER_PATIENCE": config['training']['scheduler_patience'],
    "TEST_SIZE": config['training']['test_size'],

    # Experiment tracking
    "PROJECT_NAME": config['experiment']['project_name'],
    "WANDB_API_KEY": config['experiment']['wandb_api_key'],
    "CKPT_NAME": config['experiment']['ckpt_name']
}

# ----- Load Dataset -----

print("Initializing datasets...")
dataset_manager = DatasetManager(args.data_home, datasets=args.datasets)
dataset_manager.initialize_datasets()

# Get all keys and tracks for potential use in training
keys = dataset_manager.get_all_valid_keys()  # Get all valid keys as (dataset_name, key) tuples
tracks = dataset_manager.tracks  # Get all tracks from all datasets

# Create labels (dataset names) for stratified splitting
all_labels = [dataset_name for dataset_name, _ in keys]

# Print summary
summary = dataset_manager.get_dataset_summary()
print("\nDataset Summary:")
for name, stats in summary.items():
    if name == 'overall':
        print(f"\nOverall: {stats['total_valid_tracks']} valid tracks across {stats['datasets_loaded']} datasets")
    else:
        print(f"{name}: {stats['valid_tracks']}/{stats['total_tracks']} "
              f"({stats['valid_ratio']:.2%} valid)")


# ----- Device Setup -----
if torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
else:
    device = torch.device("cpu")
    accelerator = "cpu"

num_workers = PARAMS['NUM_WORKERS']

if args.disable_wandb:
    os.environ["WANDB_MODE"] = "disabled"
    print("WandB logging disabled")
else:
    print("WandB logging enabled")

# ----- Set seeds -----
for run, seed in enumerate([42, 52, 62], start=1):

    print(f"Running run {run} with seed {seed}")

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


    # ----- Create Splits -----

    # Use the custom split function to stratify the split based on dataset names
    train_keys, val_keys, test_keys = get_split_keys(
        keys=keys,
        labels=all_labels,
        test_size=PARAMS['TEST_SIZE'],
        seed=seed,
        shuffle=True
    )

    # Print dataset distribution in splits
    dataset_manager.print_split_distribution(train_keys, "Training")
    dataset_manager.print_split_distribution(val_keys, "Validation")
    dataset_manager.print_split_distribution(test_keys, "Test")


    # Create DataLoaders for each split
    train_data = MultiBeatData(tracks, train_keys, widen=True)
    val_data = MultiBeatData(tracks, val_keys, widen=True)
    test_data = MultiBeatData(tracks, test_keys, widen=True)

    train_loader = DataLoader(train_data, batch_size=PARAMS['BATCH_SIZE'], num_workers=PARAMS['NUM_WORKERS'])
    val_loader = DataLoader(val_data, batch_size=PARAMS['BATCH_SIZE'], num_workers=PARAMS['NUM_WORKERS'])
    test_loader = DataLoader(test_data, batch_size=PARAMS['BATCH_SIZE'], num_workers=PARAMS['NUM_WORKERS'])

    # ----- Model and Lightning Module -----
    tcn = MultiTracker(
        n_filters=PARAMS["N_FILTERS"],
        n_dilations=PARAMS["N_DILATIONS"],
        kernel_size=PARAMS["KERNEL_SIZE"],
        dropout_rate=PARAMS["DROPOUT"]
    )

    model = PLTCN(model=tcn, params=PARAMS)

    # ----- Callbacks -----
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    ckpt_name = PARAMS["CKPT_NAME"]

    CKPTS_DIR = os.path.join(ROOT, 'output', 'checkpoints', timestamp)
    os.makedirs(CKPTS_DIR, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CKPTS_DIR,
        filename=f"{ckpt_name}-run{run}" + "-{epoch:02d} -{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=PARAMS["EARLY_STOP_PATIENCE"],       # stop training if no improvement for 20 epochs
        mode="min",
        min_delta=PARAMS["EARLY_STOP_MIN_DELTA"],  # minimum change to qualify as an improvement
        verbose=True
    )

    # ----- Loggers -----

    run_config = PARAMS.copy()
    # Remove any existing wandb api key
    if "WANDB_API_KEY" in run_config:
        del run_config["WANDB_API_KEY"]
    run_config.update({
        "RUN_ID": run,
        "SEED": seed
    })

    wandb.login(key=PARAMS["WANDB_API_KEY"])
    wandb_run = wandb.init(
        project=PARAMS["PROJECT_NAME"],
        name=f"{PARAMS['PROJECT_NAME']}_{timestamp}_run{run}_seed{seed}",
        config=run_config,
        reinit='finish_previous',
        mode="disabled" if args.disable_wandb else "online",

    )

    wandb_logger = WandbLogger(experiment=wandb_run)
    csv_logger = CSVLogger("lightning_logs")  # this gives you metrics.csv


    # ----- Trainer -----
    trainer = L.Trainer(
        max_epochs=PARAMS["N_EPOCHS"],
        accelerator=accelerator,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[csv_logger, wandb_logger]  # explicitly include both
    )

    # ----- Train -----
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    end_time = time.time()


    train_duration = end_time - start_time
    # Format as HH:MM:SS
    hours, rem = divmod(train_duration, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    print(f"Total training time: {time_str} (hh:mm:ss)")

    # Log training time to W&B
    wandb.log({"train_time_sec": train_duration})

    # ----- Copy Train Logs -----
    try:
        metrics_src = os.path.join(csv_logger.log_dir, "metrics.csv")

        # Target path: output/checkpoints/<timestamp>/metrics.csv
        metrics_dest = os.path.join(CKPTS_DIR, "metrics.csv")

        # Copy it
        shutil.copyfile(metrics_src, metrics_dest)
        print(f"Copied Lightning metrics.csv to: {metrics_dest}")

        # Copy the generated checkpoint into the pretrained folder
        latest_ckpt = checkpoint_callback.best_model_path
        pretrained_dir = os.path.join(ROOT, 'pretrained', 'tcn_bl')
        os.makedirs(pretrained_dir, exist_ok=True)
        shutil.copyfile(latest_ckpt, os.path.join(pretrained_dir, f"{ckpt_name}-run{run}.ckpt"))
        print(f"Copied checkpoint to: {os.path.join(pretrained_dir, f'{ckpt_name}-run{run}.ckpt')}")

    except Exception as e:
        print(f"Warning: Failed to copy checkpoint to the 'pretrained' folder: {e}. You can still find it in the output/checkpoints/<timestamp> folder.")
        print("Training will continue...")

    finally:
        # Clean up the wandb run directory
        wandb_run.finish()
        print("Wandb run finished")

print(f"Training completed successfully!")
