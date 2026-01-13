import os
import sys
from datetime import datetime
import time
import argparse
import yaml
import shutil
from lightning.pytorch.loggers import CSVLogger, WandbLogger

ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.append(SRC)

import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import mirdata

from pre.data_loader import BeatData
from model.tcn import MultiTracker
from model.lightning_module import PLTCN
from utils.split_utils import carn_split_keys
import wandb
import numpy as np
import random

# ----- Argument Parsing and Configuration -----

parser = argparse.ArgumentParser(description='Train TCN model from scratch on Carnatic Music Rhythm (CMR) dataset')
parser.add_argument('--data_home', type=str, default='../data/',
                    help='Path to the mounted datasets directory')
parser.add_argument('--disable-wandb', action='store_true',
                    help='Disable WandB logging')
args = parser.parse_args()

# Verify the data_home exists
data_home = args.data_home
if not os.path.exists(data_home):
    raise FileNotFoundError(f"Data home directory not found: {data_home}")

# Load configuration
with open('../config/train_FS.yaml', 'r') as f:
    config = yaml.safe_load(f)
with open('../config/model.yaml', 'r') as f:
    model_config = yaml.safe_load(f)
config['model'] = model_config['model']

print(f"Using data home: {data_home}")
print(f"Using config file: config/train_FS.yaml")

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
carn = mirdata.initialize('compmusic_carnatic_rhythm', version='full_dataset_1.0', data_home=data_home)
#carn.download(['index'])
print('Validating dataset...')
mis, inv = carn.validate()

if mis != {'tracks': {}}:
    print("Error: Dataset validation failed due to missing files.")
    print("Please ensure the data is in the correct location and the dataset is complete.")
    print(f"Ensure the 'CMR_full_dataset_1.0' folder exists in {data_home}")
    sys.exit(1)

carn_tracks = carn.load_tracks()
carn_keys = list(carn_tracks.keys())

# ----- Device Setup -----
if torch.cuda.is_available():
    device = torch.device("cuda")
    accelerator = "gpu"
else:
    device = torch.device("cpu")
    accelerator = "cpu"

num_workers = PARAMS['NUM_WORKERS']


# ----- Training Loop -----
    # ----- Set seeds -----
for run, seed in enumerate([42, 52, 62], start=1):

    print(f"Run {run} with seed {seed}")

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)

    # ----- Load Splits -----
    csv_path = os.path.join(ROOT, 'data', 'cmr_splits.csv')

    carn_train_keys, carn_val_keys, carn_test_keys = carn_split_keys(
                                                            csv_path=csv_path,
                                                            split_col='Taala',
                                                            test_size=PARAMS['TEST_SIZE'],
                                                            seed=seed,
                                                            reorder=True
                                                        )

    print(f"Train keys: {len(carn_train_keys)}, Val keys: {len(carn_val_keys)}, Test keys: {len(carn_test_keys)}")

    # Prepare datasets and loaders
    train_data = BeatData(carn_tracks, carn_train_keys, widen=True)
    val_data = BeatData(carn_tracks, carn_val_keys, widen=True)
    test_data = BeatData(carn_tracks, carn_test_keys, widen=True)

    train_loader = DataLoader(train_data, batch_size=PARAMS['BATCH_SIZE'], num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=PARAMS['BATCH_SIZE'], num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=PARAMS['BATCH_SIZE'], num_workers=num_workers)

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
        patience=PARAMS["EARLY_STOP_PATIENCE"],    # minimum number of epochs with no improvement
        mode="min",
        min_delta=PARAMS["EARLY_STOP_MIN_DELTA"],  # minimum change to qualify as an improvement
        verbose=True
    )

    run_config = PARAMS.copy()
    # Remove any existing wandb api key
    if "WANDB_API_KEY" in run_config:
        del run_config["WANDB_API_KEY"]
    run_config.update({
        "RUN_ID": run,
        "SEED": seed,
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
        callbacks=[checkpoint_callback, early_stop_callback]
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
        pretrained_dir = os.path.join(ROOT, 'pretrained', 'tcn_carnatic_fs')
        os.makedirs(pretrained_dir, exist_ok=True)
        shutil.copyfile(latest_ckpt, os.path.join(pretrained_dir, f"{ckpt_name}-trainfold{train_fold}-run{run}.ckpt"))
        print(f"Copied checkpoint to: {os.path.join(pretrained_dir, f'{ckpt_name}-trainfold{train_fold}-run{run}.ckpt')}")

    except Exception as e:
        print(f"Warning: Failed to copy checkpoint to the 'pretrained' folder: {e}. You can still find it in the output/checkpoints/<timestamp> folder.")
        print("Training will continue...")

    finally:
        # Clean up the wandb run directory
        wandb_run.finish()
        print("Wandb run finished")

print(f"Training completed successfully!")
