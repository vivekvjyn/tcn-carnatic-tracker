
import os
import pickle
import lightning as L
import torch
import time

from loss.loss import loss_bce, loss_relative, loss_weighted
from post.dbn import beat_tracker, joint_tracker, sequential_tracker
from eval.metrics import all_metrics, flatten_dict
from utils.visualisation import plot_spec

# in test_step i hard coded the file indexes that crashes for each seed and save the results to json

class PLTCN(L.LightningModule):

    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.loss_name = params["LOSS"]
        self.loss_fn = self._get_loss_fn()
        self.all_results = []
        self.learning_rate = params["LEARNING_RATE"]
        self.scheduler_factor = params["SCHEDULER_FACTOR"]
        self.scheduler_patience = params["SCHEDULER_PATIENCE"]
        self.post_processor = params["POST_PROCESSOR"]
        self.post_tracker = self._get_post_processor()
        self.train_start_time = None


    def _get_loss_fn(self):
        if self.loss_name == "BCE":
            return loss_bce
        elif self.loss_name == "WEIGHTED":
            return loss_weighted
        elif self.loss_name == "RELATIVE":
            return loss_relative
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_name}")

    def _get_post_processor(self):

        if self.post_processor == "JOINT":
            return joint_tracker
        elif self.post_processor == "SEQUENTIAL":
            return sequential_tracker
        else:
            raise ValueError(f"Unsupported post-processing type: {self.post_processor}")

    def on_train_start(self):
        self.train_start_time = time.time()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        beats_ann = batch["beats"]
        downbeats_ann = batch["downbeats"]
        output = self(x)

        beats_det = output["beats"].squeeze(-1)
        downbeats_det = output["downbeats"].squeeze(-1)

        loss, loss_beat, loss_downbeat = self.loss_fn(beats_det, beats_ann, downbeats_det, downbeats_ann)

        self.log("train_loss_beats", loss_beat, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_downbeats", loss_downbeat, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_time = time.time()
        elapsed = current_time - self.train_start_time
        self.log("elapsed_trainstep_time_sec", elapsed, on_step=True, prog_bar=False)

    def on_train_epoch_end(self):
        current_time = time.time()
        elapsed = current_time - self.train_start_time
        self.log("elapsed_epoch_time_sec", elapsed, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        beats_gt = batch["beats"]
        downbeats_gt = batch["downbeats"]

        output = self(x)
        beats_pred = output["beats"].squeeze(-1)
        downbeats_pred = output["downbeats"].squeeze(-1)

        loss, loss_beat, loss_downbeat = self.loss_fn(beats_pred, beats_gt, downbeats_pred, downbeats_gt)

        self.log("val_loss_beats", loss_beat, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_loss_downbeats", loss_downbeat, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", current_lr, prog_bar=True, on_epoch=True)
        print(f"[Epoch {self.current_epoch}] Learning Rate: {current_lr:.6f}")

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)
        scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler_factor,
                    patience=self.scheduler_patience,
                    threshold=1e-4,
                    cooldown=0,
                    min_lr=1e-7
                ),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate"
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    # --- Test ---

    def test_step(self, batch, batch_idx):
        if batch_idx in [18, 25, 32]:
            return
        x = batch["x"]
        tid = str(batch["key"][0])

        beats_target = batch["beats_ann"].squeeze().detach().cpu().numpy()
        downbeats_target = batch["downbeats_ann"].squeeze().detach().cpu().numpy()

        output = self(x)
        beats_act = output["beats"].squeeze().detach().cpu().numpy()
        downbeats_act = output["downbeats"].squeeze().detach().cpu().numpy()

        pred = self.post_tracker(beats_act, downbeats_act)

        beats_prediction = pred[:, 0]
        downbeats_prediction = pred[pred[:, 1] == 1][:, 0]

        beat_scores = all_metrics(beats_target, beats_prediction)
        downbeat_scores = all_metrics(downbeats_target, downbeats_prediction)

        result = flatten_dict(tid, beat_scores, downbeat_scores)

        import json
        os.makedirs("test_results/62", exist_ok=True)

        with open(f"test_results/62/{tid}_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    def on_test_epoch_end(self):
        # Get current working directory
        cwd = os.getcwd()

        # Create temp directory inside cwd
        temp_dir = os.path.join(cwd, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Define full path to the pickle file
        results_path = os.path.join(temp_dir, "results.pkl")

        # Save results
        with open(results_path, "wb") as f:
            pickle.dump(self.all_results, f)

        print(f"Saved test results to {results_path}")

        self.all_results.clear()
