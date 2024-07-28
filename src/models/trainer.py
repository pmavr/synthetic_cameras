from time import time

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np


class BaseTrainer:

    def __init__(self, model, model_params, save_path="", verbose=True):
        self.model = model
        self.params = model_params
        self.epochs = model_params["epochs"]
        self.num_of_epochs_until_save = model_params["num_of_epochs_until_save"]
        self.batch_size = model_params["batch_size"]
        self.dataloaders = model.dataloader
        self.save_path = save_path
        self.model_name = model_params["model_name"]
        self.model_version = 0
        self.verbose = verbose
        self.history = {}
        self.reshuffle_after_each_epoch = model_params["reshuffle_after_each_epoch"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to_device(self.device)

    def name_for_saved_model(self):
        def human_format(num):
            num = float("{:.3g}".format(num))
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return "{}{}".format(
                "{:f}".format(num).rstrip("0").rstrip("."),
                ["", "K", "M", "B", "T"][magnitude],
            )

        name = f"{self.model_name}_{self.model.descr}"
        res = f"res{self.dataloaders.data_params['image_output_dimensions']}"
        norm_f = f"norm{self.dataloaders.sets['train'].dataset.norm_range}"
        epochs = f"e{self.model_version}"
        lr = f"lr{self.params['lr']:.0e}"
        bs = f"bs{self.params['batch_size']}"
        sched_steps = f"st{self.params['scheduler_params']['milestones']}"
        num_of_samples = self.dataloaders.data_params["dataset_size"]
        samples = f"@{human_format(num_of_samples)}"
        total_error = f"sc{round(self.history['total_error'][-1], 2)}"
        name = f"{name}_{bs}_{lr}_{sched_steps}_{res}_{norm_f}_{samples}_{epochs}_{total_error}"
        return name

    def init_name_for_saved_model(self):
        def human_format(num):
            num = float("{:.3g}".format(num))
            magnitude = 0
            while abs(num) >= 1000:
                magnitude += 1
                num /= 1000.0
            return "{}{}".format(
                "{:f}".format(num).rstrip("0").rstrip("."),
                ["", "K", "M", "B", "T"][magnitude],
            )

        name = f"{self.model_name}_{self.model.descr}"
        res = f"res{self.dataloaders.data_params['image_output_dimensions']}"
        norm_f = f"norm{self.dataloaders.sets['train'].dataset.norm_range}"
        epochs = f"e{self.model_version}"
        lr = f"lr{self.params['lr']:.0e}"
        bs = f"bs{self.params['batch_size']}"
        sched_steps = f"st{self.params['scheduler_params']['milestones']}"
        num_of_samples = self.dataloaders.data_params["dataset_size"]
        samples = f"@{human_format(num_of_samples)}"
        name = f"{name}_{bs}_{lr}_{sched_steps}_{res}_{norm_f}_{samples}"
        return name


class BatchTrainer(BaseTrainer):

    def __init__(self, model, model_params, save_path="", verbose=True, history=None):
        super().__init__(model, model_params, save_path, verbose)
        if history is None:
            self.history["train_loss"] = []
            self.history["val_loss"] = []
            self.history["mean_error"] = []
            self.history["error_std"] = []
            self.history["total_error"] = []
            self.num_of_trained_epochs = 0
        else:
            self.history["train_loss"] = history["train_loss"]
            self.history["val_loss"] = history["val_loss"]
            self.history["mean_error"] = history["mean_error"]
            self.history["error_std"] = history["error_std"]
            self.history["total_error"] = history["total_error"]
            self.num_of_trained_epochs = len(history["train_loss"])
        self.epochs += self.num_of_trained_epochs

    def fit_model(self):

        for epoch in range(self.num_of_trained_epochs, self.epochs):
            epoch_start_time = time()
            self._train_step()
            self._evaluation_step()

            epoch_duration = time() - epoch_start_time
            if self.verbose >= 2:
                print(
                    f"Epoch {epoch + 1}/{self.epochs} "
                    f"| Train Loss: {self.history['train_loss'][-1]:.3f}"
                    f"| Val Loss : {self.history['val_loss'][-1]:.3f}"
                    f"| Mean Error : {str(np.round(self.history['mean_error'][-1], 2))}"
                    f"| Error StD : {str(np.round(self.history['error_std'][-1], 2))}"
                    f"| Total Error : {str(np.round(self.history['total_error'][-1], 2))}"
                )

            self.model_version = len(self.history[next(iter(self.history))])
            if (epoch + 1) % self.num_of_epochs_until_save == 0:
                total_error = self.history["total_error"][-1]
                if self.params["save_error_thresh"] > total_error:
                    self.save_model(
                        f"{self.save_path}{self.name_for_saved_model()}.pth"
                    )

            if self.model.scheduler is not None:
                self.model.scheduler.step()

            if self.reshuffle_after_each_epoch and epoch < (self.epochs - 1):
                self.dataloaders.reshuffle_train_cameras()

        if self.params["save_error_thresh"] > total_error:
            self.save_model(f"{self.save_path}{self.name_for_saved_model()}.pth")

    def _train_step(self):
        self.model.train()
        train_loss = 0.0
        data_loader = self.dataloaders.sets["train"]

        for features, labels in tqdm(data_loader, leave=False, desc="Training..."):
            predictions = self.model(features)
            train_loss += self.model.backward(predictions, labels) / len(data_loader)

        self.history["train_loss"].append(train_loss)

    def _evaluation_step(self):
        data_loader = self.dataloaders.sets["val"]
        num_of_batches = len(data_loader)
        val_loss = 0.0
        total_mean_error = np.zeros((num_of_batches, self.model.output_features))
        total_error_std = np.zeros((num_of_batches, self.model.output_features))
        total_error = np.zeros(num_of_batches)

        with torch.no_grad():
            self.model.eval()
            for i, (features, labels) in enumerate(
                tqdm(data_loader, leave=False, desc="Evaluating...")
            ):
                predictions = self.model(features)
                val_loss += self.model.get_loss(predictions, labels) / num_of_batches

                predictions = data_loader.dataset._denormalize_label(predictions)
                labels = data_loader.dataset._denormalize_label(labels)

                mean_error, error_std = self.model.evaluate(predictions, labels)
                total_mean_error[i] = mean_error
                total_error_std[i] = error_std
                total_error[i] = np.sum(mean_error[1:]) + np.sum(error_std[1:])

            self.history["val_loss"].append(val_loss)
            self.history["mean_error"].append(np.mean(total_mean_error, axis=0))
            self.history["error_std"].append(np.mean(total_error_std, axis=0))
            self.history["total_error"].append(np.mean(total_error, axis=0))

    def test_model(self):
        data_loader = self.dataloaders.sets["test"]
        num_of_batches = len(data_loader)
        total_error = np.zeros((num_of_batches, self.model.output_features))

        with torch.no_grad():
            self.model.eval()
            for i, (features, labels) in enumerate(
                tqdm(data_loader, leave=False, desc="Testing...")
            ):
                predictions = self.model(features)

                predictions = data_loader.dataset._denormalize_label(predictions)
                labels = data_loader.dataset._denormalize_label(labels)

                error = self.model.test(predictions, labels)
                total_error[i] = error

            total_mean_error = np.mean(total_error, axis=0)

        return total_mean_error

    def save_model(self, filename):
        """Save trained model along with its optimizer and training, plottable history."""
        self.model.save_to_file(filename, self.history)
