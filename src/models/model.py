import torch
from torch import nn
import numpy as np

from src.models.networks.efficientnet.model import EfficientNet


class Model(nn.Module):
    def __init__(self, params, dataloader):
        super(Model, self).__init__()
        self.descr = ""
        self.phase = params["phase"]
        self.device = torch.device("cpu")
        self.network = None
        self.dataloader = dataloader
        self.input_features = params["input_features"]
        self.output_features = params["num_classes"]

        if self.phase == "train":
            self.optimizer = None
            self.scheduler = None
            self.lr = params["lr"]

            criterion = params["criterion"]["method"]
            criterion_params = params["criterion"]["params"]

            if criterion is not None:
                self.criterion = criterion(**criterion_params)
            else:
                self.criterion = None

    def initialize(self):
        pass

    def forward(self, x):
        x = x.to(self.device)
        y = self.network(x)
        return y

    def backward(self, y_pred, y_true):
        y_true = y_true.to(self.device)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_loss(self, y_pred, y_true):
        y_true = y_true.to(self.device)
        loss = self.criterion(y_pred, y_true)
        return loss.item()

    def evaluate(self, y_pred, y_true):
        error = np.abs(y_pred - y_true)
        error_mean = np.mean(error, axis=0)
        error_std = np.std(error, axis=0)
        return error_mean, error_std

    def test(self, y_pred, y_true):
        error = np.abs(y_pred - y_true)
        return error

    def predict(self, x):
        x = x.unsqueeze(dim=0)
        with torch.no_grad():
            y_pred = self.forward(x)
        return y_pred

    def schedulers_step(self):
        for i in range(self.n_estimators):
            self.schedulers[i].step()

    def to_device(self, device):
        self.device = device
        self.network = self.network.to(self.device)

        if self.phase == "train":
            self._optimizer_to()
            self.criterion = self.criterion.to(self.device)

    def _optimizer_to(self):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

    def load_weights_from_file(self, filename, use_saved_optimizer=True):
        """Load trained model along with its optimizer and training, plottable history."""
        model_components = torch.load(filename)
        self.network.load_state_dict(model_components["model"])
        if self.phase == "train" and use_saved_optimizer:
            self.optimizer.load_state_dict(model_components["opt_func"])

    @staticmethod
    def load_history_from_file(filename):
        """Load trained model along with its optimizer and training, plottable history."""
        model_components = torch.load(filename)
        history = model_components["history"]
        return history

    def save_to_file(self, filename, history):
        pass


class EfficientnetModel(Model):
    def __init__(self, params, dataloader):
        super(EfficientnetModel, self).__init__(params, dataloader)
        self.net_arch = params["efficientnet_arch"]["method"]
        self.net_params = params["efficientnet_arch"]["params"]
        self.descr = f"eff-{self.net_arch.split('-')[-1]}"

        if self.phase == "train":
            self.scheduler_milestones = params["scheduler_params"]["milestones"]
            self.scheduler_gamma = params["scheduler_params"]["gamma"]

    def initialize(self, from_file=None):
        self.network = EfficientNet.from_name(
            self.net_arch,
            in_channels=self.input_features,
            num_classes=self.output_features,
        )

        if self.phase == "train":
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_milestones,
                gamma=self.scheduler_gamma,
            )
            if from_file is not None:
                self.load_weights_from_file(from_file, use_saved_optimizer=True)

        elif self.phase == "detect":
            self.network.eval()
            if from_file is not None:
                self.load_weights_from_file(from_file)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.to_device(device)

    def save_to_file(self, filename, history):
        model_components = {"model": self.network, "opt_func": self.optimizer}
        meta_data = {
            "label_max": self.dataloader.sets["train"].dataset.label_max,
            "label_min": self.dataloader.sets["train"].dataset.label_min,
            "norm_range": self.dataloader.sets["train"].dataset.norm_range,
        }
        state = {}
        for key in model_components.keys():
            component = model_components[key]
            state[key] = component.state_dict()
        state["meta_data"] = meta_data
        state["history"] = history
        torch.save(state, filename)


class AuxEfficientnetModel(EfficientnetModel):

    def save_to_file(self, filename, history):
        model_components = {"model": self.network, "opt_func": self.optimizer}
        meta_data = {
            "camera_loc": (self.dataloader.x, self.dataloader.y, self.dataloader.z),
            "label_max": self.dataloader.sets["train"].dataset.label_max,
            "label_min": self.dataloader.sets["train"].dataset.label_min,
            "norm_range": self.dataloader.sets["train"].dataset.norm_range,
        }
        state = {}
        for key in model_components.keys():
            component = model_components[key]
            state[key] = component.state_dict()
        state["meta_data"] = meta_data
        state["history"] = history
        torch.save(state, filename)


if __name__ == "__main__":
    import numpy as np
    from time import time

    model_params = {
        "phase": "detect",
        "criterion": {"method": nn.L1Loss, "params": {"reduction": "sum"}},  # MSELoss
        "num_workers": 6,
        "lr": 2e-4,
        "batch_size": 128,
        "epochs": 1,
        "num_of_epochs_until_save": 1000,
        "input_features": 11,
        "num_classes": 4,
        "efficientnet_arch": {"method": "efficientnet-b0", "params": {}},
    }

    model = EfficientnetModel(model_params)
    model.initialize()

    bs = 32
    # x = np.random.rand(bs, 1, 180, 320)
    # y_true = np.random.rand(bs, 4)
    x = np.random.rand(bs, 11)
    y_true = np.random.rand(bs, 4)
    x = torch.tensor(x, dtype=torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.float32)
    start = time()
    y_pred = model.forward(x)
    print(f"Duration: {time() - start:.6f}")
    criterion = nn.L1Loss("mean")
    loss = criterion(y_pred, y_true)

    print("gr")
