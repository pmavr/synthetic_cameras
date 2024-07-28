import sys
from torch import nn
import numpy as np

from src.models.synthetic_camera_data.dataloaders import LocationDataLoader
from src.models.model import EfficientnetModel
from src.models.trainer import BatchTrainer
import utils.utils as utils

if __name__ == "__main__":
    data_params = {
        "dataset_size": 100,
        "val_percent_size": 0.1,
        "test_percent_size": 0.1,
        "image_w": 1280,
        "image_h": 720,
        "image_output_dimensions": (640, 360),
        "data_normalization_params": {"mean": [0.0188], "std": [0.128]},
        "norm_range": [-1, 1],
        "camera_param_distributions": {
            "camera_center": {
                "mean": [52.36618474, -41.15650112, 16.82156705],
                "std": [1.23192608, 8.3825635, 2.94875254],
                "min": [48.05679141, -59.0702037, 10.13871263],
                "max": [57.84563315, -32.74178234, 23.01126126],
            },
            "focal_length": {
                "mean": 2500.5139785,
                "std": 716.06817106,
                "min": 2100.0,
                "max": 3580.0,
            },
        },
        "fl_density": 30,
        "xloc_density": 0.1,
        "yloc_density": 0.1,
        "zloc_density": 0.1,
        "slope_density": 0.001,
        "pan_std": 3.5,
        "tilt_std": 0.1,
        "data_generation": "images",  # grid | random
    }

    model_params = {
        "phase": "train",
        "model_name": "main",
        "criterion": {
            "method": nn.MSELoss,  # L1Loss, MSELoss
            "params": {"reduction": "mean"},
        },
        "scheduler_params": {"milestones": [5, 10], "gamma": 1e-1},
        "num_workers": 0,
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 15,
        "reshuffle_after_each_epoch": True,
        "num_of_epochs_until_save": 1,
        "save_error_thresh": 100,
        "input_features": 1,
        "num_classes": 6,
        "efficientnet_arch": {"method": "efficientnet-b0", "params": {}},
    }
    print("Training Setup\n-------------")
    print(model_params)
    print(data_params)
    print("-------------")
    dataloader = LocationDataLoader(model_params, data_params)
    dataloader.initialize()
    model = EfficientnetModel(model_params, dataloader)
    model.initialize()

    # Uncomment if you want to continue training existing model

    # model_filename = f"{utils.get_generated_models_path()}test-main_eff-b0_bs32_lr3e-02_st[50]_@10K_e10.pth"
    # model.load_weights_from_file(model_filename, use_saved_optimizer=True)
    # history = model.load_history_from_file(model_filename)
    # model.optimizer.param_groups[0]['lr'] = model_params['lr']

    batch_trainer = BatchTrainer(
        model=model,
        model_params=model_params,
        save_path=utils.get_generated_models_path(),
        verbose=2,
        # , history=history     # Uncomment for to continue training existing model
    )

    print(batch_trainer.init_name_for_saved_model())
    batch_trainer.fit_model()

    metrics = batch_trainer.test_model()

    print(
        f"Error\n fl    |    x    |    y    |    z   \n"
        f"{metrics[0]:.2f} | "
        f"{metrics[1]:.2f} | "
        f"{metrics[2]:.2f} | "
        f"{metrics[3]:.2f}"
    )
    sys.exit()
