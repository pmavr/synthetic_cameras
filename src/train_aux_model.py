import sys
from torch import nn

from src.models.synthetic_camera_data.dataloaders import AngleDataLoader
from src.models.model import AuxEfficientnetModel
from src.models.trainer import BatchTrainer
import utils.utils as utils
from src.utils.logger import Logger, Experiment

if __name__ == "__main__":
    data_params = {
        "dataset_size": 1000,
        "val_percent_size": 0.05,
        "test_percent_size": 0.05,
        "image_w": 1280,
        "image_h": 720,
        "image_output_dimensions": (320, 180),
        "data_normalization_params": {"mean": [0.0188], "std": [0.128]},
        "norm_range": [-1, 1],
        "player_apparent_size_range": [40.0, 85.0, 120.0],
        "focal_length_density": 30,
        "tilt_density": 0.1,
        "pan_density": 0.1,
        "mid_extra_tilt": 1.0,
        "side_extra_tilt": 2.0,
        "left_right_area_bounds": [21.45920, 83.54135],
        "data_generation": "grid",  # grid | random
    }

    model_params = {
        "phase": "train",
        "model_name": "aux_v3_agrinio_master",
        "criterion": {
            "method": nn.MSELoss,  # L1Loss, MSELoss
            "params": {"reduction": "mean"},
        },
        "scheduler_params": {"milestones": [5, 15], "gamma": 1e-1},
        "num_workers": 6,
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 15,
        "reshuffle_after_each_epoch": True,
        "num_of_epochs_until_save": 1,
        "save_error_thresh": 100,
        "input_features": 1,
        "num_classes": 3,
        "efficientnet_arch": {"method": "efficientnet-b0", "params": {}},
    }
    print("Training Setup\n-------------")
    print(model_params)
    print(data_params)
    print("-------------")

    # x, y, z = 52.59, -44.9, 12.56  # oaka master camera
    # x, y, z = 50.99, -34.19, 11.49  # giannena master camera
    # x, y, z = 52.57, -21.55, 10.95  # smyrni master camera
    # x, y, z = 52.41, -20.63, 8.12  # peristeri master camera
    x, y, z = 52.37, -18.41, 8.91  # agrinio master camera

    dataloader = AngleDataLoader(x, y, z, model_params, data_params)
    dataloader.initialize()

    print(
        f"Train samples {dataloader.sets['train'].dataset.size}\n"
        f"Eval samples {dataloader.sets['val'].dataset.size}\n"
        f"Test samples {dataloader.sets['test'].dataset.size}\n"
    )

    # Uncomment if you want to continue training existing model

    model = AuxEfficientnetModel(model_params, dataloader)
    model.initialize()

    # model_filename = f"{utils.get_generated_models_path()}aux_jpn_main_eff-b1_bs32_lr1e-03_st[5, 15]_res(320, 180)_norm[0, 100]_@100K_e12.pth"
    # model.load_weights_from_file(model_filename)
    # history = model.load_history_from_file(model_filename)
    # model.optimizer.param_groups[0]['lr'] = 1e-6

    trainer = BatchTrainer(
        model=model,
        model_params=model_params,
        save_path=utils.get_generated_models_path(),
        verbose=2,
        # , history=history
    )
    model_filename = trainer.init_name_for_saved_model()
    print(model_filename)
    print(
        f"Train samples {dataloader.sets['train'].dataset.size}\n"
        f"Eval samples {dataloader.sets['val'].dataset.size}\n"
        f"Test samples {dataloader.sets['test'].dataset.size}\n"
    )
    trainer.fit_model()

    logger = Logger(log_filepath=utils.get_generated_models_path())
    exp = Experiment(model_filename, model_params, data_params, trainer.history)
    logger.log_experiment(exp)

    metrics = trainer.test_model()

    print(
        f"Error\n fl    |    tilt    |    pan   \n"
        f"{metrics[0]:.2f} | "
        f"{metrics[1]:.2f} | "
        f"{metrics[2]:.2f}\n==================================================\n"
    )
    sys.exit()
