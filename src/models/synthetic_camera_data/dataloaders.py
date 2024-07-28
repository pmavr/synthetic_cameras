import numpy as np
from torch.utils.data import DataLoader
import random

from src.models.synthetic_camera_data.location_data import LocationImageDataset

from src.models.synthetic_camera_data.angle_data import (
    AngleGridDataset,
    AngleRandomDataset,
)

from src.modules.Camera import Camera
import src.utils.utils as utils


class CameraDataLoader:

    def __init__(self, model_params, data_params):
        self.model_params = model_params
        self.data_params = data_params
        val_percent_size, test_percent_size = (
            data_params["val_percent_size"],
            data_params["test_percent_size"],
        )
        train_percent_size = 1 - val_percent_size - test_percent_size
        self.num_of_cams = data_params["dataset_size"]
        self.num_of_train_cams = int(self.num_of_cams * train_percent_size)
        self.num_of_val_cams = int(self.num_of_cams * val_percent_size)
        self.num_of_test_cams = int(self.num_of_cams * test_percent_size)
        self.sets = {}

    def initialize(self):
        pass

    def make_dataloaders(self, train_set, val_set, test_set):
        self.sets["train"] = DataLoader(
            train_set,
            batch_size=self.model_params["batch_size"],
            num_workers=self.model_params["num_workers"],
            pin_memory=True,
        )
        self.sets["val"] = DataLoader(
            val_set,
            batch_size=self.model_params["batch_size"],
            num_workers=self.model_params["num_workers"],
            pin_memory=True,
        )
        self.sets["test"] = DataLoader(
            test_set,
            batch_size=1,
            pin_memory=True,
        )

    def reshuffle_train_cameras(self):
        self.initialize()


class LocationDataLoader(CameraDataLoader):

    def __init__(self, model_params, data_params):
        super().__init__(model_params, data_params)
        self.camera_centers = self.generate_chen_camera_centers()
        self.data_generation = data_params["data_generation"]

    def initialize(self):
        cameras = self.camera_centers[
            random.sample(range(len(self.camera_centers)), self.num_of_cams), :
        ]
        train_cams, val_cams, test_cams = self._get_train_val_test_sets(cameras)

        train_dataset = LocationImageDataset(self.data_params, train_cams)
        val_dataset = LocationImageDataset(self.data_params, val_cams)
        test_dataset = LocationImageDataset(self.data_params, test_cams)

        self.make_dataloaders(train_dataset, val_dataset, test_dataset)

    def generate_chen_camera_centers(self):
        cc_statistics = self.data_params["camera_param_distributions"]["camera_center"]
        cc_mean = cc_statistics["mean"]
        cc_std = cc_statistics["std"]
        camera_centers = np.random.normal(cc_mean, cc_std, (self.num_of_cams, 3))
        return camera_centers

    def _get_train_val_test_sets(self, cameras):
        train_cameras = cameras[: self.num_of_train_cams]
        val_test_cameras = cameras[self.num_of_train_cams :]
        val_cameras = val_test_cameras[: self.num_of_val_cams]
        test_cameras = val_test_cameras[self.num_of_val_cams :]
        return train_cameras, val_cameras, test_cameras


class AngleDataLoader(CameraDataLoader):

    def __init__(self, x, y, z, model_params, data_params):
        super().__init__(model_params, data_params)
        self.x = x
        self.y = y
        self.z = z
        self.data_generation = self.data_params["data_generation"]

    def initialize(self):
        if self.data_generation == "grid":
            train_dataset = AngleGridDataset(
                self.x, self.y, self.z, self.data_params, size=self.num_of_train_cams
            )
            val_dataset = AngleGridDataset(
                self.x, self.y, self.z, self.data_params, size=self.num_of_val_cams
            )
            test_dataset = AngleGridDataset(
                self.x, self.y, self.z, self.data_params, size=self.num_of_test_cams
            )
        elif self.data_generation == "random":
            train_dataset = AngleRandomDataset(
                self.x, self.y, self.z, self.data_params, size=self.num_of_train_cams
            )
            val_dataset = AngleRandomDataset(
                self.x, self.y, self.z, self.data_params, size=self.num_of_val_cams
            )
            test_dataset = AngleRandomDataset(
                self.x, self.y, self.z, self.data_params, size=self.num_of_test_cams
            )
        else:
            raise Exception

        self.make_dataloaders(train_dataset, val_dataset, test_dataset)


def tensor2image(image_tensor, imtype=np.uint8):
    images = list(image_tensor.cpu().float().numpy())
    images = [reverse_normalize(im) for im in images]
    images = [im.astype(imtype) for im in images]
    return images


def reverse_normalize(array):
    array = array.squeeze()
    array = (array * 0.128) + 0.0188  # mean and std used from data_transform
    array *= 255
    return array


if __name__ == "__main__":
    from time import time
    import cv2

    location_data_params = {
        "dataset_size": 100000,
        "val_percent_size": 0.01,
        "test_percent_size": 0.01,
        "image_w": 1280,
        "image_h": 720,
        "image_output_dimensions": (640, 360),
        "data_normalization_params": {
            "mean": [0.0188],  # .0288
            "std": [0.128],  # .1607
        },
        "norm_range": [-1, 1],
        "camera_descr": "offside",  # master, offside, high_behind
        "camera_param_distributions": {
            "camera_center": {  # master camera location
                "mean": [52.36618474, -41.54967, 16.82156705],
                "std": [1.23192608, 8.59213, 2.94875254],
                "min": [50.05679141, -66.0702037, 10.13871263],
                "max": [55.84563315, -16.74178234, 23.01126126],
            },
            "focal_length": {
                "mean": 2500.5139785,
                "std": 716.06817106,
                "min": 2100,  # 1463.,
                "max": 3580.0,
            },
            "bleachers_slope": {"min": 0.3, "max": 0.45},
        },
        "fl_density": 30,
        "xloc_density": 0.1,
        "yloc_density": 0.1,
        "zloc_density": 0.1,
        "slope_density": 0.001,
        "pan_std": 1.5,
        "tilt_std": 0.01,
        "data_generation": "images",  # images | lines
    }
    angle_data_params = {
        "dataset_size": 1000,
        "val_percent_size": 0.05,
        "test_percent_size": 0.05,
        "image_w": 1280,
        "image_h": 720,
        "image_output_dimensions": (320, 180),
        "data_normalization_params": {
            "mean": [0.0188],  # .0288
            "std": [0.128],  # .1607
        },
        "norm_range": [-1, 1],
        "player_apparent_size_range": [40, 85.0, 150.0],
        "focal_length_density": 30,
        "tilt_density": 0.1,
        "pan_density": 0.1,
        "mid_extra_tilt": 0.5,
        "side_extra_tilt": 0,
        "penalty_area_bounds": [[21.45920, 83.54135], [13.84, 54.16]],
        "data_generation": "grid",  # grid | random
    }
    model_params = {
        "num_workers": 0,
        "lr": 3e-3,
        "batch_size": 1,
        "epochs": 25,
        "num_of_epochs_until_save": 100,
    }

    # adl = LocationDataLoader(model_params, location_data_params)
    # adl.initialize()

    # x, y, z = 52.47, -43.864, 12.554
    # x, y, z = (52.47, -43.864, 12.554)  # master
    # x, y, z = (19.484, -43.864, 12.554)  # offside left
    # x, y, z = (86.326, -43.864, 12.554)  # offside right
    # x, y, z = (160.76, 18.648, 13.25)  # high behind right
    x, y, z = (88.32, -21.89, -7.92)  # offside right
    # x, y, z = (-35.67, 34.07, 18.82)  # high behind left

    adl = AngleDataLoader(x, y, z, model_params, angle_data_params)
    adl.initialize()
    i = 0
    start = time()
    for edge_map, y_true in adl.sets["train"]:
        images = tensor2image(edge_map)
        # i += 1
        # print(f'{i} - {time() - start:.3f}')
        # start = time()
        # y = y_true.numpy()
        # for i, im in enumerate(images):
        #     cv2.putText(im, f'{y[i, 0]:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255))
        #     cv2.putText(im, f'{y[i, 1]:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255))
        #     cv2.putText(im, f'{y[i, 2]:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (255))
        #     cv2.putText(im, f'{y[i, 3]:.2f}, {y[i, 4]:.2f}, {y[i, 5]:.2f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, (255))
        utils.show_image(images)

    print("gr")
