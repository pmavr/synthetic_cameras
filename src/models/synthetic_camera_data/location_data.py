import numpy as np
import torch

from src.models.synthetic_camera_data.data import SyntheticDataset
from src.modules.Camera import Camera
import src.utils.utils as utils

np.set_printoptions(suppress=True)


class LocationDataset(SyntheticDataset):

    def __init__(self, params, camera_centers):
        super().__init__(params)
        self.camera_centers = camera_centers
        self.num_of_cameras = len(camera_centers)

        camera_distr = params["camera_param_distributions"]
        self.label_max = np.array(camera_distr["camera_center"]["max"])
        self.label_min = np.array(camera_distr["camera_center"]["min"])

        self.camera_poses = self.generate_ptz_cameras()
        self.num_of_cameras = self.camera_poses.shape[0]
        # print(f'Search space size:{self.num_of_cameras} camera poses')

    def generate_ptz_cameras(self):
        u, v = (
            self.image_w / 2.0,
            self.image_h / 2.0,
        )

        pan_angles = self.get_pan_angle() * (-1)
        tilt_angles = self.get_tilt_angle_to_center() * (-1)
        roll_angles = np.zeros((self.num_of_cameras, 1))

        pan_angles = self.get_nearby_pan_data(pan_angles, std=self.params["pan_std"])
        tilt_angles = self.get_nearby_tilt_data(
            tilt_angles, std=self.params["tilt_std"]
        )

        focal_lengths = self.get_focal_lengths(u, v, tilt_angles, pan_angles)

        self.label_min = np.concatenate(
            [
                np.array([focal_lengths.min(), tilt_angles.min(), pan_angles.min()]),
                self.label_min,
            ]
        )
        self.label_max = np.concatenate(
            [
                np.array([focal_lengths.max(), tilt_angles.max(), pan_angles.max()]),
                self.label_max,
            ]
        )

        cameras = np.concatenate(
            [
                np.ones((self.num_of_cameras, 1)) * u,
                np.ones((self.num_of_cameras, 1)) * v,
                focal_lengths,
                tilt_angles,
                pan_angles,
                roll_angles,
                self.camera_centers,
            ],
            axis=1,
        )

        return cameras

    def get_pan_angle(self):
        x = self.camera_centers[:, 0]
        y = self.camera_centers[:, 1]
        a = x - self.court_mid_distance_x
        b = np.abs(y) + self.court_mid_distance_y
        angle = np.arctan(a / b)
        return np.degrees(angle).reshape(-1, 1)

    def get_tilt_angle_to_center(self):
        z = self.camera_centers[:, 2]
        y = self.camera_centers[:, 1]
        a = np.abs(z)
        b = np.abs(y) + self.court_mid_distance_y
        angle = np.arctan(a / b)
        return np.degrees(angle).reshape(-1, 1)

    @staticmethod
    def get_nearby_pan_data(d, std):
        std = np.random.uniform(-std, std, (d.size, 1))
        return d + std

    @staticmethod
    def get_nearby_tilt_data(d, std):
        std = np.random.uniform(-std * 85, std * 85, (d.size, 1))
        return d + std

    def get_focal_lengths(self, u, v, tilt_angles, pan_angles):
        camera_distr = self.params["camera_param_distributions"]
        x = self.camera_centers[:, 0]
        y = self.camera_centers[:, 1]
        z = self.camera_centers[:, 2]
        theta = abs(90 - pan_angles.squeeze())

        cmax = abs(y) + Camera.court_width_y
        cmin = abs(y)
        bmax = self.b_for_mid(cmax, theta)
        bmin = self.b_for_mid(cmin, theta)

        max_tilt = self.get_tilt_angle(z, bmax) * (-1)
        min_tilt = self.get_tilt_angle(z, bmin) * (-1)

        tilt_lower_min_focal_length = v / np.tan(
            np.radians(np.abs(tilt_angles.squeeze() - min_tilt))
        )
        tilt_upper_min_focal_length = v / np.tan(
            np.radians(np.abs(tilt_angles.squeeze() - max_tilt))
        )

        max_focal_length = np.minimum(
            tilt_lower_min_focal_length, tilt_upper_min_focal_length
        )
        min_focal_length = (
            np.ones(len(max_focal_length)) * camera_distr["focal_length"]["min"]
        )
        max_fl_is_less_than_min_fl = np.less(max_focal_length, min_focal_length)
        min_focal_length = np.where(
            max_fl_is_less_than_min_fl, max_focal_length, min_focal_length
        )

        focal_lengths = np.random.uniform(min_focal_length, max_focal_length)
        return focal_lengths.reshape(-1, 1)

    def __getitem__(self, ndx):
        pass


class LocationImageDataset(LocationDataset):

    def __getitem__(self, ndx):
        camera = self.camera_poses[ndx]
        edge_map = self._prepare_item(camera)

        y_true = np.array(
            [camera[2], camera[3], camera[4], camera[6], camera[7], camera[8]]
        )
        y_true = self._normalize_label(y_true)
        y_true = torch.tensor(y_true, dtype=torch.float)
        return edge_map, y_true
