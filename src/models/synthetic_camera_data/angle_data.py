import random
import pandas as pd
import numpy as np
import torch

from src.models.synthetic_camera_data.data import SyntheticDataset
from src.modules.Camera import Camera
import src.utils.utils as utils

np.set_printoptions(suppress=True)


class AngleDataset(SyntheticDataset):

    def __init__(self, x, y, z, params, size):
        super().__init__(params)
        self.x = x
        self.y = y
        self.z = z
        self.pan_density = params["pan_density"]
        self.tilt_density = params["tilt_density"]
        self.focal_length_density = params["focal_length_density"]
        self.min_player_size = params["player_apparent_size_range"][0]
        self.mid_player_size = params["player_apparent_size_range"][1]
        self.max_player_size = params["player_apparent_size_range"][2]
        self.side_extra_tilt = params["side_extra_tilt"]
        self.left_area_bound = params["left_right_area_bounds"][0]
        self.right_area_bound = params["left_right_area_bounds"][1]
        self.size = size

        self.camera_poses = self.generate_ptz_cameras()
        num_of_camera_poses = len(self.camera_poses)
        if self.size is not None and self.size < num_of_camera_poses:
            self.camera_poses = self.camera_poses[
                random.sample(range(num_of_camera_poses), size), :
            ]
        self.num_of_cameras = self.camera_poses.shape[0]

    def generate_ptz_cameras(self):
        pass

    @staticmethod
    def focal_length_limit(distance_from_camera, object_apparent_size):
        actual_height = 1.8
        fl_limit = object_apparent_size * distance_from_camera / actual_height
        return fl_limit.reshape(-1, 1)

    @staticmethod
    def b_max_for_sides(amax, phi):
        return abs(amax / np.sin(np.radians(phi)))

    @staticmethod
    def b_min_for_sides(cmin, phi):
        return abs(cmin / np.cos(np.radians(phi)))

    def pan_range(self):
        min_pan_angle = np.degrees(np.arctan(abs(self.y) / self.x)) - 90
        max_pan_angle = 90 - np.degrees(
            np.arctan(abs(self.y) / (Camera.court_length_x - self.x))
        )
        return min_pan_angle, max_pan_angle

    def __getitem__(self, ndx):
        camera = self.camera_poses[ndx]
        edge_map = self._prepare_item(camera)

        y_true = np.array([camera[2], camera[3], camera[4]])
        y_true = self._normalize_label(y_true)
        y_true = torch.tensor(y_true, dtype=torch.float)
        return edge_map, y_true


class AngleGridDataset(AngleDataset):

    def generate_ptz_cameras(self):
        u, v = (
            self.image_w / 2.0,
            self.image_h / 2.0,
        )
        extremes = self.angle_extremes_for_camera_location()
        tilt_pan_angles = self.generate_data_within_extremes(extremes)

        distances = Camera.distance_from_camera(self.z, tilt_pan_angles[:, 0])

        tilt_angles, pan_angles = tilt_pan_angles[:, 0], tilt_pan_angles[:, 1]
        x, y = Camera.coords_at_distance(self.y, self.z, tilt_angles, pan_angles)

        max_player_size = np.array(
            [
                (
                    self.max_player_size
                    if (0.0 < val < self.left_area_bound)
                    or (self.right_area_bound < val < Camera.court_length_x)
                    else self.mid_player_size
                )
                for val in x
            ]
        )
        max_focal_lengths = self.focal_length_limit(
            distances, object_apparent_size=max_player_size
        )
        min_focal_lengths = self.focal_length_limit(
            distances, object_apparent_size=self.min_player_size
        )

        #         focal_lengths = np.random.uniform(min_focal_lengths, max_focal_lengths)
        focal_length_tilt_pan_data = self.generate_focal_length_tilt_pan_data(
            tilt_pan_angles, min_focal_lengths, max_focal_lengths
        )

        num_of_cameras = focal_length_tilt_pan_data.shape[0]
        roll_angles = np.zeros((num_of_cameras, 1))

        self.label_max = np.concatenate(
            [np.array([max_focal_lengths.max()]), tilt_pan_angles.max(axis=0)], axis=0
        )
        self.label_min = np.concatenate(
            [np.array([max_focal_lengths.min()]), tilt_pan_angles.min(axis=0)], axis=0
        )

        cameras = np.concatenate(
            [
                np.ones((num_of_cameras, 1)) * u,
                np.ones((num_of_cameras, 1)) * v,
                focal_length_tilt_pan_data,
                roll_angles,
                np.ones((num_of_cameras, 1)) * self.x,
                np.ones((num_of_cameras, 1)) * self.y,
                np.ones((num_of_cameras, 1)) * self.z,
            ],
            axis=1,
        )
        np.random.shuffle(cameras)
        return cameras

    def angle_extremes_for_camera_location(self):
        min_pan_angle, max_pan_angle = self.pan_range()
        pan_range = np.arange(min_pan_angle, max_pan_angle, self.pan_density)
        tilt_range = np.zeros((len(pan_range), 2))
        for i, pan_angle in enumerate(pan_range):
            camera_params = np.array(
                [640, 360, 3500.0, -12.5, pan_angle, 0.0, self.x, self.y, self.z]
            )
            camera = Camera(camera_params)
            orientation = camera.orientation()

            if orientation == 0:  # mid case
                theta = abs(90 - pan_angle)
                cmax = abs(self.y) + camera.court_width_y
                cmin = abs(self.y)
                bmax = self.b_for_mid(cmax, theta)
                bmin = self.b_for_mid(cmin, theta)

                max_tilt = self.tilt_angle(self.z, bmax) * (-1)
                min_tilt = self.tilt_angle(self.z, bmin) * (-1)

            elif orientation > 0:  # side case
                x_camera_shift = self.x - camera.court_mid_length_x
                amax = camera.court_mid_length_x + x_camera_shift * np.sign(
                    pan_angle
                ) * (-1)
                cmin = abs(self.y)
                bmax = self.b_max_for_sides(amax, pan_angle)
                bmin = self.b_min_for_sides(cmin, pan_angle)

                max_tilt = self.tilt_angle(self.z, bmax) * (-1) + self.side_extra_tilt
                min_tilt = self.tilt_angle(self.z, bmin) * (-1)

            tilt_range[i, 0] = min_tilt
            tilt_range[i, 1] = max_tilt

        data = np.concatenate([pan_range.reshape(-1, 1), tilt_range], axis=1)
        data = pd.DataFrame(data, columns=["pan", "max_tilt", "min_tilt"])
        data = data[(data["max_tilt"] < 0.0) & (data["min_tilt"] < 0.0)]
        return data

    def generate_data_within_extremes(self, extremes):
        data = []
        for _, row in extremes.iterrows():
            interpolated_tilt_angles = np.arange(
                row["max_tilt"], row["min_tilt"], self.tilt_density
            )
            pan_angle = np.ones_like(interpolated_tilt_angles) * row["pan"]
            batch = np.stack([interpolated_tilt_angles, pan_angle], axis=1)
            data.append(batch)
        data = np.concatenate(data, axis=0)
        return data

    def generate_focal_length_tilt_pan_data(
        self, tilt_pan_angles, min_focal_lengths, max_focal_lengths
    ):
        output = []
        num_of_cameras = len(tilt_pan_angles)
        for i in range(num_of_cameras):
            min_fl, max_fl = min_focal_lengths[i, 0], max_focal_lengths[i, 0]
            interpolated_focal_lengths = np.arange(
                min_fl, max_fl, self.focal_length_density
            ).reshape(-1, 1)
            num_of_interpolated_focal_lengths = len(interpolated_focal_lengths)

            tilt_angles = (
                np.ones((num_of_interpolated_focal_lengths, 1)) * tilt_pan_angles[i, 0]
            )
            pan_angles = (
                np.ones((num_of_interpolated_focal_lengths, 1)) * tilt_pan_angles[i, 1]
            )

            output.append(
                np.concatenate(
                    [interpolated_focal_lengths, tilt_angles, pan_angles], axis=1
                )
            )

        output = np.concatenate(output, axis=0)
        return output


class AngleRandomDataset(AngleDataset):

    def generate_ptz_cameras(self):
        u, v = (
            self.image_w / 2.0,
            self.image_h / 2.0,
        )
        pan_angles = self.generate_random_pan_angles()
        tilt_angles = self.generate_random_tilt_angles(pan_angles)

        distances = Camera.distance_from_camera(self.z, tilt_angles)
        x, y = Camera.coords_at_distance(self.y, self.z, tilt_angles, pan_angles)

        max_player_size = np.array(
            [
                (
                    self.max_player_size
                    if (0.0 < val < self.left_area_bound)
                    or (self.right_area_bound < val < Camera.court_length_x)
                    else self.mid_player_size
                )
                for val in x
            ]
        )
        max_focal_lengths = self.focal_length_limit(
            distances, object_apparent_size=max_player_size.reshape(-1, 1)
        )
        min_focal_lengths = self.focal_length_limit(
            distances, object_apparent_size=self.min_player_size
        )

        focal_lengths = np.random.uniform(min_focal_lengths, max_focal_lengths)
        roll_angles = np.zeros((self.size, 1))

        self.label_max = np.array(
            [max_focal_lengths.max(), tilt_angles.max(), pan_angles.max()]
        )
        self.label_min = np.array(
            [max_focal_lengths.min(), tilt_angles.min(), pan_angles.min()]
        )

        cameras = np.concatenate(
            [
                np.ones((self.size, 1)) * u,
                np.ones((self.size, 1)) * v,
                focal_lengths,
                tilt_angles,
                pan_angles,
                roll_angles,
                np.ones((self.size, 1)) * self.x,
                np.ones((self.size, 1)) * self.y,
                np.ones((self.size, 1)) * self.z,
            ],
            axis=1,
        )
        np.random.shuffle(cameras)
        return cameras

    def generate_random_pan_angles(self):
        min_pan_angle, max_pan_angle = self.pan_range()
        pan_angles = np.random.uniform(min_pan_angle, max_pan_angle, (self.size, 1))
        return pan_angles

    def generate_random_tilt_angles(self, pan_angles):
        tilt_minmax = np.zeros((self.size, 2))

        for i, pan_angle in enumerate(pan_angles):
            camera_params = np.array(
                [640, 360, 3500.0, -12.5, pan_angle, 0.0, self.x, self.y, self.z]
            )
            camera = Camera(camera_params)
            orientation = camera.orientation()

            if orientation == 0:  # mid case
                theta = abs(90 - pan_angle)
                cmax = abs(self.y) + camera.court_width_y
                cmin = abs(self.y)
                bmax = self.b_max_for_mid(cmax, theta)
                bmin = self.b_min_for_mid(cmin, theta)

                tilt_minmax[i, 1] = self.tilt_angle(self.z, bmax) * (-1)
                tilt_minmax[i, 0] = self.tilt_angle(self.z, bmin) * (-1)

            elif orientation > 0:  # side case
                x_camera_shift = self.x - camera.court_mid_length_x
                amax = camera.court_mid_length_x + x_camera_shift * np.sign(
                    pan_angle
                ) * (-1)
                cmin = abs(self.y)
                bmax = self.b_max_for_sides(amax, pan_angle)
                bmin = self.b_min_for_sides(cmin, pan_angle)

                tilt_minmax[i, 1] = (
                    self.tilt_angle(self.z, bmax) * (-1) + self.side_extra_tilt
                )
                tilt_minmax[i, 0] = self.tilt_angle(self.z, bmin) * (-1)

        tilt_angles = np.random.uniform(
            tilt_minmax[:, 0].reshape(-1, 1),
            tilt_minmax[:, 1].reshape(-1, 1),
            (self.size, 1),
        )
        return tilt_angles
