import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from src.modules.Camera import Camera
import src.utils.utils as utils

np.set_printoptions(suppress=True)


class SyntheticDataset(Dataset):

    def __init__(self, params):
        self.params = params
        self.binary_court = utils.get_court_template()
        self.court_mid_distance_x = 52.500276
        self.court_mid_distance_y = 34.001964
        self.image_w = 1280
        self.image_h = 720
        self.image_output_dim = params["image_output_dimensions"]

        self.num_of_cameras = 0
        self.label_min = np.zeros(3)
        self.label_max = np.zeros(3)
        self.norm_range = params["norm_range"]
        normalization_params = params["data_normalization_params"]
        self.data_transform = Compose([ToTensor(), Normalize(**normalization_params)])

    @staticmethod
    def b_for_mid(c, theta):
        return abs(c / np.sin(np.radians(theta)))

    @staticmethod
    def tilt_angle(z, y):
        angle = np.arctan(np.abs(z) / y)
        return np.degrees(angle)

    def _normalize_label(self, label):
        t_min, t_max = self.norm_range
        return (label - self.label_min) / (self.label_max - self.label_min) * (
            t_max - t_min
        ) + t_min

    def _denormalize_label(self, norm_label):
        if isinstance(norm_label, torch.Tensor):
            norm_label = norm_label.detach().cpu().numpy().squeeze()
        t_min, t_max = self.norm_range
        return (norm_label - t_min) / (t_max - t_min) * (
            self.label_max - self.label_min
        ) + self.label_min

    @staticmethod
    def apply_pix2pix_effect(edge_map):
        img = np.copy(edge_map)
        img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
        img = cv2.blur(img, (5, 5))
        return img

    def _prepare_item(self, item):
        item = Camera(item).to_edge_map(self.binary_court)
        item = self.apply_pix2pix_effect(item)
        item = cv2.resize(item, self.image_output_dim)
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
        item = self.data_transform(item)
        return item

    def __len__(self):
        return self.num_of_cameras
