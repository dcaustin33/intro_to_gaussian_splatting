from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


class PreprocessedScene(NamedTuple):
    points: torch.Tensor
    colors: torch.Tensor
    covariance_2d: torch.Tensor
    depths: torch.Tensor
    inverse_covariance_2d: torch.Tensor
    radius: torch.Tensor
    points_xy: torch.Tensor
    min_x: torch.Tensor
    min_y: torch.Tensor
    max_x: torch.Tensor
    max_y: torch.Tensor
    sigmoid_opacity: torch.Tensor
