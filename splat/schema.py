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
    min_x_tiles: torch.Tensor
    min_y_tiles: torch.Tensor
    max_x_tiles: torch.Tensor
    max_y_tiles: torch.Tensor
    opacity: torch.Tensor