from typing import List
import torch
from pydantic.dataclasses import dataclass


@dataclass
class ClippingPlanes:
    near: float
    far: float
    right: float
    left: float
    top: float
    bottom: float

@dataclass
class Fov:
    fovX: float
    fovY: float
    
@dataclass
class PreprocessedGaussian:
    means_3d: torch.Tensor
    covariance_2d: torch.Tensor
    inverted_covariance_2d: torch.Tensor
    radius: torch.Tensor
    tiles_touched: torch.Tensor
    top_left: List[torch.Tensor]
    bottom_right: List[torch.Tensor]
    color: torch.Tensor
    opacity: torch.Tensor
