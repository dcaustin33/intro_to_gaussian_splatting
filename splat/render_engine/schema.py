from typing import List
import torch
from pydantic import BaseModel


class ClippingPlanes(BaseModel):
    near: float
    far: float
    right: float
    left: float
    top: float
    bottom: float

class Fov(BaseModel):
    fovX: float
    fovY: float

class PreprocessedGaussian(BaseModel):
    means_3d: torch.Tensor
    covariance_2d: torch.Tensor
    inverted_covariance_2d: torch.Tensor
    radius: torch.Tensor
    tiles_touched: torch.Tensor
    top_left: List[torch.Tensor]
    bottom_right: List[torch.Tensor]
    color: torch.Tensor
    opacity: torch.Tensor

    model_config = {
        "arbitrary_types_allowed": True
    }

