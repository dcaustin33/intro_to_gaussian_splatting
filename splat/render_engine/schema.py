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
    top_left: torch.Tensor
    bottom_right: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor

    model_config = {
        "arbitrary_types_allowed": True
    }

    def clone(self) -> "PreprocessedGaussian":
        return PreprocessedGaussian(
            means_3d=self.means_3d.clone(),
            covariance_2d=self.covariance_2d.clone(),
            inverted_covariance_2d=self.inverted_covariance_2d.clone(),
            radius=self.radius.clone(),
            tiles_touched=self.tiles_touched.clone(),
            top_left=self.top_left.clone(),
            bottom_right=self.bottom_right.clone(),
            color=self.color.clone(),
            opacity=self.opacity.clone(),
        )

