import os
from typing import Dict, Tuple

import pycolmap
import torch
from torch import nn

from splat.gaussians import Gaussians
from splat.image import GaussianImage
from splat.utils import read_camera_file, read_image_file


class GaussianScene(nn.Module):

    def __init__(
        self,
        colmap_path: str,
        gaussians: Gaussians,
    ) -> None:
        super().__init__()

        camera_dict = read_camera_file(colmap_path)
        image_dict = read_image_file(colmap_path)
        self.images = {}
        for idx in image_dict.keys():
            image = image_dict[idx]
            camera = camera_dict[image.camera_id]
            image = GaussianImage(camera=camera, image=image)
            self.images[idx] = image

        self.gaussians = gaussians

    def render_points_image(self, image_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function that can be gotten rid of once we know
        implementation is correct
        """
        return self.images[image_idx].project_point_to_camera(
            self.gaussians.points, self.gaussians.colors
        )
    
    def render_points_image2(self, image_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function that can be gotten rid of once we know
        implementation is correct
        """
        return self.images[image_idx].project_point_to_camera_authors(
            self.gaussians.points, self.gaussians.colors
        )
