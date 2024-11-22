import numpy as np
import torch

from splat.gaussians import Gaussians
from splat.gaussian_scene import GaussianScene
from splat.read_splat_file import read_splat_file


def separate_points(points: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    scales = []
    colors = []
    rotations = []
    opacities = []

    for point in points:
        positions.append(point["position"])
        scales.append(point["scales"])
        colors.append(point["color"])
        rotations.append(point["rotation"])
        opacities.append(point["opacity"])
        
    return np.array(positions, dtype=np.float32), np.array(scales, dtype=np.float32), np.array(colors, dtype=np.float32), np.array(rotations, dtype=np.float32), np.array(opacities, dtype=np.float32)

if __name__ == "__main__":
    SPLAT_FILE = "/Users/derek/Desktop/intro_to_gaussian_splatting/bridal-dress.splat"
    points = read_splat_file(SPLAT_FILE)
    positions, scales, colors, rotations, opacities = separate_points(points)
    gaussians = Gaussians(
        points=torch.from_numpy(positions),
        colors=torch.from_numpy(colors),
        scales=torch.from_numpy(scales),
        quaternions=torch.from_numpy(rotations),
        opacity=torch.from_numpy(opacities),
    )
    scene = GaussianScene(gaussians)