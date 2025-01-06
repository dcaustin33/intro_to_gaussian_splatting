import numpy as np
import pycolmap
import torch
import torch.nn as nn

from splat.gaussians import Gaussians
from splat.image import GaussianImage
from splat.read_utils.read_gs_ply_files import convert_to_gaussian_schema, read_ply_file
from splat.utils import (
    build_rotation,
    get_extrinsic_matrix,
    getIntrinsicMatrix,
    read_camera_file,
    read_image_file,
    read_images_binary,
)
from splat.test.derivatives import get_matrices

if __name__ == "__main__":
    ply_path = "/Users/derek/Desktop/intro_to_gaussian_splatting/data/truck/point_cloud/iteration_30000/point_cloud.ply"
    vertices = read_ply_file(ply_path)
    gaussians = convert_to_gaussian_schema(vertices[:1])

    colmap_path = "/Users/derek/Desktop/intro_to_gaussian_splatting/colmap_data/truck/sparse/0"
    reconstruction = pycolmap.Reconstruction(colmap_path)

    points3d = reconstruction.points3D
    images = read_images_binary(f"{colmap_path}/images.bin")
    cameras = reconstruction.cameras
    

    camera_dict = read_camera_file(colmap_path)
    image_dict = read_image_file(colmap_path)
    images = {}
    for idx in image_dict.keys():
        image = image_dict[idx]
        camera = camera_dict[image.camera_id]
        image = GaussianImage(camera=camera, image=image)
        images[idx] = image


    image_num = 2
    rotation_matrix = build_rotation(torch.Tensor(image_dict[image_num].qvec).unsqueeze(0))
    translation = torch.Tensor(image_dict[image_num].tvec).unsqueeze(0)
    focal_x, focal_y = camera_dict[image_dict[image_num].camera_id].params[:2] / 2
    focal_x = int(focal_x)
    focal_y = int(focal_y)
    c_x, c_y = camera_dict[image_dict[image_num].camera_id].params[2:4]
    width = camera_dict[image_dict[image_num].camera_id].width / 2
    height = camera_dict[image_dict[image_num].camera_id].height / 2
    
    width = int(width)
    height = int(height)
    extrinsic_matrix = get_extrinsic_matrix(rotation_matrix, translation).T
    intrinsic_matrix = getIntrinsicMatrix(focal_x, focal_y, height, width).T
    
    
