import numpy as np
import pycolmap
import torch
import torch.nn as nn
import torch.cuda

from splat.gaussians import Gaussians
from splat.image import GaussianImage
from splat.read_utils.read_gs_ply_files import convert_to_gaussian_schema, read_ply_file
from splat.render_engine.gaussianScene2 import GaussianScene2
from splat.utils import (
    build_rotation,
    get_extrinsic_matrix,
    getIntinsicMatrix,
    read_camera_file,
    read_image_file,
    read_images_binary,
    read_images_text,
)


def get_image_info(image_num, image_dict, camera_dict):
    rotation_matrix = build_rotation(
        torch.Tensor(image_dict[image_num].qvec).unsqueeze(0)
    )
    translation = torch.Tensor(image_dict[image_num].tvec).unsqueeze(0)
    extrinsic_matrix = get_extrinsic_matrix(rotation_matrix, translation).T
    focal_x, focal_y = camera_dict[image_dict[image_num].camera_id].params[:2] / 2
    focal_x = int(focal_x)
    focal_y = int(focal_y)
    c_x, c_y = camera_dict[image_dict[image_num].camera_id].params[2:4]
    width = camera_dict[image_dict[image_num].camera_id].width / 2
    width = int(width)
    height = camera_dict[image_dict[image_num].camera_id].height / 2
    height = int(height)
    intrinsic_matrix = getIntinsicMatrix(focal_x, focal_y, height, width).T

    return extrinsic_matrix, intrinsic_matrix, height, width, focal_x, focal_y


if __name__ == "__main__":
    stem = "/home/da2986/intro_to_gaussian_splatting"
    models_path = "/home/da2986/intro_to_gaussian_splatting/models"
    ply_path = f"/home/da2986/gaussian-splatting/truck/point_cloud/iteration_30000/point_cloud.ply"
    vertices = read_ply_file(ply_path)
    gaussians = convert_to_gaussian_schema(vertices)

    colmap_path = f"{stem}/data/treehill/sparse/0"
    colmap_path = f"/home/da2986/gaussian-splatting/tandt/truck/sparse/0"
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

    scene = GaussianScene2(gaussians=gaussians)
    scene.device = "cuda"
    TILE_SIZE = 16

    total_preprocess_time = 0
    total_render_time = 0

    for image_num in images.keys():
        extrinsic_matrix, intrinsic_matrix, height, width, focal_x, focal_y = (
            get_image_info(image_num, image_dict, camera_dict)
        )

        # Start preprocessing timer
        torch.cuda.synchronize()
        preprocess_start = torch.cuda.Event(enable_timing=True)
        preprocess_end = torch.cuda.Event(enable_timing=True)
        preprocess_start.record()

        processed_gaussians = scene.preprocess(
            extrinsic_matrix=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
            focal_x=focal_x,
            focal_y=focal_y,
            width=width,
            height=height,
            tile_size=TILE_SIZE
        )

        preprocess_end.record()
        torch.cuda.synchronize()
        preprocess_time = preprocess_start.elapsed_time(preprocess_end)
        total_preprocess_time += preprocess_time

        with torch.no_grad():
            # Start rendering timer
            torch.cuda.synchronize()
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            render_start.record()

            output_image, starting_indices, final_tile_indices, array_indices, array = scene.render_cuda(
                preprocessed_gaussians=processed_gaussians, height=height, width=width, tile_size=TILE_SIZE
            )

            render_end.record()
            torch.cuda.synchronize()
            render_time = render_start.elapsed_time(render_end)
            total_render_time += render_time

    print(f"Total preprocessing time: {total_preprocess_time:.2f} ms")
    print(f"Total rendering time: {total_render_time:.2f} ms")
