import os

import cv2
import numpy as np
import pycolmap
import torch
import torch.cuda
from PIL import Image
from tqdm import tqdm

from splat.image import GaussianImage
from splat.read_utils.read_gs_ply_files import convert_to_gaussian_schema, read_ply_file
from splat.render_engine.gaussianScene2 import GaussianScene2
from splat.utils import (
    build_rotation,
    get_extrinsic_matrix,
    getIntrinsicMatrix,
    read_camera_file,
    read_image_file,
    read_images_binary,
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
    intrinsic_matrix = getIntrinsicMatrix(focal_x, focal_y, height, width).T

    return (
        extrinsic_matrix,
        intrinsic_matrix,
        height,
        width,
        focal_x,
        focal_y,
        image_dict[image_num].name,
    )


def get_image(path_to_images, image_name, height, width):
    image = Image.open(os.path.join(path_to_images, image_name))
    image = image.resize((width, height))
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image / 255.0
    image = image.cuda()
    return image


def save_image(image, path):
    # import pdb; pdb.set_trace()
    image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


if __name__ == "__main__":
    output_image_path = "/home/da2986/intro_to_gaussian_splatting/output_images"
    path_to_images = "/home/da2986/gaussian-splatting/tandt/truck/images"
    stem = "/home/da2986/intro_to_gaussian_splatting"
    models_path = "/home/da2986/intro_to_gaussian_splatting/models"
    ply_path = f"/home/da2986/gaussian-splatting/truck/point_cloud/iteration_7000/point_cloud.ply"
    vertices = read_ply_file(ply_path)
    gaussians = convert_to_gaussian_schema(vertices, requires_grad=True)

    colmap_path = f"{stem}/data/treehill/sparse/0"
    colmap_path = f"/home/da2986/gaussian-splatting/tandt/truck/sparse/0"
    reconstruction = pycolmap.Reconstruction(colmap_path)

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
    IMAGE_NUMBER = 1

    extrinsic_matrix, intrinsic_matrix, height, width, focal_x, focal_y, image_name = (
        get_image_info(IMAGE_NUMBER, image_dict, camera_dict)
    )
    image = get_image(path_to_images, image_name, height, width)
    save_image(image, f"gt_image.png")

    params = scene.get_params()

    optimizer = torch.optim.Adam(params, lr=0.001)

    for i in tqdm(range(1000)):
        processed_gaussians = scene.preprocess(
            extrinsic_matrix=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
            focal_x=focal_x,
            focal_y=focal_y,
            width=width,
            height=height,
            tile_size=TILE_SIZE,
        )
        output_image = scene.render_cuda(
            preprocessed_gaussians=processed_gaussians,
            height=height,
            width=width,
            tile_size=TILE_SIZE,
        )

        l1_loss = torch.nn.L1Loss()(output_image, image)
        optimizer.zero_grad()
        l1_loss.backward()
        optimizer.step()
        print(f"L1 loss: {l1_loss}")
        if i % 100 == 0:
            save_image(output_image.detach().cpu(), f"generated_image_{i}.png")
    gaussians.save_params()
