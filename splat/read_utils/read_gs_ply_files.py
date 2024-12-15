import struct
from typing import Dict, List

import torch
import tqdm

from splat.gaussians import Gaussians


def parse_ply_header(file):
    """
    Parses the PLY header to extract metadata and property formats.
    """
    header = []
    properties = []
    vertex_count = 0
    while True:
        line = file.readline().decode("ascii").strip()
        header.append(line)
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("property"):
            _, dtype, name = line.split()
            properties.append((dtype, name))
        elif line == "end_header":
            break
    return vertex_count, properties


def read_binary_vertex_data(file, vertex_count, properties):
    """
    Reads binary vertex data from the PLY file.
    """
    data = []
    # Create a struct format string based on property types
    dtype_map = {
        "float": "f",
        "uchar": "B",  # unsigned char
        "int": "i"
    }
    struct_format = "".join(dtype_map[prop[0]] for prop in properties)
    struct_size = struct.calcsize(struct_format)
    
    # for _ in tqdm.tqdm(range(vertex_count)):
    for _ in tqdm.tqdm(range(1000)):
        binary_data = file.read(struct_size)
        unpacked_data = struct.unpack(struct_format, binary_data)
        data.append(dict(zip([prop[1] for prop in properties], unpacked_data)))
    return data


def read_ply_file(filename):
    """
    Reads a binary PLY file and extracts vertex data.
    """
    with open(filename, "rb") as file:
        # Parse header to get vertex count and properties
        vertex_count, properties = parse_ply_header(file)
        print(f"Vertex Count: {vertex_count}")
        print(f"Properties: {properties}")
        
        # Read vertex data
        vertices = read_binary_vertex_data(file, vertex_count, properties)
    return vertices

def convert_sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    sh_dv = (sh_dc * C0) + 0.5
    
    return torch.clamp(sh_dv, 0, 1)


def convert_to_gaussian_schema(vertices: List[Dict[str, float]]) -> Gaussians:
    gaussian_means = [[v['x'], v['y'], v['z']] for v in vertices]
    gaussian_dc_sh = [[v['f_dc_0'], v['f_dc_1'], v['f_dc_2']] for v in vertices]
    opacity = [v['opacity'] for v in vertices]
    scale = [[v['scale_0'], v['scale_1'], v['scale_2']] for v in vertices]
    rotation = [[v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']] for v in vertices]
    print(gaussian_dc_sh[0])
    colors = convert_sh_to_rgb(torch.tensor(gaussian_dc_sh))
    print(colors[0])
    gaussians = Gaussians(
        points=torch.tensor(gaussian_means),
        colors=convert_sh_to_rgb(torch.tensor(gaussian_dc_sh)),
        scales=torch.tensor(scale),
        quaternions=torch.tensor(rotation),
        opacity=torch.tensor(opacity),
    )
    return gaussians


# Usage
if __name__ == "__main__":
    ply_file = "/Users/derek/Desktop/intro_to_gaussian_splatting/bicycle/point_cloud/iteration_30000/point_cloud.ply"  # Replace with your PLY file path
    vertices = read_ply_file(ply_file)
    print(f"Read {len(vertices)} vertices.")
    
    # Example: Print the first vertex data
    if vertices:
        print("First vertex data:")
        print(vertices[0])