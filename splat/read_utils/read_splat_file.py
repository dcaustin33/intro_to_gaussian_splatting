import argparse
import struct

import numpy as np


def read_splat_file(splat_file_path):
    """
    Reads a .splat file and extracts its data into a structured format.

    Args:
        splat_file_path (str): Path to the .splat file to read.

    Returns:
        list[dict]: A list of points, where each point is a dictionary with the following fields:
            - position (np.array): x, y, z coordinates
            - scales (np.array): scale_x, scale_y, scale_z
            - color (np.array): r, g, b, alpha values
            - rotation (np.array): normalized quaternion
    """
    points = []
    with open(splat_file_path, "rb") as f:
        while True:
            # Read position (x, y, z)
            position_bytes = f.read(12)  # 3 * 4 bytes (float32)
            if len(position_bytes) < 12:
                break
            position = np.frombuffer(position_bytes, dtype=np.float32)

            # Read scales (scale_x, scale_y, scale_z)
            scales_bytes = f.read(12)  # 3 * 4 bytes (float32)
            scales = np.frombuffer(scales_bytes, dtype=np.float32)

            # Read color (r, g, b, alpha)
            color_bytes = f.read(4)  # 4 * 1 byte (uint8)
            color = np.frombuffer(color_bytes, dtype=np.uint8) / 255.0
            colors = color[:3]
            opacity = color[3]
            # inverse sigmoid
            opacity = np.log(opacity / (1 - opacity)).clip(-30, 30)

            # Read rotation quaternion (rot_0, rot_1, rot_2, rot_3)
            rotation_bytes = f.read(4)  # 4 * 1 byte (uint8)
            rotation = (np.frombuffer(rotation_bytes, dtype=np.uint8) - 128) / 128.0
            rotation /= np.linalg.norm(rotation)  # Normalize quaternion
            

            # Store point data
            points.append({
                "position": position,
                "scales": scales,
                "color": colors,
                "rotation": rotation,
                "opacity": opacity
            })

    return points


def main():
    parser = argparse.ArgumentParser(description="Read and parse SPLAT files.")
    parser.add_argument("--input_file", help="The input SPLAT file to read.", default="bridal-dress.splat")
    args = parser.parse_args()

    print(f"Reading {args.input_file}...")
    points = read_splat_file(args.input_file)

    print(f"Read {len(points)} points from {args.input_file}.")
    print("Example point data:")
    for key, value in points[0].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()