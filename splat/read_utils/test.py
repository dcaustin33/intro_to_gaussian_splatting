import os

from splat.read_utils.read_gs_ply_files import read_ply_file

path = "/Users/derek/Downloads/models/treehill/point_cloud/iteration_7000/point_cloud.ply"
vertices = read_ply_file(path)

import pdb; pdb.set_trace()
