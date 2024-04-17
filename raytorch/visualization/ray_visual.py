import torch

from raytorch.LiDAR import LiDAR_base

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import RayBundle


def visualize_LiDAR(lidar: LiDAR_base):
    light_directions = lidar.light_directions

    N = light_directions.size(0)
    origins = lidar.origin.unsqueeze(dim=0).repeat(N, 1)
    lengths = torch.ones([N, 2]) * \
        torch.tensor([lidar.min_range, lidar.max_range])
    xys = torch.zeros([N, 2])

    ray_bundle = RayBundle(origins=origins,
                           lengths=lengths,
                           xys=xys,
                           directions=light_directions)

    return ray_bundle


def visualize_point_clouds(pts: torch.Tensor):
    point_clouds = []

    if pts.size().__len__() == 3:
        for mask in range(pts.size(0)):
            point_clouds.append(pts[mask])
    else:
        point_clouds.append(pts)

    point_clouds = Pointclouds(point_clouds)
    return point_clouds
