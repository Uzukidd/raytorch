import torch
import numpy as np
from raytorch.ops.ray_intersect import ray_triangle_intersect

from pytorch3d.structures import Meshes

class LiDAR_base:

    def __init__(self,
                 origin: torch.Tensor,
                 azi_range: list = [0.0, 180.0],
                 polar_range: list = [-24.8, 2.0],
                 azi_res: float = 1.0,
                 polar_num: int = 64,
                 min_range: float = 1.5,
                 max_range: float = 20.0) -> None:
        self.origin = origin
        self.azi_range = azi_range
        self.polar_range = polar_range
        self.azi_res = azi_res
        self.polar_num = polar_num
        
        self.max_range = max_range
        self.min_range = min_range
        
        self.__init_light_direction()

    def scan_triangles(self, meshes: Meshes, 
                       method:str="batch_ray",
                       aabb_test:bool=True):
        vertices = meshes.verts_packed()
        faces = meshes.faces_packed()
        vert_aligned = vertices[faces]
        
        light_directions = self.light_directions
        N = light_directions.size(0)
        origins = self.origin.unsqueeze(dim=0).repeat(N, 1)
        
        aabb = meshes.get_bounding_boxes().squeeze(dim=0)  # [xyz, min + max]
        aabb_mask = origins.new_ones(N).bool()
        if aabb_test:
            aabb_mask = self.__ray_aabb_intersect(origins,
                                      light_directions,
                                      aabb)
        
        intersection = ray_triangle_intersect(origins[aabb_mask],
                                                light_directions[aabb_mask],
                                                vert_aligned,
                                                method=method)
        
        return intersection
        
    def __ray_aabb_intersect(self, ray_origins, ray_directions, aabb):
        """
            Args:
                ray_origins: [N, 3]
                ray_directions: [N, 3]
                aabb: [xyz, min + max] 
        """
        t_min = (aabb[:, 0] - ray_origins) / ray_directions # ([3] - [N, 3]) / [N, 3] -> [N, 3]
        t_max = (aabb[:, 1] - ray_origins) / ray_directions

        tmin = torch.min(t_min, t_max) # [N, 3]
        tmax = torch.max(t_min, t_max) # [N, 3]
        
        tmax = torch.min(tmax, dim=-1).values
        tmin = torch.max(tmin, dim=-1).values

        intersect = torch.logical_and(tmax >= 0, tmin <= tmax)

        return intersect
    
    def __init_light_direction(self):

        light_directions = []
        for azimuthal_angle in np.arange(self.azi_range[0],
                                         self.azi_range[1],
                                         self.azi_res):
            for polar_angle in np.linspace(self.polar_range[0],
                                         self.polar_range[1],
                                         self.polar_num):
                azimuthal_angle_rad = np.radians(azimuthal_angle)
                polar_angle_rad = np.radians(polar_angle)

                x = np.cos(polar_angle_rad) * np.cos(azimuthal_angle_rad)
                y = np.cos(polar_angle_rad) * np.sin(azimuthal_angle_rad)
                z = np.sin(polar_angle_rad)

                light_directions.append([x, y, z])

        self.light_directions = torch.tensor(light_directions).float().to(self.origin.device)


class LiDAR_HDL_64E(LiDAR_base):

    def __init__(self, origin: torch.Tensor, 
                 azi_range: list = [0, 180], 
                 polar_range: list = [80, 100], 
                 res: list = [2.5, 5], 
                 min_range: float = 1.5, 
                 max_range: float = 20) -> None:
        #  TODO
        super().__init__(origin, azi_range, polar_range, res, min_range, max_range)