import torch
import numpy as np
from raytorch.ops.ray_intersect import ray_triangle_intersect

from pytorch3d.structures import Meshes

class LiDAR_base:

    def __init__(self,
                 origin: torch.Tensor,
                 azi_range: list = [0.0, 180.0],
                 polar_range: list = [80.0, 100.0],
                 res: list = [2.5, 5.0],
                 min_range: float = 1.5,
                 max_range: float = 20.0) -> None:
        self.origin = origin
        self.azi_range = azi_range
        self.polar_range = polar_range
        self.res = res
        
        self.max_range = max_range
        self.min_range = min_range
        
        self.__init_light_direction()

    def __init_light_direction(self):

        light_directions = []
        for azimuthal_angle in np.arange(self.azi_range[0], 
                                         self.azi_range[1], 
                                         self.res[0]):
            for polar_angle in np.arange(self.polar_range[0],
                                          self.polar_range[1],
                                          self.res[1]):
                azimuthal_angle_rad = np.radians(azimuthal_angle)
                polar_angle_rad = np.radians(polar_angle)

                x = np.sin(polar_angle_rad) * np.cos(azimuthal_angle_rad)
                y = np.sin(polar_angle_rad) * np.sin(azimuthal_angle_rad)
                z = np.cos(polar_angle_rad)

                light_directions.append([x, y, z])
                
        self.light_directions = torch.tensor(light_directions).float()
        

    def scan_triangles(self, meshes: Meshes, method:str="single_ray"):
        vertices = meshes.verts_packed()
        faces = meshes.faces_packed()
        vert_aligned = vertices[faces]
        
        light_directions = self.light_directions
        N = light_directions.size(0)
        origins = self.origin.unsqueeze(dim=0).repeat(N, 1)
        
        intersection = ray_triangle_intersect(origins,
                                                   light_directions,
                                                   vert_aligned,
                                                   method=method)
        
        return intersection

class LiDAR_HDL_64E(LiDAR_base):

    def __init__(self, origin: torch.Tensor, 
                 azi_range: list = [0, 180], 
                 polar_range: list = [80, 100], 
                 res: list = [2.5, 5], 
                 min_range: float = 1.5, 
                 max_range: float = 20) -> None:
        #  TODO
        super().__init__(origin, azi_range, polar_range, res, min_range, max_range)