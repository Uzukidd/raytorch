import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

class learnable_meshes:
    
    def __init__(self, meshes: Meshes) -> None:
        self.meshes:Meshes = meshes
        self.deform_parameter: torch.Tensor = \
            torch.zeros_like(meshes.verts_packed(), requires_grad=True)
            
    def get_deformed_meshes(self, transform: Transform3d = None):
        meshes = self.meshes
        if transform is not None:
            temp_verts = transform.transform_points(meshes.verts_padded())
            meshes = meshes.update_padded(temp_verts)
        meshes = meshes.offset_verts(self.deform_parameter)
        
        return meshes
    
    def get_meshes(self) -> Meshes:
        return self.meshes
    
    def update_parameters(self, input: torch.Tensor):
        self.deform_parameter = input.detach().clone()
        self.deform_parameter.requires_grad_(True)
    
    def get_parameters(self) -> torch.Tensor:
        return self.deform_parameter
    
    def get_gradient(self) -> torch.Tensor:
        return self.deform_parameter.grad
