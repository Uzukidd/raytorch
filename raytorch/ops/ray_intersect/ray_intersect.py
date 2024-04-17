import numpy as np
import torch

EPS = 0.000001


def test_print():
    print("tesing II")


def ray_triangle_intersect(rays: torch.Tensor, triangles: torch.Tensor):
    """
        Args: 
            rays: [N, 2, 3] -> origin, direction
            triangles: [M, 3, 3]
        Return: 
            intersection: [L, 3]
    """


def ray_triangle_intersect_iter(origins: torch.Tensor,
                                directions: torch.Tensor,
                                triangles: torch.Tensor):
    """
        Args:
            origin: [N, 3]
            direction: [N, 3]
            triangles: [M, 3, 3]
        Return:
            intersection: [L, 3]
    """
    intersection = []
    for light_idx in range(origins.size(0)):
        intersect_depth = torch.Tensor([torch.inf])
        orig = origins[light_idx]
        direction = directions[light_idx]
        for face_idx in range(triangles.size(0)):

            vert = triangles[face_idx]
            t = __ray_triangle_intersect_iter(orig,
                                              direction,
                                              vert)

            if t > 0 and t.item() < intersect_depth.item():
                intersect_depth = t

        if intersect_depth.item() < torch.inf:
            intersection.append(orig + intersect_depth * direction)

    intersection = torch.stack(intersection)
    return intersection


def __ray_triangle_intersect_iter(origin: torch.Tensor, direction: torch.Tensor, triangles: torch.Tensor):
    """
        Args:
            origin: [3]
            direction: [3]
            triangles: [3, 3]
        Return:
            t: [1]
    """
    v0 = triangles[0]
    v1 = triangles[1]
    v2 = triangles[2]

    v0v1 = v1 - v0
    v0v2 = v2 - v0

    pvec = direction.cross(v0v2, dim=-1)

    det = v0v1.dot(pvec)

    if det.item() < EPS:
        return -torch.nan

    invDet = 1.0 / det
    tvec = origin - v0
    u = tvec.dot(pvec) * invDet

    if u.item() < 0 or u.item() > 1:
        return -torch.nan

    qvec = tvec.cross(v0v1, dim=-1)
    v = direction.dot(qvec) * invDet

    if v.item() < 0 or u.item() + v.item() > 1:
        return -torch.nan

    return v0v2.dot(qvec) * invDet
