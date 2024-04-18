import numpy as np
import torch

EPS = 0.000001


def test_print():
    print("tesing II")


def ray_triangle_intersect(origins: torch.Tensor,
                           directions: torch.Tensor,
                           triangles: torch.Tensor,
                           method="single_ray"):
    """
        Args: 
            rays: [N, 2, 3] -> origin, direction
            triangles: [M, 3, 3]
        Return: 
            intersection: [L, 3]
    """
    if method == "single_ray":
        return ray_triangle_intersect_single_ray(origins,
                                        directions,
                                        triangles)
    
    if method == "iter":
        return ray_triangle_intersect_iter(origins,
                                        directions,
                                        triangles)
        
    raise NotImplementedError

def ray_triangle_intersect_single_ray(origins: torch.Tensor,
                                        directions: torch.Tensor,
                                        triangles: torch.Tensor):
    intersection = []
    for light_idx in range(origins.size(0)):
        orig = origins[light_idx]
        direction = directions[light_idx]
        t, drop_mask = __ray_triangle_intersect_single_ray(orig,
                                                direction, 
                                                triangles)
        intersect_depth = t[~drop_mask]
        if intersect_depth.size(0) != 0:
            intersection.append(orig + intersect_depth.min() * direction)
    if intersection.__len__() != 0:
        intersection = torch.stack(intersection)
    else:
        intersection = torch.tensor([])
    return intersection

def __ray_triangle_intersect_single_ray(origins: torch.Tensor,
                                directions: torch.Tensor,
                                triangles: torch.Tensor):
    """
        Args:
            origin: [3]
            direction: [3]
            triangles: [M, 3, 3]
        Return:
            intersection: [L, 3]
    """
    drop_mask = torch.zeros((triangles.size(0))).bool()
    v0 = triangles[:, 0]  # [M, 3]
    v1 = triangles[:, 1]  # [M, 3]
    v2 = triangles[:, 2]  # [M, 3]

    v0v1 = v1 - v0  # [M, 3]
    v0v2 = v2 - v0  # [M, 3]

    pvec = directions.expand_as(v0v2).cross(
        v0v2, dim=-1)  # [M, 3] x [M, 3] -> [M, 3]

    # vector-wise dot product
    # det = v0v1.dot(pvec)
    elementwise_product = torch.mul(v0v1, pvec)
    det = torch.sum(elementwise_product, dim=-1)
    
    det = torch.where(det < EPS, EPS, det)
    drop_mask[det < EPS] = True

    invDet = 1.0 / det # [M] NaN assigned
    tvec = origins - v0 # [M, 3]
    
    # vector-wise dot product
    # u = tvec.dot(pvec) * invDet
    elementwise_product = torch.mul(tvec, pvec)
    u = torch.sum(elementwise_product, dim=-1) * invDet

    # invDet = torch.where((u < 0) | (u > 1), -torch.nan, invDet)
    drop_mask[(u < 0) | (u > 1)] = True

    qvec = tvec.cross(v0v1, dim=-1)  # [M, 3] x [M, 3] -> [M, 3]
    
    elementwise_product = torch.mul(directions.expand_as(qvec), qvec)
    v = torch.sum(elementwise_product, dim=-1) * invDet

    # invDet = torch.where((v < 0) | ((u + v)> 1), -torch.nan, invDet)
    drop_mask[(v < 0) | ((u + v) > 1)] = True

    
    elementwise_product = torch.mul(v0v2, qvec)
    t = torch.sum(elementwise_product, dim=-1) * invDet

    return t, drop_mask

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
