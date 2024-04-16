import numpy as np

def test_print():
    print("tesing II")

def ray_triangle_intersection(ray_origin, ray_direction, vertex0, vertex1, vertex2):
    EPSILON = 1e-6
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON:
        return None  # 射线平行于三角形
    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    # 计算射线参数t
    t = f * np.dot(edge2, q)
    if t > EPSILON:
        # 返回交点
        intersection_point = ray_origin + t * ray_direction
        return intersection_point
    else:
        return None  # 射线在三角形背面
