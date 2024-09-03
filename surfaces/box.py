from surfaces.surface import Surface
import numpy as np
from intersection import Intersection

EPSILON = 0.000001


class Box(Surface):
    def __init__(self, center, dimensions, rotation_matrix, material_index):
        super(Box, self).__init__(material_index)
        self.center = np.array(center, dtype="float")
        self.dimensions = np.array(dimensions, dtype="float")
        self.rotation_matrix = np.array(rotation_matrix, dtype="float")
        self.half_dimensions = self.dimensions / 2

    def get_intersection_with_ray(self, ray):
        # Transform the ray to the box's local coordinate system
        local_origin = np.dot(self.rotation_matrix.T, ray.origin - self.center)
        local_direction = np.dot(self.rotation_matrix.T, ray.v)

        t_min = -np.inf
        t_max = np.inf

        for i in range(3):
            if np.abs(local_direction[i]) < EPSILON:
                if (
                    local_origin[i] < -self.half_dimensions[i]
                    or local_origin[i] > self.half_dimensions[i]
                ):
                    return None
            else:
                t1 = (-self.half_dimensions[i] - local_origin[i]) / local_direction[i]
                t2 = (self.half_dimensions[i] - local_origin[i]) / local_direction[i]

                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))

                if t_min > t_max or t_max < 0:
                    return None

        t = t_min if t_min >= 0 else t_max
        hit_point = ray.origin + t * ray.v
        return Intersection(self, ray, t)

    def get_intersection_with_rays(self, rays):
        intersections = []
        for ray in rays:
            intersection = self.get_intersection_with_ray(ray)
            intersections.append(intersection)
        return intersections

    def get_normal(self, point):
        # Transform the point to the box's local coordinate system
        local_point = np.dot(self.rotation_matrix.T, point - self.center)
        normals = [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
        ]
        for i, normal in enumerate(normals):
            if np.isclose(local_point[i // 2], self.half_dimensions[i // 2]):
                return np.dot(self.rotation_matrix, normal)
        return None
