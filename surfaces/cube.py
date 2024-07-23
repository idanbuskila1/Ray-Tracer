import numpy as np
from intersection import Intersection
from surfaces.infinite_plane import InfinitePlane
from ray import Ray


class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index
        self.planes = [
            InfinitePlane(
                np.array([0, -1, 0], dtype="float"),
                -(self.position[1] - self.scale / 2),
                self.material_index,
            ),
            InfinitePlane(
                np.array([0, 0, 1], dtype="float"),
                self.position[2] + self.scale / 2,
                self.material_index,
            ),
            InfinitePlane(
                np.array([1, 0, 0], dtype="float"),
                self.position[0] + self.scale / 2,
                self.material_index,
            ),
            InfinitePlane(
                np.array([-1, 0, 0], dtype="float"),
                -(self.position[0] - self.scale / 2),
                self.material_index,
            ),
            InfinitePlane(
                np.array([0, 0, -1], dtype="float"),
                -(self.position[2] - self.scale / 2),
                self.material_index,
            ),
            InfinitePlane(
                np.array([0, 1, 0], dtype="float"),
                self.position[1] + self.scale / 2,
                self.material_index,
            ),
        ]

    def get_intersection_with_ray(self, ray):
        with np.errstate(divide="ignore"):  # division by zero is ok.
            min = (self.position - (self.scale / 2) - ray.origin) / ray.v
            max = (self.position + (self.scale / 2) - ray.origin) / ray.v
            enter = np.max(np.minimum(min, max))
            exit = np.min(np.maximum(min, max))
            if enter > exit:
                return None
            return Intersection(self, ray, enter)

    def get_intersection_with_rays(self, rays):
        rays_v = np.array([ray.v for ray in rays])
        rays_origin = np.array([ray.origin for ray in rays])
        with np.errstate(divide="ignore"):  # division by zero is ok.
            t_min = (self.position - (self.scale / 2) - rays_origin) / rays_v
            t_max = (self.position + (self.scale / 2) - rays_origin) / rays_v
            t_enter = np.max(np.minimum(t_min, t_max), axis=1)
            t_exit = np.min(np.maximum(t_min, t_max), axis=1)
            return [
                (
                    Intersection(self, rays[i], t_enter[i])
                    if t_exit[i] >= t_enter[i]
                    else None
                )
                for i in range(len(rays))
            ]

    def get_material(self, materials):
        return materials[self.material_index - 1]

    def in_cube(self, point):
        for i in range(3):
            if not (
                self.position[i] - self.scale / 2
                <= point[i]
                <= self.position[i] + self.scale / 2
            ):
                return False
        return True

    def get_normal(self, point):
        for plane in self.planes:
            if abs(np.dot(plane.normal, point) - plane.offset) < 0.00001:
                return plane.normal

    def get_reflected_ray(self, ray, point):
        normal = self.get_normal(point)
        reflection_dir = ray.v - 2 * (ray.v @ normal) * normal
        return Ray(point, reflection_dir)
