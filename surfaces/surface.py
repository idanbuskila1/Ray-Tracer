from abc import ABC, abstractmethod

import numpy as np

from ray import Ray


class Surface(ABC):
    def __init__(self, material_index):
        self.material_index = material_index

    def get_material(self, materials):
        return materials[self.material_index - 1]

    @abstractmethod
    def get_intersection_with_ray(self, ray):
        pass

    @abstractmethod
    def get_intersection_with_rays(self, rays):
        pass

    @abstractmethod
    def get_normal(self, point):
        pass

    def get_reflected_ray(self, ray, point):
        surface_normal = self.get_normal(point)
        cos_theta = np.dot(ray.v, surface_normal)
        reflected_direction = ray.v - 2 * cos_theta * surface_normal
        return Ray(point, reflected_direction)
