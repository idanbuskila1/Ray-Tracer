from surfaces.surface import Surface
import numpy as np
from intersection import Intersection

EPSILON = 0.000001


class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        super(InfinitePlane, self).__init__(material_index)
        self.normal = np.array(normal, dtype="float")
        self.normal /= np.linalg.norm(self.normal)
        self.offset = offset

    def get_intersection_with_ray(self, ray):
        NV = np.dot(self.normal, ray.v)
        t = np.dot(self.offset * self.normal - ray.origin, self.normal) / NV

        if t < 0 or np.abs(NV) < EPSILON:
            return None

        return Intersection(self, ray, t)

    def get_intersection_with_rays(self, rays):
        if len(rays) == 1:
            return [self.get_intersection_with_ray(rays[0])]

        with np.errstate(divide='ignore'):
            rays_v = np.array([ray.v for ray in rays])
            rays_origin = np.array([ray.origin for ray in rays])
            NV = np.dot(rays_v, self.normal)
            t = np.divide(np.dot(self.offset * self.normal - rays_origin, self.normal), NV)
            valid = np.logical_or(np.abs(NV) < EPSILON, t < 0)
            return [Intersection(self, rays[i], t[i]) if not valid[i] else None for i in
                    range(len(rays))]

    def get_normal(self, point):
        return self.normal