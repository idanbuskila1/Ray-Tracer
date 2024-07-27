import numpy as np

from intersection import Intersection


class Sphere:
    def __init__(self, material_index, position, radius):
        super().__init__(material_index)
        self.position = position
        self.radius = radius

    @staticmethod
    def closest_t(t1, t2):
        t1[t1 < 0] = np.inf
        t2[t2 < 0] = np.inf
        t = np.minimum(t1, t2)
        t[np.isinf(t)] = np.nan  # Handle cases of negativity
        return t

    def get_intersection_with_ray(self, ray):
        # result = self.get_intersection_with_rays([ray])
        # return result[0]
        # if it doesnt work than put it:
        P0 = ray.origin - self.position
        b = 2 * (ray.v @ P0)
        c = (P0 @ P0) - self.radius ** 2
        discriminant = b ** 2 - 4 * c
        if discriminant <= 0:
            return None
        discriminant_sqrt = np.sqrt(discriminant)
        t1 = (-b + discriminant_sqrt) / 2
        t2 = (-b - discriminant_sqrt) / 2
        t = Sphere.closest_t(t1, t2)

        if t is None or np.isnan(t).all():
            return None
        return Intersection(self, ray, t)

    def get_intersection_with_rays(self, rays):

        ray_origins = np.array([ray.origin for ray in rays])
        ray_directions = np.array([ray.v for ray in rays])
        P0 = ray_origins - self.position

        # Coefficients for the quadratic equation
        b = 2 * np.einsum('ij,ij->i', P0, ray_directions)
        c = np.einsum('ij,ij->i', P0, P0) - self.radius ** 2

        # Solve the quadratic equation
        discriminant = b ** 2 - 4 * c
        illegals = discriminant < 0
        discriminant[illegals] = 0
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / 2
        t2 = (-b - sqrt_discriminant) / 2
        t = Sphere.closest_t(t1, t2) # Select the smallest non-negative t value

        # Determine which t values are valid (not NaN)
        valid = ~np.isnan(t)

        # Create Intersection objects for valid intersections, None otherwise
        intersections = [Intersection(self, rays[i], t[i]) if valid[i] and not illegals[i] else None for i in range(len(rays))]

        return intersections

    def get_normal(self, point):
        N = point - self.position
        return N / np.linalg.norm(N)
