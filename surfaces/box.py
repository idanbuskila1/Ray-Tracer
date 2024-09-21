from intersection import Intersection
from surfaces.surface import Surface
from surfaces.infinite_plane import InfinitePlane
from surfaces.cube import Cube
import numpy as np
from ray import Ray
from numpy import linalg as LA

EPSILON = 1e-3


class Box(Surface):
    def __init__(self, position, scale, rotation_matrix, material_index):
        super(Box, self).__init__(material_index)
        self.position = np.array(position, dtype="float")
        self.scale = np.array(scale, dtype="float")
        self.min = self.position - self.scale / 2
        self.max = self.position + self.scale / 2
        self.rotation_matrix = np.array(rotation_matrix).reshape((3, 3))
        # Generate the 6 planes of the box in local space
        self.generate_planes()

    def generate_planes(self):
        """Generate the 6 planes that define the box's faces in local space."""
        # Local axes
        half_scale = self.scale / 2  # l w h

        # Box corners, based on its size and local orientation
        # self.planes = [
        #     # Right and left planes
        #     InfinitePlane(
        #         self.rotation_matrix[:, 0],
        #         np.dot(
        #             self.rotation_matrix[:, 0],
        #             self.position + half_scale[0] * self.rotation_matrix[:, 0],
        #         ),
        #         self.material_index,
        #     ),
        #     InfinitePlane(
        #         -self.rotation_matrix[:, 0],
        #         -np.dot(
        #             self.rotation_matrix[:, 0],
        #             self.position - half_scale[0] * self.rotation_matrix[:, 0],
        #         ),
        #         self.material_index,
        #     ),
        #     # Top and bottom planes
        #     InfinitePlane(
        #         self.rotation_matrix[:, 1],
        #         np.dot(
        #             self.rotation_matrix[:, 1],
        #             self.position + half_scale[1] * self.rotation_matrix[:, 1],
        #         ),
        #         self.material_index,
        #     ),
        #     InfinitePlane(
        #         -self.rotation_matrix[:, 1],
        #         -np.dot(
        #             self.rotation_matrix[:, 1],
        #             self.position - half_scale[1] * self.rotation_matrix[:, 1],
        #         ),
        #         self.material_index,
        #     ),
        #     # Front and back planes
        #     InfinitePlane(
        #         self.rotation_matrix[:, 2],
        #         np.dot(
        #             self.rotation_matrix[:, 2],
        #             self.position + half_scale[2] * self.rotation_matrix[:, 2],
        #         ),
        #         self.material_index,
        #     ),
        #     InfinitePlane(
        #         -self.rotation_matrix[:, 2],
        #         -np.dot(
        #             self.rotation_matrix[:, 2],
        #             self.position - half_scale[2] * self.rotation_matrix[:, 2],
        #         ),
        #         self.material_index,
        #     ),
        # ]

        # self.planes = [
        #     # Right and left planes
        #     InfinitePlane(
        #         self.rotation_matrix[0, :],
        #         self.position[0] + half_scale[0],
        #         self.material_index,
        #     ),
        #     InfinitePlane(
        #         -self.rotation_matrix[:, 0],
        #         -(self.position[0] - half_scale[0]),
        #         self.material_index,
        #     ),
        #     # Top and bottom planes
        #     InfinitePlane(
        #         self.rotation_matrix[1, :],
        #         self.position[1] + half_scale[1],
        #         self.material_index,
        #     ),
        #     InfinitePlane(
        #         -self.rotation_matrix[:, 1],
        #         -(self.position[1] - half_scale[1]),
        #         self.material_index,
        #     ),
        #     # Front and back planes
        #     InfinitePlane(
        #         self.rotation_matrix[2, :],
        #         self.position[2] + half_scale[2],
        #         self.material_index,
        #     ),
        #     InfinitePlane(
        #         -self.rotation_matrix[:, 2],
        #         -(self.position[2] - half_scale[2]),
        #         self.material_index,
        #     ),
        # ]

        self.planes = [
            InfinitePlane(
                np.array([1, 0, 0], dtype="float"),
                self.position[0] + self.scale[0] / 2,
                self.material_index,
            ),
            InfinitePlane(
                np.array([-1, 0, 0], dtype="float"),
                -(self.position[0] - self.scale[0] / 2),
                self.material_index,
            ),
            InfinitePlane(
                np.array([0, 1, 0], dtype="float"),
                self.position[1] + self.scale[1] / 2,
                self.material_index,
            ),
            InfinitePlane(
                np.array([0, -1, 0], dtype="float"),
                -(self.position[1] - self.scale[1] / 2),
                self.material_index,
            ),
            InfinitePlane(
                np.array([0, 0, 1], dtype="float"),
                self.position[2] + self.scale[2] / 2,
                self.material_index,
            ),
            InfinitePlane(
                np.array([0, 0, -1], dtype="float"),
                -(self.position[2] - self.scale[2] / 2),
                self.material_index,
            ),
        ]

    def get_intersection_with_ray(self, ray):
        """Calculate the intersection of a ray with the box."""
        # Transform the ray to the box's local space
        local_ray = Ray(
            self.rotation_matrix.T @ (ray.origin - self.position),
            self.rotation_matrix.T @ ray.v,
        )
        # Calculate the intersection with the box in local space
        t_min = (self.min - self.position - local_ray.origin) / local_ray.v + 1e-6
        t_max = (self.max - self.position - local_ray.origin) / local_ray.v + 1e-6
        t_enter = np.max(np.minimum(t_min, t_max))
        t_exit = np.min(np.maximum(t_min, t_max))
        if t_enter > t_exit:
            return None
        # Transform the hit point back to world space
        hit_point = local_ray.origin + t_enter * local_ray.v
        hit_point = (self.rotation_matrix @ hit_point) + self.position
        # assert not np.any(
        #     abs(LA.norm(hit_point - self.position, 2) - self.scale / 2) < EPSILON
        # )
        # print(hit_point)
        return Intersection(self, ray, t_enter, hit_point)

    def get_normal(self, point):
        """Calculate the normal at a point on the box surface."""
        local_point = np.array(self.rotation_matrix.T @ (point - self.position))
        minplane = self.planes[0]
        minn = float("inf")
        for plane in self.planes:
            cur = abs(np.dot(plane.normal, local_point) - plane.offset)
            if cur < minn:
                minn = cur
                minplane = plane
            # if np.isclose(np.dot(plane.normal, point), plane.offset, atol=EPSILON):
            #     return plane.normal
        # print(minn)
        world_normal = self.rotation_matrix @ minplane.normal
        return world_normal
        # for plane in self.planes:
        #     x = abs(np.dot(plane.normal, point) - plane.offset)
        #     print(x)
        #     if x < EPSILON:
        #         return plane.normal
        # print("fuck")

    def get_intersection_with_rays(self, rays):
        ret = []
        for ray in rays:
            ret.append(self.get_intersection_with_ray(ray))
        return ret
