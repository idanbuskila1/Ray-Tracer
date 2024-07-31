import numpy as np

normalize = lambda vec: vec / np.linalg.norm(vec)


class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = np.array(position, dtype=float)
        self.look_at = np.array(look_at, dtype=float)
        self.up_vector = np.array(up_vector, dtype=float)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
        self.direction = normalize(self.look_at - self.position)
        # self.right = normalize(np.cross(self.direction, self.up_vector))
        # self.up_vector = normalize(np.cross(self.right, self.direction))
        self.screen_center = self.position + self.screen_distance * self.direction
        self.ratio = -1

    def set_ratio(self, w):
        self.ratio = self.screen_width / w
