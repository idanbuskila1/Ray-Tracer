import numpy as np

normalize = lambda vec: vec / np.linalg.norm(vec)


class Ray:
    def __init__(self, *params):
        self.v = None
        self.origin = None
        if len(params) == 5:
            self.init_from_camera(params[0], params[1], params[2], params[3], params[4])
        if len(params) == 2:
            self.init_directly(params[0], params[1])

    def init_from_camera(self, camera, i, j, rx, ry):
        p = (
            camera.screen_center
            + (j - rx // 2) * camera.ratio * camera.right
            - (i - ry // 2) * camera.ratio * camera.up_vector
        )
        self.v = normalize(p - camera.position)
        self.origin = camera.position

    def init_directly(self, origin, v):
        self.origin = origin
        self.v = normalize(v)
