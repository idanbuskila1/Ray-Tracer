class Intersection:
    def __init__(self, *params):
        if len(params) == 3:
            self.init_without_hit_point(params[0], params[1], params[2])
        if len(params) == 4:
            self.init_with_hit_point(params[0], params[1], params[2], params[3])

    def init_without_hit_point(self, surface, ray, t):
        self.ray = ray
        self.surface = surface
        self.hit_point = ray.origin + t * ray.v
        self.t = t

    def init_with_hit_point(self, surface, ray, t, hit):
        self.ray = ray
        self.surface = surface
        self.hit_point = hit
        self.t = t

    def get_normal(self, point):
        return self.surface.get_normal(point)
