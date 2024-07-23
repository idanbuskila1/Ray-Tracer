import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray
import time

EPSILON = 0.000001
camera: Camera
scene_settings: SceneSettings
surfaces = []
materials = []
lights = []
width = -1
height = -1
bonus = False
bonus_type = -1

normalize = lambda vec: vec / np.linalg.norm(vec)


def get_light_intensity_batch(light, intersection):
    global scene_settings
    global bonus
    ratio = light.radius / scene_settings.root_number_shadow_rays
    plane_normal = normalize(intersection.hit_point - light.position)
    transform_matrix = xy_to_general_plane(plane_normal, light.position)

    x_values_vec, y_values_vec = np.meshgrid(
        range(scene_settings.root_number_shadow_rays),
        range(scene_settings.root_number_shadow_rays),
    )  # build grid starting points
    x_values_vec = x_values_vec.reshape(-1)
    y_values_vec = y_values_vec.reshape(-1)
    base_xy = np.array(
        [
            x_values_vec * ratio - light.radius / 2,  # center grid around 0,0
            y_values_vec * ratio - light.radius / 2,
            np.zeros_like(x_values_vec),
            np.zeros_like(x_values_vec),
        ]
    )
    offset = np.array(  # select random point inside each grid cell
        [
            np.random.uniform(0, ratio, y_values_vec.shape),
            np.random.uniform(0, ratio, x_values_vec.shape),
            np.zeros_like(x_values_vec),
            np.ones_like(x_values_vec),
        ]
    )  # for translation matrix
    rectangle_points = base_xy + offset

    light_points = (transform_matrix @ rectangle_points)[:3].T

    rays = [Ray(point, intersection.hit_point - point) for point in light_points]
    if not bonus:
        return calc_light_intensity_regular(light, intersection, rays), light.color
    return calc_light_intensity_bonus(light, intersection, rays)


def calc_light_intensity_regular(light, intersection, rays):
    light_hits = find_closest_rays_intersections_batch(rays)
    c = sum(
        1
        for light_hit in light_hits
        if light_hit is not None
        and np.linalg.norm(intersection.hit_point - light_hit.hit_point) < EPSILON
    )
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (
        c / (scene_settings.root_number_shadow_rays**2)
    )
    return light_intensity


def get_reflection_color(intersection, recursion):
    global scene_settings
    if recursion >= scene_settings.max_recursions:
        return scene_settings.background_color
    reflection_ray = intersection.surface.get_reflected_ray(
        intersection.ray, intersection.hit_point
    )
    return get_ray_color(
        find_all_ray_intersections_sorted(reflection_ray), recursion + 1
    )


def construct_image(res):
    global width
    global height
    global camera
    camera.set_ratio(width)
    for i in range(height):
        for j in range(width):
            ray = Ray(camera, i, j, width, height)
            res[i, j] = get_ray_color(find_all_ray_intersections_sorted(ray))
    res[res > 1] = 1
    res[res < 0] = 0
    return res * 255


def xy_to_general_plane(plane_normal, plane_point):
    z = np.array([0, 0, 1], dtype="float")
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = plane_point
    if np.linalg.norm(z - abs(plane_normal)) < EPSILON:  # no need to rotate
        return translation_matrix

    rotation_axis = np.cross(z, plane_normal)
    cos_theta = np.dot(z, plane_normal)
    sin_theta = np.linalg.norm(rotation_axis) / np.linalg.norm(plane_normal)
    rotation_axis /= np.linalg.norm(rotation_axis)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = (cos_theta * np.eye(3)) + (
        sin_theta
        * np.array(
            [
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0],
            ],
            dtype="float",
        )
        + (1 - cos_theta) * np.outer(rotation_axis, rotation_axis)
    )
    return np.matmul(translation_matrix, rotation_matrix)


def initialize_data(cam, settings, surfs, mats, light, w, h, bonus_t):
    global camera
    global scene_settings
    global surfaces
    global materials
    global lights
    global width
    global height
    global bonus
    global bonus_type
    bonus = False if bonus_t == -1 else True
    bonus_type = bonus_t
    camera = cam
    scene_settings = settings
    surfaces = surfs
    materials = mats
    lights = light
    width = w
    height = h


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(
                    params[:3], params[3:6], params[6:9], params[9], params[10]
                )
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(
                    params[:3], params[3:6], params[6:9], params[9], params[10]
                )
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description="Python Ray Tracer")
    parser.add_argument("scene_file", type=str, help="Path to the scene file")
    parser.add_argument("output_image", type=str, help="Name of the output image file")
    parser.add_argument("--width", type=int, default=500, help="Image width")
    parser.add_argument("--height", type=int, default=500, help="Image height")
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


if __name__ == "__main__":
    main()
