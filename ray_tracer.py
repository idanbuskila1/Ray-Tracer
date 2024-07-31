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
    return 0
    # return calc_light_intensity_bonus(light, intersection, rays)


def calc_light_intensity_regular(light, intersection, rays):
    light_hits = find_closest_rays_intersections_batch(rays)
    c = sum(
        1
        for light_hit in light_hits
        if light_hit is not None
        and np.linalg.norm(intersection.hit_point - light_hit.hit_point) < EPSILON
    )
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (
            c / (scene_settings.root_number_shadow_rays ** 2)
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


#=====================ALMOG====================================

def find_closest_rays_intersections_batch(
        rays):  #  After we will see it is working we will call it nearest_rays_intersections
    global surfaces
    closest_t_values = np.full(len(rays), float('inf'))
    nearest_intersections = np.full(len(rays), None)

    for surface in surfaces:
        if len(rays) == 1:
            intersections_with_ray = np.array(surface.get_intersection_with_ray(rays))
        else:
            intersections_with_ray = np.array(surface.get_intersection_with_rays(rays))
        t_values = np.array([intersection.t if intersection is not None else float('inf') for intersection in
                             intersections_with_ray])

        closer_t = t_values < closest_t_values
        closest_t_values = np.where(closer_t, t_values, closest_t_values)
        nearest_intersections = np.where(closer_t, intersections_with_ray, nearest_intersections)

    return nearest_intersections


def find_closest_ray_intersections(ray):  # to think whether delete later  - it is a redaundent function
    #the upcoming name is find_nearest_ray_intersection
    return find_closest_rays_intersections_batch([ray])[0]


def find_all_ray_intersections_sorted(ray): #ray_intersections_sorted_by_t
    global surfaces, EPSILON
    intersections = [
        intersection for surface in surfaces
        if (intersection := surface.get_intersection_with_ray(ray)) and intersection.t > EPSILON
    ]
    intersections.sort(key=lambda inter: inter.t)
    return intersections


def calculate_diffuse_color(intersection, light_intensity, light_color):
    global materials,  lights

    diffuse = np.array([0, 0, 0], dtype='float')
    N = intersection.surface.get_normal(intersection.hit_point)

    for light, intensity, color in zip(lights, light_intensity, light_color):
        L = light.position - intersection.hit_point
        L = normalize(L)
        NL = N @ L
        if NL <= 0:
            continue

        diffuse += color * intensity * NL

    diffuse_color = diffuse * intersection.surface.get_material(materials).diffuse_color
    return diffuse_color


def calculate_specular_color(intersection, light_intensity, light_color):
    global materials, lights

    specular = np.array([0, 0, 0], dtype='float')
    V = intersection.ray.origin - intersection.hit_point
    V = normalize(V)

    for light, intensity, color in zip(lights, light_intensity, light_color):
        L = light.position - intersection.hit_point
        L = normalize(L)

        light_ray = Ray(light.position, -L)
        R = intersection.surface.get_reflected_ray(light_ray, intersection.hit_point).v

        specular += color * intensity * light.specular_intensity * (np.power(np.dot(R, V), intersection.surface.get_material(materials).shininess))

    specular_color = specular * intersection.surface.get_material(materials).specular_color
    return specular_color


def get_ray_color(intersections, reflection_rec_level=0):
    global materials, scene_settings, lights

    if intersections is None or len(intersections) == 0:
        return scene_settings.background_color

    for i in range(len(intersections)):
        if intersections[i].surface.get_material(materials).transparency == 0:
            intersections = intersections[0:i + 1]
            break

    bg_color = scene_settings.background_color
    color = None
    for intersection in reversed(intersections):
        transparency = intersection.surface.get_material(materials).transparency

        if intersection is not None:
            light_intensity_color_pairs = [get_light_intensity_batch(light, intersection) for light in lights]
            light_intensity = [pair[0] for pair in light_intensity_color_pairs]
            light_color = [pair[1] for pair in light_intensity_color_pairs]  #I think  we can improve here -  TBD

            diffuse_color = calculate_diffuse_color(intersection, light_intensity, light_color)
            specular_color = calculate_specular_color(intersection, light_intensity, light_color)
            d_s_color = diffuse_color + specular_color

        else:
            d_s_color = scene_settings.background_color

        reflection_color = get_reflection_color(intersection, reflection_rec_level) * intersection.surface.get_material(
            materials).reflection_color

        color = ((1 - transparency) * d_s_color + transparency * bg_color) + reflection_color
        bg_color = color

    return color


#=====================ALMOG====================================


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


def save_image(image_array, img_name):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(f"{img_name}.png")


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Python Ray Tracer")
    parser.add_argument("scene_file", type=str, help="Path to the scene file")
    parser.add_argument("output_image", type=str, help="Name of the output image file")
    parser.add_argument("--width", type=int, default=500, help="Image width")
    parser.add_argument("--height", type=int, default=500, help="Image height")
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, materials, lights = parse_scene_file(args.scene_file)
    initialize_data(camera, scene_settings, surfaces, materials, lights, args.width, args.height, args.bonus)

    # TODO: Implement the ray tracer
    res = np.zeros((height, width, 3), dtype='float')
    # Dummy result
    image_array = construct_image(res)

    # Save the output image
    save_image(image_array, args.output_image)
    print(time.time() - start)


if __name__ == "__main__":
    main()
