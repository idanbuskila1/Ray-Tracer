import argparse
from PIL import Image
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray
import numpy as np
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


def parse_scene_file(file_path):
    surfaces = []
    lights = []
    materials = []
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
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, surfaces, materials, lights


def find_all_ray_intersections_sorted(ray):
    global surfaces, EPSILON
    intersections = [
        intersection
        for surface in surfaces
        if (intersection := surface.get_intersection_with_ray(ray))
        and intersection.t > EPSILON
    ]
    intersections.sort(key=lambda inter: inter.t)
    return intersections


def find_closest_rays_intersections_batch(rays):
    global surfaces
    closest_t_values = np.full(len(rays), float("inf"))
    nearest_intersections = np.full(len(rays), None)

    for surface in surfaces:
        if len(rays) == 1:
            intersections_with_ray = np.array(surface.get_intersection_with_ray(rays))
        else:
            intersections_with_ray = np.array(surface.get_intersection_with_rays(rays))
        t_values = np.array(
            [
                intersection.t if intersection is not None else float("inf")
                for intersection in intersections_with_ray
            ]
        )

        closer_t = t_values < closest_t_values
        closest_t_values = np.where(closer_t, t_values, closest_t_values)
        nearest_intersections = np.where(
            closer_t, intersections_with_ray, nearest_intersections
        )

    return nearest_intersections


def find_closest_ray_intersections(ray):
    # global surfaces
    # min_t = float('inf')
    # closest_intersection = None
    #
    # for surface in surfaces:
    #     intersection = surface.get_intersection_with_ray(ray)
    #     if intersection is None:
    #         continue
    #     if min_t > intersection.t:
    #         min_t = intersection.t
    #         closest_intersection = intersection
    # return closest_intersection
    return find_closest_rays_intersections_batch([ray])[0]


def get_base_points(light):
    global scene_settings
    ratio = light.radius / scene_settings.root_number_shadow_rays

    x_values_vec, y_values_vec = np.meshgrid(
        range(scene_settings.root_number_shadow_rays),
        range(scene_settings.root_number_shadow_rays),
    )
    x_values_vec = x_values_vec.reshape(-1)
    y_values_vec = y_values_vec.reshape(-1)
    base_xy = np.array(
        [
            x_values_vec * ratio - light.radius / 2,
            y_values_vec * ratio - light.radius / 2,
            np.zeros_like(x_values_vec),
            np.zeros_like(x_values_vec),
        ]
    )
    offset = np.array(
        [
            np.random.uniform(0, ratio, y_values_vec.shape),
            np.random.uniform(0, ratio, x_values_vec.shape),
            np.zeros_like(x_values_vec),
            np.ones_like(x_values_vec),
        ]
    )
    return base_xy + offset


def get_light_intensity_batch(light, intersection):
    global bonus
    normal = normalize(intersection.hit_point - light.position)
    transform_mat = transform_plane(normal, light.position)
    base_points = get_base_points(light)

    lights = (transform_mat @ base_points)[:3].T

    rays = [Ray(hit, intersection.hit_point - hit) for hit in lights]
    if not bonus:
        return get_intensity(light, intersection, rays), light.color
    return get_intensity_bonus(light, intersection, rays)


def get_intensity(light, intersection, rays):
    hits = find_closest_rays_intersections_batch(rays)
    count = 0
    for hit in hits:
        if (
            hit is not None
            and np.linalg.norm(intersection.hit_point - hit.hit_point) < EPSILON
        ):
            count += 1
    ret = (1 - light.shadow_intensity) + light.shadow_intensity * (
        count / (scene_settings.root_number_shadow_rays**2)
    )
    return ret


def get_intensity_bonus(light, intersection, rays):
    light_hits = [find_all_ray_intersections_sorted(ray) for ray in rays]
    light_color = np.array([0, 0, 0], dtype="float")
    c = 0
    for light_intersections in light_hits:
        ray_contribution = 1
        curr_ray_color = np.array(light.color)

        for light_intersection in light_intersections:

            if (
                np.linalg.norm(intersection.hit_point - light_intersection.hit_point)
                < EPSILON
            ):
                c += ray_contribution
                light_color += curr_ray_color
                break
            curr_ray_color *= light_intersection.surface.get_material(
                materials
            ).diffuse_color
            ray_contribution *= light_intersection.surface.get_material(
                materials
            ).transparency
            if ray_contribution == 0:
                break
    light_intensity = (1 - light.shadow_intensity) + light.shadow_intensity * (
        c / (scene_settings.root_number_shadow_rays**2)
    )
    if bonus_type == 2:
        return light_intensity, light.color * 0.9 + 0.1 * (
            light_color / (scene_settings.root_number_shadow_rays**2)
        )
    return light_intensity, light.color


def calc_reflection(intersection, reflection_rec_level):
    ray = intersection.surface.get_reflected_ray(
        intersection.ray, intersection.hit_point
    )
    return get_ray_color(
        find_all_ray_intersections_sorted(ray), reflection_rec_level + 1
    )


def get_reflection(intersection, reflection_rec_level):
    global scene_settings
    if reflection_rec_level >= scene_settings.max_recursions:
        return scene_settings.background_color
    return calc_reflection(intersection, reflection_rec_level)


def get_ray_color(intersections, reflection_rec_level=0):
    global materials, scene_settings, lights

    if intersections is None or len(intersections) == 0:
        return scene_settings.background_color

    for i in range(len(intersections)):
        if intersections[i].surface.get_material(materials).transparency == 0:
            intersections = intersections[0 : i + 1]
            break

    bg_color = scene_settings.background_color
    color = None
    for intersection in reversed(intersections):
        transparency = intersection.surface.get_material(materials).transparency

        if intersection is not None:
            light_intensity_color_pairs = [
                get_light_intensity_batch(light, intersection) for light in lights
            ]
            light_intensity = [pair[0] for pair in light_intensity_color_pairs]
            light_color = [
                pair[1] for pair in light_intensity_color_pairs
            ]  # I think  we can improve here -  TBD

            diffuse_color = calculate_diffuse_color(
                intersection, light_intensity, light_color
            )
            specular_color = calculate_specular_color(
                intersection, light_intensity, light_color
            )
            d_s_color = diffuse_color + specular_color

        else:
            d_s_color = scene_settings.background_color

        reflection_color = (
            get_reflection(intersection, reflection_rec_level)
            * intersection.surface.get_material(materials).reflection_color
        )

        color = (
            (1 - transparency) * d_s_color + transparency * bg_color
        ) + reflection_color
        bg_color = color

    return color


def calculate_diffuse_color(intersection, light_intensity, light_color):
    global materials, lights

    diffuse = np.array([0, 0, 0], dtype="float")
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

    specular = np.array([0, 0, 0], dtype="float")
    V = intersection.ray.origin - intersection.hit_point
    V = normalize(V)

    for light, intensity, color in zip(lights, light_intensity, light_color):
        L = light.position - intersection.hit_point
        L = normalize(L)

        light_ray = Ray(light.position, -L)
        R = intersection.surface.get_reflected_ray(light_ray, intersection.hit_point).v

        specular += (
            color
            * intensity
            * light.specular_intensity
            * (
                np.power(
                    np.dot(R, V), intersection.surface.get_material(materials).shininess
                )
            )
        )

    specular_color = (
        specular * intersection.surface.get_material(materials).specular_color
    )
    return specular_color


def get_diffuse_and_specular_color(intersection):
    global scene_settings
    global materials
    global lights
    if intersection is None:
        return scene_settings.background_color
    dif_sum = np.array([0, 0, 0], dtype="float")
    spec_sum = np.array([0, 0, 0], dtype="float")
    for light in lights:
        N = intersection.surface.get_normal(intersection.hit_point)
        L = light.position - intersection.hit_point
        L /= np.linalg.norm(L)
        N_L_dot = N @ L
        if N_L_dot <= 0:
            continue
        light_intensity, light_color = get_light_intensity_batch(light, intersection)
        V = intersection.ray.origin - intersection.hit_point
        V /= np.linalg.norm(V)
        light_ray = Ray(light.position, -L)
        R = intersection.surface.get_reflected_ray(light_ray, intersection.hit_point).v
        dif_sum += light_color * light_intensity * N_L_dot
        spec_sum += (
            light_color
            * light_intensity
            * light.specular_intensity
            * (
                np.power(
                    np.dot(R, V), intersection.surface.get_material(materials).shininess
                )
            )
        )
    diffuse_color = dif_sum * intersection.surface.get_material(materials).diffuse_color
    specular_color = (
        spec_sum * intersection.surface.get_material(materials).specular_color
    )

    color = diffuse_color + specular_color
    return color


def calc_img(img):
    global width
    global height
    global camera
    camera.set_ratio(width)
    for i in range(height):
        print(f"row:{i}")
        for j in range(width):
            ray = Ray(camera, i, j, width, height)
            img[i, j] = get_ray_color(find_all_ray_intersections_sorted(ray))
    img[img > 1] = 1
    img[img < 0] = 0
    return img * 255


def save_image(image_array, img_name):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save(f"{img_name}.png")


def calc_rotation(plane_normal, z):
    axis = np.cross(z, plane_normal)
    cos = np.dot(z, plane_normal)
    sin = np.linalg.norm(axis) / np.linalg.norm(plane_normal)
    axis /= np.linalg.norm(axis)

    ret = np.eye(4)
    ret[:3, :3] = (cos * np.eye(3)) + (
        sin
        * np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ],
            dtype="float",
        )
        + (1 - cos) * np.outer(axis, axis)
    )
    return ret


def transform_plane(plane_normal, plane_point):
    z = np.array([0, 0, 1], dtype="float")
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = plane_point
    if np.linalg.norm(z - abs(plane_normal)) < EPSILON:
        return translation_matrix

    rotation_matrix = calc_rotation(plane_normal, z)
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
    width = w
    height = h
    camera = cam
    materials = mats
    surfaces = surfs
    lights = light
    scene_settings = settings
    bonus = False if bonus_t == -1 else True
    bonus_type = bonus_t


def main():
    start = time.time()
    parser = argparse.ArgumentParser(description="Python Ray Tracer")
    parser.add_argument(
        "scene_file",
        type=str,
        default="./scenes/Room.txt",
        help="Path to the scene file",
    )
    parser.add_argument(
        "output_image",
        type=str,
        default="./output/try1.png",
        help="Name of the output image file",
    )
    parser.add_argument("--width", type=int, default=500, help="Image width")
    parser.add_argument("--height", type=int, default=500, help="Image height")
    parser.add_argument(
        "--bonus",
        type=int,
        default=-1,
        help="Bonus (-1 for  no bonus, 1 for bonus without coloring and 2 for bonus with coloring) ",
    )

    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, surfaces, materials, lights = parse_scene_file(
        args.scene_file
    )
    initialize_data(
        camera,
        scene_settings,
        surfaces,
        materials,
        lights,
        args.width,
        args.height,
        args.bonus,
    )
    res = np.zeros((height, width, 3), dtype="float")
    image_array = calc_img(res)
    # Save the output image
    save_image(image_array, args.output_image)
    print(time.time() - start)


if __name__ == "__main__":
    main()
