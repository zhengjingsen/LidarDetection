import math
import os
from pathlib import Path
import pickle

import numpy as np
from sklearn import linear_model
from tqdm import tqdm


def find_plane(points):
    XY = points[:, :2]
    Z = points[:, 2]

    ransac = linear_model.RANSACRegressor(residual_threshold=0.02)
    ransac.fit(XY, Z)
    a, b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_

    return a, b, d  # Z = aX + bY + d


def angle_rotate(a, b, d):
    x = np.arange(30)
    y = np.arange(30)
    X, Y = np.meshgrid(x, y)
    Z = a * X + b * Y + d
    rad = math.atan2(Y[1][0] - Y[0][0], (Z[1][0] - Z[0][0]))
    return np.pi / 2 - rad


def get_angle_pitch(a, b, d):
    return -math.atan2(a, 1)


def estimate_ground_plane_batch(file_path):
    blank_scene_path = Path(file_path)
    blank_scene_list = os.listdir(blank_scene_path)

    ground_plane_params_dict = {}
    for blank_scene in tqdm(blank_scene_list):
        if 'txt' not in blank_scene:
            continue
        with open(blank_scene_path / blank_scene, 'r') as a:
            b = a.readlines()
            a.close()
            for i in range(len(b)):
                b[i] = b[i].strip("\n")
                b[i] = b[i].split()
                b[i][:] = map(float, b[i][:])
            scene = np.asarray(b)

            a, b, d = find_plane(scene)
            ground_plane_params_dict.update({blank_scene: (a, b, d)})

    output_filename = blank_scene_path / 'ground_plane_params.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(ground_plane_params_dict, f)
        print("Ground plane params estimation finished.")


def pts_rotation_pitch(points, pitch_angle):
    rot_matrix = np.array([[math.cos(pitch_angle), 0, math.sin(pitch_angle)],
                           [0, 1, 0],
                           [-math.sin(pitch_angle), 0, math.cos(pitch_angle)]])
    points_rotated = np.matmul(points, rot_matrix)
    with open("/home/ethan/Downloads/points_rotated.txt", 'w') as f:
        np.savetxt(f, points_rotated)

    return points_rotated


def gen_rectified_plane(file_path):
    blank_scene_path = Path(file_path)
    blank_scene_list = os.listdir(blank_scene_path)
    rotated_path = blank_scene_path / 'rotated'
    rotated_path.mkdir(parents=True, exist_ok=True)

    ground_plane_params_dict = {}
    for blank_scene in tqdm(blank_scene_list):
        if 'txt' not in blank_scene:
            continue
        with open(blank_scene_path / blank_scene, 'r') as a:
            b = a.readlines()
            a.close()
            for i in range(len(b)):
                b[i] = b[i].strip("\n")
                b[i] = b[i].split()
                b[i][:] = map(float, b[i][:])
            scene = np.asarray(b)

            # Find the ground plane using RANSAC
            a, b, d = find_plane(scene)
            # Rotate the ground plane
            pitch_angle = get_angle_pitch(a, b, d)
            scene_rotated = pts_rotation_pitch(scene, pitch_angle)
            with open(rotated_path / blank_scene, 'w') as f:
                np.savetxt(f, scene_rotated)
            # Update the ground plane params
            normal_vec = np.array([a, b, -1])
            normal_vec_rotated = pts_rotation_pitch(normal_vec, pitch_angle)
            ground_plane_params_dict.update({blank_scene: (normal_vec_rotated[0] / - normal_vec_rotated[2],
                                                           normal_vec_rotated[1] / - normal_vec_rotated[2],
                                                           d)})

    output_filename = rotated_path / 'ground_plane_rotated_params.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump(ground_plane_params_dict, f)
        print("Ground plane rotated params estimation finished.")


if __name__ == '__main__':
    estimate_ground_plane_batch("/home/ethan/Workspace/dataset/PlusAI/blank_scene")
    # gen_rectified_plane("/home/ethan/Workspace/dataset/PlusAI/blank_scene")
