from pathlib import Path
import os
import pickle

import numpy as np
from tqdm import tqdm


def load_pcd_as_nparray(pcd_file):
    points = []
    with open(pcd_file, 'r') as f:
        lines = f.readlines()
    for line in lines[11:]:
        point = line.split()
        points.append(point)
    return np.asarray(points, dtype=np.float)


def prepare_pointcloud(root_path):
    sequence_list = os.listdir(root_path / 'data')
    sequence_list.sort()
    output_path = root_path / 'training' / 'pointcloud'
    output_path.mkdir(parents=True, exist_ok=True)
    print("Preparing pointcloud ...")

    frame_idx = 0
    for sequence in tqdm(sequence_list):
        lidar_data_path = root_path / 'data' / sequence / 'lidar'
        lidar_data_list = os.listdir(lidar_data_path / '1')
        lidar_data_list.sort()
        for lidar_frame in tqdm(lidar_data_list):
            horizon_front_data = load_pcd_as_nparray(lidar_data_path / '1' / lidar_frame)
            horizon_front_data[:, :3] = np.matmul(
                np.concatenate((horizon_front_data[:, :3], np.ones((horizon_front_data.shape[0], 1))), axis=1),
                lidar_extrinsic_param[0, :, :].T)[:, :3]
            horizon_front_left_data = load_pcd_as_nparray(lidar_data_path / '2' / lidar_frame)
            horizon_front_left_data[:, :3] = np.matmul(
                np.concatenate((horizon_front_left_data[:, :3], np.ones((horizon_front_left_data.shape[0], 1))),
                               axis=1),
                lidar_extrinsic_param[1, :, :].T)[:, :3]
            horizon_front_right_data = load_pcd_as_nparray(lidar_data_path / '5' / lidar_frame)
            horizon_front_right_data[:, :3] = np.matmul(
                np.concatenate((horizon_front_right_data[:, :3], np.ones((horizon_front_right_data.shape[0], 1))),
                               axis=1),
                lidar_extrinsic_param[4, :, :].T)[:, :3]
            tele_front_data = load_pcd_as_nparray(lidar_data_path / '6' / lidar_frame)
            tele_front_data[:, :3] = np.matmul(
                np.concatenate((tele_front_data[:, :3], np.ones((tele_front_data.shape[0], 1))), axis=1),
                lidar_extrinsic_param[5, :, :].T)[:, :3]
            lidar_data = np.concatenate(
                (horizon_front_data, horizon_front_left_data, horizon_front_right_data, tele_front_data), axis=0)

            with open(output_path / ('%06d.bin' % frame_idx), 'wb') as f:
                lidar_data.tofile(f)
            frame_idx += 1


def prepare_label(root_path):
    sequence_list = os.listdir(root_path / 'label')
    sequence_list.sort()
    output_path = root_path / 'training' / 'label'
    output_path.mkdir(parents=True, exist_ok=True)
    print("Preparing labels ...")

    frame_idx = 0
    for sequence in tqdm(sequence_list):
        label_files_path = root_path / 'label' / sequence / 'lidar' / 'all'
        label_files_list = os.listdir(label_files_path)
        label_files_list.sort()
        for label_file in label_files_list:
            with open(label_files_path / label_file, 'r') as f:
                lines = f.readlines()

            labels = []
            for line in lines:
                anno = line.split()
                label = {'name': anno[1], 'box3d_lidar': anno[4:11]}
                labels.append(label)

            with open(output_path / ('%06d.pkl' % frame_idx), 'wb') as f:
                pickle.dump(labels, f)
            frame_idx += 1


if __name__ == "__main__":
    data_path = Path("/home/tong.wang/datasets/Livox")
    lidar_extrinsic_param = np.array([
        [[0.999972, 0, 0.00750485, 0],
         [0, 1, 0, 0],
         [-0.00750485, 0, 0.999972, 1.909],
         [0, 0, 0, 1]],
        [[0.338535, -0.939803, 0.0465286, -0.001],
         [0.929107, 0.341685, 0.141463, 0.338],
         [-0.148845, -0.00465988, 0.98885, 1.909],
         [0, 0, 0, 1]],
        [[-0.742701, -0.656677, -0.131041, -1.109],
         [0.64927, -0.754084, 0.0990242, 0.15],
         [-0.163842, -0.0115354, 0.986419, 1.909],
         [0, 0, 0, 1]],
        [[-0.770801, 0.62401, -0.128361, -1.29],
         [-0.617308, -0.781369, -0.0916196, -0.149],
         [-0.157469, 0.00861766, 0.987486, 1.909],
         [0, 0, 0, 1]],
        [[0.33825, 0.939574, 0.0528062, -0.009],
         [-0.929838, 0.342329, -0.134952, -0.301],
         [-0.144874, -0.00345383, 0.989444, 1.909],
         [0, 0, 0, 1]],
        [[0.999931, -0.000861352, 0.0117086, -0.037],
         [0.00104713, 0.999874, -0.0158696, 0.019],
         [-0.0116934, 0.0158807, 0.999806, 1.909],
         [0, 0, 0, 1]],
    ])
    prepare_pointcloud(data_path)
    prepare_label(data_path)
