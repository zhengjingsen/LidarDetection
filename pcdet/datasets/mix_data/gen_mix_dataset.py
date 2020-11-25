from pathlib import Path
import os
import pickle

import numpy as np
from tqdm import tqdm


def get_lidar(file_path, dataset_idx):
    with open(file_path, 'rb') as f:
        lidar_points = np.fromfile(f)

    if dataset_idx == 0:
        lidar_points = lidar_points.reshape(-1, 4)[:, :3]
        lidar_points[:, 2] -= 0.32  # adjust livox points height to the same as fake data
    else:
        lidar_points = lidar_points.reshape(-1, 3)

    return lidar_points


def get_label(file_path, dataset_idx):
    with open(file_path, 'rb') as f:
        labels = pickle.load(f)

    if dataset_idx == 0:
        for label in labels:
            # Combine Livox label names
            if label['name'] in ['car', 'police_car']:
                label['name'] = 'Car'
            elif label['name'] in ['bus', 'truck', 'Engineering_vehicles', 'trailer']:
                label['name'] = 'Truck'
            # Adjust the bounding boxes' height
            label['box3d_lidar'][2] = str(float(label['box3d_lidar'][2]) - 0.32)

    return labels


def generate_mix_dataset(livox_path, fake_lidar_path, output_path, num_frames):
    livox_frame_list = os.listdir(livox_path / 'training' / 'pointcloud')
    fake_lidar_frame_list = os.listdir(fake_lidar_path / 'training' / 'pointcloud')
    (output_path / 'training' / 'pointcloud').mkdir(parents=True, exist_ok=True)
    (output_path / 'training' / 'label').mkdir(parents=True, exist_ok=True)

    for frame_idx in tqdm(range(num_frames)):
        livox_percent = 5
        fake_lidar_percent = 3
        random_dataset = np.random.randint(0, livox_percent + fake_lidar_percent)
        if random_dataset < livox_percent:
            dataset_idx = 0
        else:
            dataset_idx = 1

        if dataset_idx == 0:
            rand_frame = np.random.randint(0, len(livox_frame_list))
            frame_path = livox_path / 'training' / 'pointcloud' / livox_frame_list[rand_frame]
            livxo_label_name = livox_frame_list[rand_frame].replace('bin', 'pkl')
            label_path = livox_path / 'training' / 'label' / livxo_label_name
            livox_frame_list.pop(rand_frame)
        else:
            rand_frame = np.random.randint(0, len(fake_lidar_frame_list))
            frame_path = fake_lidar_path / 'training' / 'pointcloud' / fake_lidar_frame_list[rand_frame]
            fake_lidar_label_name = fake_lidar_frame_list[rand_frame].replace('bin', 'pkl')
            label_path = fake_lidar_path / 'training' / 'label' / fake_lidar_label_name
            fake_lidar_frame_list.pop(rand_frame)

        output_frame_path = output_path / 'training' / 'pointcloud' / ('%06d.bin' % frame_idx)
        output_label_path = output_path / 'training' / 'label' / ('%06d.pkl' % frame_idx)
        lidar_points = get_lidar(frame_path, dataset_idx)
        labels = get_label(label_path, dataset_idx)
        with open(output_frame_path, 'wb') as f:
            lidar_points.tofile(f)
        with open(output_label_path, 'wb') as f:
            pickle.dump(labels, f)


if __name__ == "__main__":
    livox_path = Path("/home/yao.xu/datasets/Livox")
    fake_lidar_path = Path("/home/yao.xu/datasets/PlusAI")
    output_path = Path("/home/yao.xu/datasets/Mix")
    generate_mix_dataset(livox_path, fake_lidar_path, output_path, 7200)