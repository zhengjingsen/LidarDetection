import os
import time
import pickle

import numpy as np
from pathlib import Path
from tqdm import tqdm

from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils


def get_lidar(file_path):
    lidar_points = np.fromfile(file_path, dtype=np.float)
    lidar_points = np.reshape(lidar_points, (-1, 3))

    return lidar_points


def sample_groundtruth_objects(root_path, scene, ground_plane_params, num_sample_objects, object_range):
    db_info_save_path = Path(root_path) / 'plusai_gt_dbinfos.pkl'
    with open(db_info_save_path, 'rb') as f:
        gt_dbinfos = pickle.load(f)

    # Place the objects on the scene
    sampled_objects = []
    sampled_boxes = []
    object_pts_list = []
    for _ in range(num_sample_objects):
        # Random sample objects from different categories
        car_percent = category_dist[0]
        truck_percent = category_dist[1]
        tram_percent = category_dist[2]
        category_idx = np.random.randint(0, car_percent + truck_percent + tram_percent)
        if category_idx <= car_percent:
            category = 'Car'
        elif category_idx <= car_percent + truck_percent:
            category = 'Truck'
        else:
            category = 'Tram'

        # Random pick objects
        total_num_gt_objects = len(gt_dbinfos[category])
        sample_idx = np.random.randint(0, total_num_gt_objects)
        obj = gt_dbinfos[category][sample_idx]
        sampled_box = obj['box3d_lidar']

        # Filter objects with few points
        if obj['num_points_in_gt'] < 10:
            continue

        # Filter objects out of RoI
        if obj['box3d_lidar'][0] <= object_range[0] or obj['box3d_lidar'][0] >= object_range[3] \
                or obj['box3d_lidar'][1] <= object_range[1] or obj['box3d_lidar'][1] >= object_range[4] \
                or obj['box3d_lidar'][2] <= object_range[2] or obj['box3d_lidar'][2] >= object_range[5]:
            continue

        # Filter objects by size
        if obj['box3d_lidar'][3] <= 3.5 \
                or obj['box3d_lidar'][4] <= 1.5 or obj['box3d_lidar'][4] >= 5. \
                or obj['box3d_lidar'][5] <= 1. or obj['box3d_lidar'][5] >= 6.:
            continue

        # Avoid conflict
        if len(sampled_boxes):
            sampled_box_np = np.array([sampled_box], dtype=np.float)
            # Enlarge the box to ignore close box
            sampled_box_enlarged_np = sampled_box_np
            sampled_box_enlarged_np[0][3] += 1.0   # Length
            sampled_box_enlarged_np[0][4] += 0.5   # Width
            sampled_boxes_np = np.array(sampled_boxes, dtype=np.float)
            iou_3d = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes_np, sampled_box_enlarged_np)
            if np.sum(iou_3d) > 0:
                continue

        # Move the sampled box onto the ground plane according to the plane params
        ground_height = ground_plane_params[0] * sampled_box[0] + \
                        ground_plane_params[1] * sampled_box[1] + ground_plane_params[2]
        ground_height = min(max(ground_height, -0.6), 0.6)  # Set the bounds of the ground height
        sampled_box[2] = ground_height + sampled_box[5] / 2
        obj['box3d_lidar'][2] = ground_height + obj['box3d_lidar'][5] / 2
        sampled_boxes.append(sampled_box)
        sampled_objects.append(obj)

        # Get object points and add to object points list
        lidar_path = Path(root_path) / obj['path']
        obj_pts = get_lidar(lidar_path)
        obj_pts[:, :3] += sampled_box[:3]
        object_pts_list.append(obj_pts)

    # Remove points inside the enlarged sampled boxes
    obj_pts = np.concatenate(object_pts_list, axis=0)
    sampled_boxes_np = np.array(sampled_boxes, dtype=np.float)
    sampled_boxes_enlarged_np = sampled_boxes_np
    sampled_boxes_enlarged_np[:, 3] += 0.05
    sampled_boxes_enlarged_np[:, 4] += 0.05
    sampled_boxes_enlarged_np[:, 5] *= 2
    scene = box_utils.remove_points_in_boxes3d(scene, sampled_boxes_enlarged_np)
    # Add the object points to the scene
    scene = np.concatenate([scene, obj_pts], axis=0)

    return scene, sampled_objects


def generate_fake_lidar(num_frames,
                        num_objects_per_frame,
                        object_range=None,
                        gt_data_path=None,
                        blank_scene_path=None,
                        output_path=None):
    print("Generating fake lidar data ...")
    # Load all blank scenes
    blank_scene_path = Path(blank_scene_path)
    blank_scene_list = os.listdir(blank_scene_path)
    total_num_blank_scene = len(blank_scene_list)

    # Load ground plane estimation param dict
    ground_plane_params_dict_file = blank_scene_path / 'ground_plane_params.pkl'
    with open(ground_plane_params_dict_file, 'rb') as f:
        ground_plane_params_dict = pickle.load(f)

    for idx in tqdm(range(num_frames)):
        # Randomly load blank scene
        scene_idx = np.random.randint(0, total_num_blank_scene)
        # scene = np.loadtxt(blank_scene_path / blank_scene_list[scene_idx])
        if '.txt' not in blank_scene_list[scene_idx]:
            continue  # Ignore non txt files
        with open(blank_scene_path / blank_scene_list[scene_idx], 'r') as a:
            b = a.readlines()
            a.close()
            for i in range(len(b)):
                b[i] = b[i].strip("\n")
                b[i] = b[i].split()
                b[i][:] = map(float, b[i][:])
            scene = np.asarray(b)

        # Load ground plane estimation params from dict
        ground_plane_params = ground_plane_params_dict[blank_scene_list[scene_idx]]

        # Randomly sample objects from database
        sampled_scene, sampled_objects = sample_groundtruth_objects(gt_data_path,
                                                                    scene,
                                                                    ground_plane_params,
                                                                    num_objects_per_frame,
                                                                    object_range)

        # Save the fake lidar points
        lidar_path = Path(output_path) / "pointcloud"
        lidar_path.mkdir(parents=True, exist_ok=True)
        lidar_file = lidar_path / ("%06d.bin" % idx)
        with open(lidar_file, 'wb') as f:
            sampled_scene.tofile(f)

        # Save sampled object annos
        frame_annotations = []
        for obj in sampled_objects:
            anno = {'name': obj['name'],
                    'image_idx': "%06d" % idx,
                    'box3d_lidar': obj['box3d_lidar'],
                    'num_points_in_gt': obj['num_points_in_gt']}
            frame_annotations.append(anno)

        anno_path = Path(output_path) / "label"
        anno_path.mkdir(parents=True, exist_ok=True)
        anno_file = anno_path / ("%06d.pkl" % idx)
        with open(anno_file, 'wb') as f:
            pickle.dump(frame_annotations, f)


if __name__ == "__main__":
    bev_range = [1, -6, -2, 148, 6, 4]
    category_dist = [5, 4, 4]     # Define the sample ratio for different categories
    num_fake_frames = 5200
    max_num_objects_per_frame = 20
    blank_scene_root = "/home/ethan/Workspace/dataset/PlusAI/blank_scene"

    generate_fake_lidar(num_fake_frames,
                        max_num_objects_per_frame,
                        object_range=bev_range,
                        gt_data_path="/home/ethan/Workspace/dataset/PlusAI",
                        blank_scene_path=blank_scene_root,
                        output_path="/home/ethan/Workspace/dataset/PlusAI/training")
