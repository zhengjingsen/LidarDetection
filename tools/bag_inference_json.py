import argparse
import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.utils.data_viz import plot_gt_boxes
from pcdet.datasets.processor.data_processor import DataProcessor


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--bag_file', type=str, default=None, help='specify the bag file to be inferenced')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for inference')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='model checkpoint')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    np.random.seed(1024)

    return args, cfg


def main():
    args, cfg = parse_config()

    log_file = 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = common_utils.create_logger(log_file, rank=0)

    # Build dataset config
    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=args.workers, training=False
    )

    # Build network
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # Load the data
    # bag = rosbag.Bag(args.bag_file, 'r')
    test_scene_path = Path("/home/tong.wang/datasets/test_scene")
    test_scene_list = os.listdir(test_scene_path)
    total_num_test_scene = len(test_scene_list)

    # json_dict = {'objects': [{'bounds': [], 'size': {'x': 0, 'y': 0, 'z': 0}, 'uuid': 0}]}
    result_pkl = []

    with torch.no_grad():
        # load checkpoint
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()

        # start evaluation
        class_names = cfg.CLASS_NAMES
        point_cloud_range = np.array([0, -8, -2, 152, 8, 6])
        processor = DataProcessor(cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range, training=False)

        model.eval()

        for idx in tqdm(range(total_num_test_scene)):
            lidar_file = '%06d.bin' % idx
            with open(test_scene_path / lidar_file, 'rb') as f:
                points = np.fromfile(f)
                points = np.reshape(points, (-1, 3))
                points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)
            batch_dict = processor.forward({'points': points, 'use_lead_xyz': True, 'batch_size': 1})
            batch_dict['voxel_coords'] = np.concatenate((np.zeros((batch_dict['voxel_coords'].shape[0], 1)), batch_dict['voxel_coords']), axis=1)
            load_data_to_gpu(batch_dict)

            with torch.no_grad():
                pred_dicts, _ = model(batch_dict)

            result_pkl.append(pred_dicts)
    
    with open('result.pkl', 'wb') as f:
        pickle.dump(result_pkl, f)
        print("result saved.")


if __name__ == '__main__':
    main()
