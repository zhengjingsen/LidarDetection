import argparse
import datetime
import os
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
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    np.random.seed(1024)

    return args, cfg


def main():
    args, cfg = parse_config()
    total_gpus = 1

    assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
    args.batch_size = args.batch_size // total_gpus

    log_file = 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = common_utils.create_logger(log_file, rank=0)

    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, training=False
    )

    # Load the data
    test_scene_list = os.listdir(test_scene_path)
    total_num_test_scene = len(test_scene_list)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        # load checkpint
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
            batch_dict['points'] = np.concatenate((np.zeros((batch_dict['points'].shape[0], 1)), batch_dict['points']), axis=1)
            batch_dict['voxel_coords'] = np.concatenate((np.zeros((batch_dict['voxel_coords'].shape[0], 1)), batch_dict['voxel_coords']), axis=1)
            load_data_to_gpu(batch_dict)

            pred_dicts, _ = model(batch_dict)

            # ------------------------------ Plot DET results ------------------------------
            PLOT_BOX = False
            if PLOT_BOX:
                points = batch_dict['points'][:, 1:4].cpu().detach().numpy()
                det_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
                bev_range = [0, -8, -2, 152, 8, 6]
                plot_gt_boxes(points, det_boxes, bev_range, name="%04d" % idx)
            # -------------------------------------------------------------------------------


if __name__ == '__main__':
    test_scene_path = Path("/home/tong.wang/datasets/test_scene")
    main()
