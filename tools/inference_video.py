import argparse
import datetime
import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import pickle
from tqdm import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets.plusai.plusai_bag_dataset import DemoDataset
from pcdet.utils.data_viz import plot_gt_boxes, plot_multiframe_boxes


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the dataset or point cloud data directory')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--save_video_path', type=str, default=None, help='path to save the inference video')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    np.random.seed(1024)

    return args, cfg

def main():
    args, cfg = parse_config()
    log_file = 'log_inference_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logger = common_utils.create_logger(log_file, rank=0)
    logger.info('-----------------Inference of OpenPCDet-------------------------')
    test_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(test_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    if args.save_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(args.save_video_path, 'result.avi'), fourcc, 10.0, (400, 1600))
        bev_range = [-5, -20, -2, 155, 20, 5]

    with torch.no_grad():
        for idx, data_dict in tqdm(enumerate(test_dataset)):
            data_dict = test_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if args.save_video_path is not None:
                boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
                boxes = boxes[:, np.newaxis, :].repeat(3, axis=1)
                gt_boxes = None
                if test_dataset.split == 'val':
                    gt_boxes = test_dataset.val_data_list[idx]['annos']['gt_boxes_lidar']
                    gt_boxes = gt_boxes[:, np.newaxis, :].repeat(3, axis=1)
                image = plot_multiframe_boxes(data_dict['points'][:, 1:].cpu().numpy(),
                                              boxes, bev_range, gt_boxes=gt_boxes)
                cv2.imshow('show_result', image)
                cv2.waitKey(1)
                out.write(image)

    out.release()

if __name__ == '__main__':
    main()
