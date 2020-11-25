import argparse
import pickle
import os
import math
import io as sysio
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.utils.data_viz import plot_gt_boxes, plot_gt_det_cmp
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.kitti.kitti_object_eval_python.eval import bev_box_overlap


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    np.random.seed(1024)

    return args, cfg


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def get_metrics(gt_boxes, det_boxes, range_thres, iou_thres):
    num_valid_det = 0
    num_valid_gt = 0
    tp = 0
    dist_err = 0.0
    invalid_gt_idx = []
    for idx in range(det_boxes.shape[0]):
        if det_boxes[idx, 0] > range_thres:
            continue
        num_valid_det += 1

    for idx in range(gt_boxes.shape[0]):
        if gt_boxes[idx, 0] > range_thres:
            invalid_gt_idx.append(idx)
            continue
        num_valid_gt += 1

    if not (gt_boxes.shape[0] and det_boxes.shape[0]):
        return tp, num_valid_det, num_valid_gt, dist_err

    # Calculate overlaps
    gt_boxes_bev = gt_boxes[:, [0, 1, 3, 4, 6]]
    det_boxes_bev = det_boxes[:, [0, 1, 3, 4, 6]]
    overlaps = bev_box_overlap(gt_boxes_bev, det_boxes_bev)
    if overlaps.shape[1]:
        overlaps_reduced = np.max(overlaps, axis=1)
        overlaps_reduced[invalid_gt_idx] = 0
        tp = np.where(overlaps_reduced >= iou_thres)
        tp = tp[0].size

        # Calculate distance error of bounding box in x axis
        for idx in range(overlaps.shape[1]):
            det_box_loc = det_boxes[idx, 0] - det_boxes[idx, 3] / 2
            if np.max(overlaps[:, idx]) < iou_thres or det_boxes[idx, 0] > range_thres:
                continue
            related_gt_box_idx = np.argmax(overlaps[:, idx])
            gt_box_loc = gt_boxes[related_gt_box_idx, 0] - gt_boxes[related_gt_box_idx, 3] / 2
            dist_err += abs(det_box_loc - gt_box_loc)

    return tp, num_valid_det, num_valid_gt, dist_err


def main():
    args, cfg = parse_config()
    total_gpus = 1

    assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
    args.batch_size = args.batch_size // total_gpus

    log_file = 'log_mot_benchmark.txt'
    logger = common_utils.create_logger(log_file, rank=0)

    test_set, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, training=False
    )

    # Load the data
    mot_dataset_path = Path("/home/yao.xu/datasets/mot_dataset")
    test_scene_list = os.listdir(mot_dataset_path)
    test_scene_list.sort()
    test_scene_list = test_scene_list[0:1]

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    with torch.no_grad():
        # load checkpoint
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        # start inference
        class_names = cfg.CLASS_NAMES
        point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        processor = DataProcessor(cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range, training=False)

        # Initialize evaluation metrics
        dist_ranges = [50, 100, 150]
        ious = [0.3, 0.5, 0.7]
        total_num_valid_det = np.zeros((len(ious), len(dist_ranges)))
        total_num_valid_gt = np.zeros((len(ious), len(dist_ranges)))
        total_num_tp = np.zeros((len(ious), len(dist_ranges)))
        total_dist_err = np.zeros((len(ious), len(dist_ranges)))

        frame_idx = 0
        for test_scene in tqdm(test_scene_list):
            test_scene_path = mot_dataset_path / test_scene
            test_frame_list = os.listdir(test_scene_path / 'pointcloud')

            for lidar_file in tqdm(test_frame_list):
                with open(test_scene_path / 'pointcloud' / lidar_file, 'rb') as f:
                    points = np.fromfile(f, dtype=np.float32)
                    points = np.reshape(points, (-1, 4))[:, :3]
                    points = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)
                batch_dict = processor.forward({'points': points, 'use_lead_xyz': True, 'batch_size': 1})
                batch_dict['points'] = np.concatenate(
                    (np.zeros((batch_dict['points'].shape[0], 1)), batch_dict['points']), axis=1)
                batch_dict['voxel_coords'] = np.concatenate(
                    (np.zeros((batch_dict['voxel_coords'].shape[0], 1)), batch_dict['voxel_coords']), axis=1)
                load_data_to_gpu(batch_dict)

                pred_dicts, _ = model(batch_dict)
                det_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()

                # Load annotation from pickle file
                gt_boxes = []
                label_file = lidar_file.replace('bin', 'pkl')
                try:
                    assert (test_scene_path / 'label' / label_file).exists()
                except AssertionError:
                    continue
                with open(test_scene_path / 'label' / label_file, 'rb') as f:
                    anno = pickle.load(f, encoding='iso-8859-1')
                    for obj in anno['obstacle_list']:
                        loc = np.array([obj['position']['x'], obj['position']['y'], obj['position']['z']])
                        dims = obj['size']
                        rotz = np.array([math.atan(obj['direction']['y'] / obj['direction']['x'])])
                        if loc[0] < point_cloud_range[0] or loc[0] > point_cloud_range[3] \
                                or loc[1] < point_cloud_range[1] or loc[1] > point_cloud_range[4]:
                            continue
                        gt_boxes.append(np.concatenate((loc, dims, rotz), axis=0))
                gt_boxes = np.array(gt_boxes)

                ###########################Plot DET results###############################
                PLOT_BOX = False
                if PLOT_BOX:
                    points = batch_dict['points'][:, 1:4].cpu().detach().numpy()
                    bev_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
                    # plot_gt_boxes(points, det_boxes, bev_range, name="mot_bench_%04d" % idx)
                    plot_gt_det_cmp(points, gt_boxes, det_boxes, bev_range, name="mot_bench_%04d" % frame_idx)
                ##########################################################################

                # Evaluate current frame
                for iou_idx in range(len(ious)):
                    for dist_range_idx in range(len(dist_ranges)):
                        tp, num_valid_det, num_valid_gt, dist_err = get_metrics(gt_boxes, det_boxes,
                                                                                dist_ranges[dist_range_idx],
                                                                                ious[iou_idx])
                        total_num_tp[iou_idx, dist_range_idx] += tp
                        total_num_valid_det[iou_idx, dist_range_idx] += num_valid_det
                        total_num_valid_gt[iou_idx, dist_range_idx] += num_valid_gt
                        total_dist_err[iou_idx, dist_range_idx] += dist_err

                frame_idx += 1

    # Print benchmark results
    avg_precision = total_num_tp / total_num_valid_det * 100
    avg_recall = total_num_tp / total_num_valid_gt * 100
    avg_dist_err = total_dist_err / total_num_tp

    print("================== Evaluation Results ==================")
    result = ''
    result += print_str(
        (f"Precision "
         "@ {:.1f}m, {:.1f}m, {:.1f}m:".format(*dist_ranges)))
    result += print_str((f"IoU    0.3: {avg_precision[0, 0]:.4f}, "
                         f"{avg_precision[0, 1]:.4f}, "
                         f"{avg_precision[0, 2]:.4f}"))
    result += print_str((f"IoU    0.5: {avg_precision[1, 0]:.4f}, "
                         f"{avg_precision[1, 1]:.4f}, "
                         f"{avg_precision[1, 2]:.4f}"))
    result += print_str((f"IoU    0.7: {avg_precision[2, 0]:.4f}, "
                         f"{avg_precision[2, 1]:.4f}, "
                         f"{avg_precision[2, 2]:.4f}"))
    result += "--------------------------------------------------------\n"
    result += print_str(
        (f"Recall    "
         "@ {:.1f}m, {:.1f}m, {:.1f}m:".format(*dist_ranges)))
    result += print_str((f"IoU    0.3: {avg_recall[0, 0]:.4f}, "
                         f"{avg_recall[0, 1]:.4f}, "
                         f"{avg_recall[0, 2]:.4f}"))
    result += print_str((f"IoU    0.5: {avg_recall[1, 0]:.4f}, "
                         f"{avg_recall[1, 1]:.4f}, "
                         f"{avg_recall[1, 2]:.4f}"))
    result += print_str((f"IoU    0.7: {avg_recall[2, 0]:.4f}, "
                         f"{avg_recall[2, 1]:.4f}, "
                         f"{avg_recall[2, 2]:.4f}"))
    result += "--------------------------------------------------------\n"
    result += print_str(
        (f"Dist Error"
         "@ {:.1f}m, {:.1f}m, {:.1f}m:".format(*dist_ranges)))
    result += print_str((f"IoU    0.3: {avg_dist_err[0, 0]:.4f}, "
                         f"{avg_dist_err[0, 1]:.4f}, "
                         f"{avg_dist_err[0, 2]:.4f}"))
    result += print_str((f"IoU    0.5: {avg_dist_err[1, 0]:.4f}, "
                         f"{avg_dist_err[1, 1]:.4f}, "
                         f"{avg_dist_err[1, 2]:.4f}"))
    result += print_str((f"IoU    0.7: {avg_dist_err[2, 0]:.4f}, "
                         f"{avg_dist_err[2, 1]:.4f}, "
                         f"{avg_dist_err[2, 2]:.4f}"))

    print(result)


if __name__ == '__main__':
    main()
