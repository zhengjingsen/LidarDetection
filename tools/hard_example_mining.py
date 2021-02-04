import os
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import cv2
import uuid

from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate
from pcdet.utils.data_viz import plot_gt_boxes, plot_multiframe_boxes
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
from pcdet.datasets.plusai.plusai_bag_dataset import BagMultiframeDatasetUnifyLidar
from tracking_utils.tracker import AB3DMOT as TrackingManager

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--bag_path', type=str, default='', help='specify the bag file to be inferenced')
    parser.add_argument('--labeling_path', type=str, default='', help='specify the labeling file path')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for inference')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_path', default='../data/plusai/inference_result/', help='path to save the inference result')
    parser.add_argument('--ckpt', type=str, default=None, help='model checkpoint')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    np.random.seed(1024)

    return args, cfg

def build_bound_info(timestamp, odom, pred_box):
    timestr = '%0.9f' % timestamp
    timestr = timestr.split('.')

    obj_loc = pred_box[:3].tolist()
    obj_dim = pred_box[3:6].tolist()
    obj_rz = pred_box[6].tolist()

    # Rotate the object center
    loc_x = obj_loc[0] * math.cos(-obj_rz) - obj_loc[1] * math.sin(-obj_rz)
    loc_y = obj_loc[0] * math.sin(-obj_rz) + obj_loc[1] * math.cos(-obj_rz)

    bound_info = {'Tr_imu_to_world': {'qw': odom[0], 'qx': odom[1], 'qy': odom[2], 'qz': odom[3],
                                       'x': odom[4], 'y': odom[5], 'z': odom[6]},
                  'timestamp': int(timestr[0]),
                  'timestamp_nano': int(timestr[1]),
                  'velocity': {'x': 0, 'y': 0, 'z': 0},
                  'center': {'x': loc_x, 'y': loc_y, 'z': obj_loc[2]},
                  'direction': {'x': 0, 'y': 0, 'z': 0},
                  'heading': obj_rz,
                  'is_front_car': 0,
                  'position': {'x': obj_loc[0], 'y': obj_loc[1], 'z': obj_loc[2]},
                  'size': {'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]},
                  }
    return bound_info

def save_json_dict(json_dict, json_file_name):
    # generate uuid
    for object in json_dict['objects']:
        object['uuid'] = str(uuid.uuid4())
    json_txt = json.dumps(json_dict, indent=4)
    with open(json_file_name, 'w') as f:
        f.write(json_txt)
        logger.info("JSON file saved at {}".format(json_file_name))

def inference_bag(model, bag_file):
    test_set = BagMultiframeDatasetUnifyLidar(cfg.DATA_CONFIG,
                                    bag_path=bag_file,
                                    class_names=cfg.CLASS_NAMES)

    # Initialize tracking manager
    tracking_manager = TrackingManager(cfg)

    json_dict = {'objects': []}
    frame_idx = 0

    image_resolution = 0.1

    # Save video
    mode = 'multi' if 'STACK_FRAME_SIZE' in cfg.DATA_CONFIG else 'single'
    bev_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # MJPG XVID DIVX
        video_file_name = os.path.join(args.save_path, 'inf_result_{}.avi'.format(bag_file.split('/')[-1][:-4]))
        video_output = cv2.VideoWriter(video_file_name, fourcc, 10.0, (int((bev_range[4] - bev_range[1]) / image_resolution) * 2,
                                                                       int((bev_range[3] - bev_range[0]) / image_resolution)))
    else:
        image_save_path = os.path.join(args.save_path, 'inf_result_{}'.format(bag_file.split('/')[-1]))
        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)

    # start evaluation
    tracking_list = []
    for timestamp, pose, data_dict in test_set:
        odom_tmp = [pose[1][3], pose[1][0], pose[1][1], pose[1][2], pose[0][0], pose[0][1], pose[0][2]]

        batch_dict = test_set.collate_batch([data_dict])
        load_data_to_gpu(batch_dict)

        # print("Predicting message %0.3f %04d" % (timestamp, frame_idx))
        pred_dicts, _ = model(batch_dict)

        det_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
        scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        labels = pred_dicts[0]['pred_labels'].cpu().numpy()
        # points = batch_dict['points'][:, 1:].cpu().detach().numpy()
        points = data_dict['points']
        if mode == 'multi' and det_boxes.size > 0:
            det_boxes = det_boxes[:, np.newaxis, :].repeat(3, axis=1)
            det_frame = plot_multiframe_boxes(points, det_boxes, bev_range,
                                          scores=scores, labels=labels,
                                          info='detect ts: {:.3f}'.format(timestamp))
        else:
            det_frame = plot_gt_boxes(points, det_boxes, bev_range, ret=True)

        # Update the tracking manager
        tracked_objects = tracking_manager.update_tracking(pred_dicts)
        det_boxes = tracked_objects['pred_boxes']
        if mode == 'multi' and det_boxes.size > 0:
            det_boxes = det_boxes[:, np.newaxis, :].repeat(3, axis=1)
            track_frame = plot_multiframe_boxes(points, det_boxes, bev_range,
                                          scores=[cfg.CLASS_NAMES[obj_type-1] for obj_type in tracked_objects['object_types']],
                                          labels=tracked_objects['object_ids'],
                                          info='track ts: {:.3f}'.format(timestamp))
        else:
            track_frame = plot_gt_boxes(points, det_boxes, bev_range, ret=True)

        frame = cv2.hconcat([det_frame, track_frame])
        img_file_name = '{:0>4d}.png'.format(frame_idx)
        cv2.putText(frame, img_file_name, (30, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0, 255, 0), thickness=1)

        # Save video
        if args.save_video:
            video_output.write(frame)
        else:
            cv2.imwrite(os.path.join(image_save_path, img_file_name), frame)

        tracking_list.append({'timestamp': timestamp,
                              'ids': tracked_objects['object_ids'],
                              'pred_boxes': tracked_objects['pred_boxes']})
        # Format the det result
        for obj_idx in range(tracked_objects['pred_boxes'].shape[0]):
            # Traverse all objects in json_dict's object list, find the item with same uuid
            FIND_IN_ARCHIVE = False
            for archived_object in json_dict['objects']:
                if tracked_objects['object_ids'][obj_idx] == int(archived_object['uuid']):
                    archived_object['bounds'].append(build_bound_info(timestamp, odom_tmp,
                                                                      tracked_objects['pred_boxes'][obj_idx]))
                    FIND_IN_ARCHIVE = True
                    break

            if FIND_IN_ARCHIVE:
                continue

            # If not find, create new object info
            new_object_info = {
                'bounds': [build_bound_info(timestamp, odom_tmp, tracked_objects['pred_boxes'][obj_idx])],
                'uuid': str(tracked_objects['object_ids'][obj_idx]),
                'start_frame': frame_idx
            }
            new_object_info.update({'size': new_object_info['bounds'][0]['size']})
            json_dict['objects'].append(new_object_info)
        frame_idx += 1

    if args.save_video:
        video_output.release()
        logger.info("Inference results video saved as {}".format(video_file_name))

    return json_dict, tracking_list

def is_in_pc_range(x, y, z):
    bev_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    return x > bev_range[0] and y > bev_range[1] and z > bev_range[2] and \
           x < bev_range[3] and y < bev_range[4] and z < bev_range[5]

def mining_with_gt(json_dict, tracking_list, labeling_file, logger):
    tracking_list.sort(key=lambda label: label['timestamp'])
    for frame in tracking_list:
        frame.update({'gt_boxes': [], 'gt_ids': []})

    with open(labeling_file) as f:
        labeling = json.load(f, encoding='utf-8')
    for obj in labeling['objects']:
        uuid = obj['uuid']
        for observation in obj['bounds']:
            obs_timestamp = observation['timestamp'] + observation['timestamp_nano'] * 1.e-9
            min_diff = 1e3
            min_idx = -1
            for idx, frame in enumerate(tracking_list):
                ts = frame['timestamp']
                if abs(ts - obs_timestamp) < min_diff:
                    min_diff = abs(ts - obs_timestamp)
                    min_idx = idx
            if min_diff < 0.001 and is_in_pc_range(observation['position']['x'],
                                                   observation['position']['y'],
                                                   observation['position']['z']):
                tracking_list[min_idx]['gt_boxes'].append(
                    np.array([observation['position']['x'], observation['position']['y'], observation['position']['z'],
                              obj['size']['x'], obj['size']['y'], obj['size']['z'], observation['heading']], dtype=np.float32))
                tracking_list[min_idx]['gt_ids'].append(uuid)

    # get gt_statistical_info
    gt_statistical_info = {}
    det_statistical_info = {}
    for idx, frame in enumerate(tracking_list):
        gt_boxes = np.vstack(frame['gt_boxes'])
        pred_boxes = frame['pred_boxes']
        iou_matrix = boxes_iou_bev(torch.from_numpy(gt_boxes).float().cuda(),
                                   torch.from_numpy(pred_boxes).float().cuda()).cpu().numpy()
        for gt_idx in range(gt_boxes.shape[0]):
            gt_id = frame['gt_ids'][gt_idx]
            if gt_id not in gt_statistical_info:
                gt_statistical_info.update({
                    gt_id: {
                        'num_obs': 0,
                        'det_num': [],
                        'ids': set(),
                        'iou': []
                    }})
            gt_statistical_info[gt_id]['num_obs'] += 1

            iou = 0.0
            det_num = 0
            for pred_idx in range(pred_boxes.shape[0]):
                if iou_matrix[gt_idx, pred_idx] > 1e-2:
                    det_num += 1
                    gt_statistical_info[gt_id]['ids'].add(frame['ids'][pred_idx])
                    if iou_matrix[gt_idx, pred_idx] > iou:
                        iou = iou_matrix[gt_idx, pred_idx]

            if det_num > 0:
                gt_statistical_info[gt_id]['det_num'].append(det_num)
                gt_statistical_info[gt_id]['iou'].append(iou)

        for pred_idx in range(pred_boxes.shape[0]):
            pred_id = frame['ids'][pred_idx]
            if pred_id not in det_statistical_info:
                det_statistical_info.update({
                    pred_id: {
                        'num_obs': 0,
                        'split_num': [],
                        'ids': set()
                    }})
            det_statistical_info[pred_id]['num_obs'] += 1

            split_num = 0
            for gt_idx in range(gt_boxes.shape[0]):
                if iou_matrix[gt_idx, pred_idx] > 1e-2:
                    split_num += 1
                    det_statistical_info[pred_id]['ids'].add(frame['gt_ids'][gt_idx])

            if split_num > 0:
                det_statistical_info[pred_id]['split_num'].append(split_num)

    get_hard_example(gt_statistical_info, det_statistical_info, logger)

def get_hard_example(gt_statistical_info, det_statistical_info, logger):
    for key, val in gt_statistical_info.items():
        if len(val['det_num']) > 10:
            # get false negative example
            recall = len(val['det_num']) / val['num_obs']
            if recall < 0.8:
                logger.info('gt_obs {} ids {} recall is too low, det_num: {}, num_obs: {}!'.format(
                    key, val['ids'], len(val['det_num']), val['num_obs']))

            # get big vehicle split into multi-vehicle example
            split_num = 0.
            for num in val['det_num']:
                if num >= 2:
                    split_num += 1
            split_ratio = split_num / len(val['det_num'])
            if split_num > 3 and split_ratio > 0.2:
                logger.info('gt_obs {} ids {} has split detection many times, times: {}, ratio: {}'.format(
                    key, val['ids'], split_num, split_ratio))

            # get low iou examples
            if sum(val['iou']) / len(val['iou']) < 0.6:
                logger.info('gt_obs {} ids {} has low iou with detections!'.format(key, val['ids']))

        # get tracking id switch frequently example
        if len(val['ids']) > 3:
            logger.info('gt_obs {} ids {} is tracked with many different id!'.format(key, val['ids']))

    for key, val in det_statistical_info.items():
        # get false positive example
        if val['num_obs'] > 10:
            precision = len(val['split_num']) / val['num_obs']
            if precision < 0.6:
                logger.info('det_obs {} precision is too low,  true_num: {}, num_obs: {}'.format(
                    key, len(val['split_num']), val['num_obs']))

        # get multi-vehicle merged to one vehicle example
        if len(val['split_num']) > 0:
            split_num = 0.
            for num in val['split_num']:
                if num >= 2:
                    split_num += 1
            split_ratio = split_num / len(val['split_num'])
            if split_num > 3 and split_ratio > 0.2:
                logger.info('det_obs {} merged multi-vehicle into one, times: {}, ratio: {}!'.format(key, split_num, split_ratio))

def mining_without_gt(json_dict, tracking_list, logger):
    for obstacle in json_dict['objects']:
        if len(obstacle['bounds']) > 100:
            continue
        start_pos = np.array([obstacle['bounds'][0]['position']['x'],
                              obstacle['bounds'][0]['position']['y'],
                              obstacle['bounds'][0]['position']['z']], dtype=np.float32)
        end_pos = np.array([obstacle['bounds'][-1]['position']['x'],
                            obstacle['bounds'][-1]['position']['y'],
                            obstacle['bounds'][-1]['position']['z']], dtype=np.float32)

        if start_pos[0] < 30 and (start_pos[1] > 3 or start_pos[1] < -3) and len(obstacle['bounds']) < 20:
            logger.info('det {} locate in near ego left or near ego right, may be missed!'.format(obstacle['uuid']))

        if len(obstacle['bounds']) < 10:
            logger.info('det {} has short tracking trajectory!'.format(obstacle['uuid']))
            continue

        length = []
        for bound in obstacle['bounds']:
            length.append(bound['size']['x'])
        if np.std(np.array(length, dtype=np.float32)) > 2:
            logger.info('det {} length is not stable!'.format(obstacle['uuid']))


import pickle
if __name__ == '__main__':
    args, cfg = parse_config()
    log_file = 'log_hard_example_mining.txt'
    logger = common_utils.create_logger(log_file, rank=0)

    dataset = DatasetTemplate(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger)

    # Build network
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)

    # Gather bag files for inference
    bag_files = []
    if os.path.isfile(args.bag_path):
        bag_files.append(args.bag_path)
    else:
        bags = os.listdir(args.bag_path)
        for bag in bags:
            if bag.endswith('.bag'):
                bag_files.append(os.path.join(args.bag_path, bag))

    # Inference with model
    with torch.no_grad():
        # load checkpoint
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        for bag_file in bag_files:
            logger.info('====== Start process bag {} ======'.format(bag_file))
            # json_dict, tracking_list = inference_bag(model, bag_file)

            # code for debug
            json_dict_file = './json_dict.pkl'
            tracking_list_file = './tracking_list.pkl'
            # with open(json_dict_file, 'wb') as f:
            #     pickle.dump(json_dict, f)
            # with open(tracking_list_file, 'wb') as f:
            #     pickle.dump(tracking_list, f)
            with open(json_dict_file, 'rb') as f:
                json_dict = pickle.load(f)
            with open(tracking_list_file, 'rb') as f:
                tracking_list = pickle.load(f)

            labeling_file = os.path.join(args.labeling_path, bag_file.split('/')[-1] + '.json')
            if os.path.exists(labeling_file) and os.path.isfile(labeling_file):
                mining_with_gt(json_dict, tracking_list, labeling_file, logger)
            else:
                mining_without_gt(json_dict, tracking_list, logger)

    logger.info("All results have been saved in {}".format(args.save_path))