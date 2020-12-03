import os
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import cv2

from pcdet.models import load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.utils.data_viz import plot_gt_boxes, plot_multiframe_boxes
from pcdet.utils.tracker_for_inference import TrackingManager
from pcdet.datasets.plusai.plusai_bag_dataset import BagMultiframeDataset, BagMultiframeDatasetUnifyLidar


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--bag_file', type=str, default=None, help='specify the bag file to be inferenced')
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

def main():
    log_file = 'log_bag_inference.txt'
    logger = common_utils.create_logger(log_file, rank=0)

    test_set = BagMultiframeDatasetUnifyLidar(cfg.DATA_CONFIG,
                                    bag_path=args.bag_file,
                                    class_names=cfg.CLASS_NAMES)

    # Build network
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # Initialize tracking manager
    tracking_manager = TrackingManager(cfg)

    json_dict = {'objects': []}
    frame_idx = 0
    object_id = 0

    image_resolution = 0.1

    # Save video
    mode = 'multi' if 'STACK_FRAME_SIZE' in cfg.DATA_CONFIG else 'single'
    bev_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # MJPG XVID DIVX
        video_file_name = os.path.join(args.save_path, 'inf_result_{}.avi'.format(args.bag_file.split('/')[-1][:-4]))
        video_output = cv2.VideoWriter(video_file_name, fourcc, 10.0, (int((bev_range[4] - bev_range[1]) / image_resolution),
                                                                       int((bev_range[3] - bev_range[0]) / image_resolution)))
    else:
        image_save_path = os.path.join(args.save_path, 'inf_result_{}'.format(args.bag_file.split('/')[-1]))
        if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)

    logger.info('-----------------Start inference of OpenPCDet with bag: {}-----------------'.format(args.bag_file))
    # Inference with model
    with torch.no_grad():
        # load checkpoint
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.cuda()
        model.eval()

        # start evaluation
        for timestamp, pose, data_dict in test_set:
            odom_tmp = [pose[1][3], pose[1][0], pose[1][1], pose[1][2], pose[0][0], pose[0][1], pose[0][2]]

            timestr = '%0.9f' % timestamp
            timestr = timestr.split('.')

            batch_dict = test_set.collate_batch([data_dict])
            load_data_to_gpu(batch_dict)

            print("Predicting message %0.3f %04d" % (timestamp, frame_idx))
            pred_dicts, _ = model(batch_dict)

            # Update the tracking manager
            tracked_objects = tracking_manager.update_tracking(pred_dicts)

            # det_boxes = tracked_objects['pred_boxes']
            det_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
            scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            # points = batch_dict['points'][:, 1:].cpu().detach().numpy()
            points = data_dict['points']
            if mode == 'multi' and det_boxes.size > 0:
                det_boxes = det_boxes[:, np.newaxis, :].repeat(3, axis=1)
                frame = plot_multiframe_boxes(points, det_boxes, bev_range,
                                              scores=scores, labels=labels,
                                              info='ts: {:.3f}'.format(timestamp))
            else:
                frame = plot_gt_boxes(points, det_boxes, bev_range, ret=True)

            cv2.imshow('debug', frame)
            cv2.waitKey(1)
            # Save video
            if args.save_video:
                video_output.write(frame)
            else:
                cv2.imwrite(os.path.join(image_save_path, '{:0>4d}.png'.format(frame_idx)), frame)

            # Format the det result
            for obj_idx in range(tracked_objects['pred_boxes'].shape[0]):
                # Traverse all objects in json_dict's object list, find the item with same uuid
                FIND_IN_ARCHIVE = False
                for archived_object in json_dict['objects']:
                    if tracked_objects['object_ids'][obj_idx] == int(archived_object['uuid']):
                        bound_info = {'Tr_imu_to_world': {'qw': odom_tmp[0], 'qx': odom_tmp[1],
                                                          'qy': odom_tmp[2], 'qz': odom_tmp[3],
                                                          'x': odom_tmp[4], 'y': odom_tmp[5],
                                                          'z': odom_tmp[6]},
                                      'timestamp': int(timestr[0]),
                                      'timestamp_nano': int(timestr[1]),
                                      'velocity': {'x': 0, 'y': 0, 'z': 0}}
                        obj_loc = tracked_objects['pred_boxes'][obj_idx, :3].tolist()
                        obj_dim = tracked_objects['pred_boxes'][obj_idx, 3:6].tolist()
                        obj_rz = tracked_objects['pred_boxes'][obj_idx, 6].tolist()

                        # Rotate the object center
                        loc_x = obj_loc[0] * math.cos(-obj_rz) - obj_loc[1] * math.sin(-obj_rz)
                        loc_y = obj_loc[0] * math.sin(-obj_rz) + obj_loc[1] * math.cos(-obj_rz)

                        bound_info.update(
                            {'center': {'x': loc_x, 'y': loc_y, 'z': obj_loc[2]},
                             'direction': {'x': 0, 'y': 0, 'z': 0},
                             'heading': obj_rz,
                             'is_front_car': 0,
                             'position': {'x': obj_loc[0], 'y': obj_loc[1], 'z': obj_loc[2]},
                             'size': {'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]},
                             })

                        archived_object['bounds'].append(bound_info)
                        FIND_IN_ARCHIVE = True
                        break

                if FIND_IN_ARCHIVE:
                    continue

                # If not find, create new object info
                new_object_info = {'bounds': [{'Tr_imu_to_world': {'qw': odom_tmp[0], 'qx': odom_tmp[1],
                                                                   'qy': odom_tmp[2], 'qz': odom_tmp[3],
                                                                   'x': odom_tmp[4], 'y': odom_tmp[5],
                                                                   'z': odom_tmp[6]},
                                               'timestamp': int(timestr[0]),
                                               'timestamp_nano': int(timestr[1]),
                                               'velocity': {'x': 0, 'y': 0, 'z': 0}}
                                              ],
                                   'size': {},
                                   'uuid': str(tracked_objects['object_ids'][obj_idx])
                                   }
                obj_loc = tracked_objects['pred_boxes'][obj_idx, :3].tolist()
                obj_dim = tracked_objects['pred_boxes'][obj_idx, 3:6].tolist()
                obj_rz = tracked_objects['pred_boxes'][obj_idx, 6].tolist()

                # Rotate the object center
                loc_x = obj_loc[0] * math.cos(-obj_rz) - obj_loc[1] * math.sin(-obj_rz)
                loc_y = obj_loc[0] * math.sin(-obj_rz) + obj_loc[1] * math.cos(-obj_rz)

                new_object_info['bounds'][0].update(
                    {'center': {'x': loc_x, 'y': loc_y, 'z': obj_loc[2]},
                     'direction': {'x': 1, 'y': 0, 'z': 0},
                     'heading': obj_rz,
                     'is_front_car': 0,
                     'position': {'x': obj_loc[0], 'y': obj_loc[1], 'z': obj_loc[2]},
                     'size': {'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]},
                     })
                new_object_info['size'].update({'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]})
                json_dict['objects'].append(new_object_info)
                object_id += 1
            frame_idx += 1

    if args.save_video:
        video_output.release()
        print("Inference results video saved as {}".format(video_file_name))

    json_txt = json.dumps(json_dict, indent=4)
    json_file_name = os.path.join(args.save_path, args.bag_file.split('/')[-1] + '.json')
    with open(json_file_name, 'w') as f:
        f.write(json_txt)
        print("JSON file saved at {}".format(json_file_name))


if __name__ == '__main__':
    args, cfg = parse_config()
    main()
