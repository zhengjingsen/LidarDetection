'''
@Time       : 
@Author     : Jingsen Zheng
@File       : gen_mot_dataset
@Brief      : 
'''

import os
import math
import rosbag
import argparse
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils.common_utils import transform_mtx, create_logger
from pcdet.datasets.plusai.plusai_bag_dataset import BagMultiframeDataset, BagMultiframeDatasetUnifyLidar

def process_single_bag(bag_name, re_unified=True):
  data_path = args.data_path
  save_path = os.path.join(data_path, 'mot_dataset', bag_name)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.mkdir(save_path + '/pointcloud')
    os.mkdir(save_path + '/label')
  else:
    logger.info('{} has been processed, skip this bag!'.format(save_path))
    return 'skip'

  lidar_timestamps = []
  new_labeling = []
  bag_file = os.path.join(data_path, 'bag', bag_name)
  test_set = BagMultiframeDatasetUnifyLidar(cfg.DATA_CONFIG,
                                            bag_path=bag_file,
                                            class_names=cfg.CLASS_NAMES,
                                            stack_frame_size=1,
                                            model_input=False) if re_unified else \
      BagMultiframeDataset(cfg.DATA_CONFIG,
                           bag_path=bag_file,
                           class_names=cfg.CLASS_NAMES,
                           stack_frame_size=1,
                           model_input=False)
  for timestamp, pose, data_dict in test_set:
      timestamp_str = '{:.6f}'.format(timestamp)
      file_name = os.path.join(save_path, 'pointcloud', timestamp_str + '.bin')
      # print('Dump lidar frame {} ...'.format(file_name))
      data_dict['points'][:, :4].tofile(file_name)
      lidar_timestamps.append(timestamp)
      new_labeling.append({'timestamp': timestamp_str,
                           'trans': pose[0],
                           'quat': pose[1],
                           'bag_name': bag_name,
                           'obstacle_list': []})
  if len(new_labeling) == 0:
      return 'no_unifiy_lidar'

  label_file = os.path.join(data_path, 'label', bag_name + '.json')
  with open(label_file) as f:
    labeling = json.load(f, encoding='utf-8')

  new_labeling.sort(key=lambda label: label['timestamp'])
  for obj in labeling['objects']:
    size = np.array([obj['size']['x'], obj['size']['y'], obj['size']['z']], dtype=np.float32)
    uuid = obj['uuid']
    for observation in obj['bounds']:
      obs_timestamp = observation['timestamp'] + observation['timestamp_nano'] * 1.e-9
      min_diff = 1e3
      min_idx = -1
      for idx, ts in enumerate(lidar_timestamps):
        if abs(ts - obs_timestamp) < min_diff:
          min_diff = abs(ts - obs_timestamp)
          min_idx = idx
      observation.update({'size': size, 'uuid': uuid})
      new_labeling[min_idx]['obstacle_list'].append(observation)

  for i in range(1, len(new_labeling) - 1): # since the first and the last frame maybe not labeled, we get rid of them
    label = new_labeling[i]
    label.update({'frame_index': i})
    with open(os.path.join(save_path, 'label', label['timestamp'] + '.pkl'), 'wb') as f:
      pickle.dump(label, f)
  return 'done'

def preprocess_dataset():
    import concurrent.futures as futures
    label_files = os.listdir(os.path.join(args.data_path, 'label'))
    bag_files = os.listdir(os.path.join(args.data_path, 'bag'))

    valid_bags = []
    for label_file in label_files:
        if not label_file.endswith('json'):
            continue
        bag_name = label_file[:-5]
        if bag_name not in bag_files:
          continue
        # print('Processing bag {} ...'.format(bag_name))
        valid_bags.append(bag_name)

    with futures.ThreadPoolExecutor(args.num_workers) as executor:
        results = list(tqdm(executor.map(process_single_bag, valid_bags), total=len(valid_bags)))
        logger.info('Extract data from bag and label result: ', results)

def process_obstacles(obstacles_dict):
  # We will process obstacles from dict to list
  # For those obstacles which maybe lost in some frames,
  # we will make up the obstacle observation with const velocity model
  def process_single_instance(obstacle):
    window_size = len(obstacle)
    left_idx = 0
    right_idx = 0
    for i in range(window_size):
      if obstacle[i]:
        left_idx = i
        right_idx = i+1
        continue
      else:
        while right_idx < window_size - 1 and not obstacle[right_idx]:
          right_idx += 1
        assert obstacle[left_idx] or obstacle[right_idx], 'left_idx: {}, right_idx:{}, window_size: {}'.format(left_idx, right_idx, window_size)
        if obstacle[left_idx] and obstacle[right_idx]:
          obstacle[i].update(obstacle[left_idx])
          ratio = (right_idx - i) / float(right_idx - left_idx)
          obstacle[i].update({'location': obstacle[left_idx]['location'] * ratio + obstacle[right_idx]['location'] * (1. - ratio),
                              'velocity' : obstacle[left_idx]['velocity'] * ratio + obstacle[right_idx]['velocity'] * (1. - ratio)})
          # obstacle[i].update({'heading': math.atan2(obstacle[i]['velocity'][1], obstacle[i]['velocity'][0])})
          obstacle[i].update({'heading': (obstacle[left_idx]['heading'] * ratio + obstacle[right_idx]['heading'] * (1. - ratio))})
        elif obstacle[left_idx] and obstacle[left_idx]['velocity'][0] > -20.0:
          obstacle[i].update(obstacle[left_idx])
          obstacle[i].update(
            {'location': obstacle[left_idx]['location'] + obstacle[left_idx]['velocity'] * 0.1 * (i - left_idx)})     # 0.1 means 100ms
        elif obstacle[right_idx] and obstacle[right_idx]['velocity'][0] > -20.0:
          obstacle[i].update(obstacle[right_idx])
          obstacle[i].update(
            {'location': obstacle[right_idx]['location'] + obstacle[right_idx]['velocity'] * 0.1 * (i - right_idx)})  # 0.1 means 100ms
        else:
            return False
        left_idx = i
    return True

  obstacles = []
  for _, obs in obstacles_dict.items():
    if process_single_instance(obs):
        obstacles.append(obs)
  return obstacles

def get_obstacle_class(obstacle):
  if obstacle['size'][0] < 6.0:
      return 'Car'
  # elif obstacle['size'][0] < 11.0 and obstacle['size'][2] > 3.0:
  #     return 'Bus'
  else:
      return 'Truck'

obstacle_attr = {}
def obstacle_attr_statistics(obstacles):
    global obstacle_attr
    for obs in obstacles:
        class_name = obs[1]['class']
        if class_name not in obstacle_attr:
            obstacle_attr[class_name] = {'size_sum': np.zeros(3, dtype=np.float64), 'bottom_height_sum': 0.0, 'num': 0}
        obstacle_attr[class_name]['size_sum'] += obs[1]['size']
        obstacle_attr[class_name]['bottom_height_sum'] += (obs[1]['location'][2] - obs[1]['size'][2] / 2)
        obstacle_attr[class_name]['num'] += 1

def is_stack_frame_valid(stack_labels):
    max_time_step = 0.15
    for i in range(len(stack_labels) - 1):
        if abs(float(stack_labels[i]['timestamp']) - float(stack_labels[i+1]['timestamp'])) > max_time_step:
            return False
    return True

def prepare_multiframe_scenes(scene_list, data_path):
    stack_frame_size = 3
    base_frame_index = 1    # stack frame is 0, 1, 2, all frames will be transformed to base frame coordinate

    for scene_name in tqdm(scene_list):
        lidar_path = os.path.join(data_path, 'multiframe', scene_name, 'pointcloud')
        if not os.path.exists(lidar_path):
            os.makedirs(lidar_path)
        label_path = os.path.join(data_path, 'multiframe', scene_name, 'label')
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        frame_idx = 0

        file_list = os.listdir(os.path.join(data_path, 'mot_dataset', scene_name, 'label'))
        file_list.sort()
        num_frames = len(file_list)
        for idx1 in range(0, num_frames - stack_frame_size + 1):
            stack_pcds = []
            poses = []
            stack_labels = []
            for idx2 in range(stack_frame_size):
                frame_name = file_list[idx1 + idx2].split('/')[-1][:-4]

                label_file_name = os.path.join(data_path, 'mot_dataset', scene_name, 'label', (frame_name + '.pkl'))
                with open(label_file_name, 'rb') as f:
                    # annos = pickle.load(f, encoding='iso-8859-1')
                    annos = pickle.load(f)

                stack_labels.append(annos)
                poses.append(transform_mtx(annos['trans'], annos['quat']))

                pcd_file_name = os.path.join(data_path, 'mot_dataset', scene_name, 'pointcloud', (frame_name + '.bin'))
                point_cloud = np.fromfile(pcd_file_name, dtype=np.float32).reshape([-1, 4])
                point_cloud = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1), dtype=np.float32) * idx2), axis=-1)
                stack_pcds.append(point_cloud)
            if not is_stack_frame_valid(stack_labels):
                logger.info('Stack frame {} in scene {} is uncontinuous, we will skip this frame!'.format(stack_labels[base_frame_index]['timestamp'], scene_name))
                continue

            final_labels = {'timestamp': stack_labels[base_frame_index]['timestamp'],
                            'trans': stack_labels[base_frame_index]['trans'],
                            'quat': stack_labels[base_frame_index]['quat'],
                            'bag_name': stack_labels[base_frame_index]['bag_name'],
                            'frame_index': stack_labels[base_frame_index]['frame_index'],
                            'obstacles': []}
            obstacles = {}
            for i in range(len(stack_pcds)):
              # transform point cloud and annotation to base_frame coordinate
              delta_pose = np.dot(np.linalg.inv(poses[base_frame_index]), poses[i])
              stack_pcds[i][:, 0:3] = (np.matmul(delta_pose[0:3, 0:3], stack_pcds[i][:, 0:3].T) + delta_pose[0:3, 3:]).T
              for obs in stack_labels[i]['obstacle_list']:
                uuid = obs['uuid']
                if obs['position']['x'] is None or obs['position']['y'] is None or obs['position']['z'] is None:
                  logger.info('WARNING: obs {} in scene {} has invalid position({}, {}, {}), please check!'.format(uuid, scene_name, obs['position']['x'], obs['position']['y'], obs['position']['z']))
                  continue
                if obs['direction']['x'] is None or obs['direction']['y'] is None:
                  logger.info('WARNING: obs {} in scene {} has invalid direction({}, {}), please check!'.format(uuid, scene_name, obs['direction']['x'], obs['direction']['y']))
                  continue
                if obs['velocity']['x'] is None or obs['velocity']['y'] is None or obs['velocity']['z'] is None:
                  logger.info('WARNING: obs {} in scene {} has invalid velocity({}, {}, {}), please check!'.format(uuid, scene_name, obs['velocity']['x'], obs['velocity']['y'], obs['velocity']['z']))
                  velocity = np.array([-100.0, 0.0, 0.0])
                else:
                  velocity = np.matmul(delta_pose[0:3, 0:3],
                                       np.array([obs['velocity']['x'], obs['velocity']['y'], obs['velocity']['z']]).T)
                if not uuid in obstacles:
                  obstacles[uuid] = [{} for _ in range(stack_frame_size)]

                location = np.matmul(delta_pose[0:3, 0:3], np.array([obs['position']['x'], obs['position']['y'], obs['position']['z']]).T) + delta_pose[0:3, 3]
                obstacles[uuid][i] = {'class': get_obstacle_class(obs),
                                      'size': obs['size'],
                                      'is_front_car': obs['is_front_car'],
                                      'location': location,
                                      'heading': math.atan2(obs['direction']['y'], obs['direction']['x']),
                                      'velocity': velocity}
            final_labels['obstacles'] = process_obstacles(obstacles)
            obstacle_attr_statistics(final_labels['obstacles'])

            with open(os.path.join(lidar_path, ('%06d.bin' % frame_idx)), 'wb') as f:
                stack_pcds = np.vstack(stack_pcds)
                stack_pcds.tofile(f)
            with open(os.path.join(label_path, ('%06d.pkl' % frame_idx)), 'wb') as f:
                pickle.dump(final_labels, f)
            frame_idx += 1

            if args.visualize:
              import mayavi.mlab
              fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1080, 1080))
              mayavi.mlab.points3d(stack_pcds[:, 0], stack_pcds[:, 1], stack_pcds[:, 2],
                                   stack_pcds[:, 3],  # Values used for Color
                                   mode="point",
                                   colormap='jet',  # 'bone', 'copper', 'gnuplot', 'spectral'
                                   # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                                   figure=fig,
                                   )
              mayavi.mlab.show()

def prepare_multiframe_dataset():
    data_path = args.data_path

    all_scene_list = os.listdir(os.path.join(data_path, 'mot_dataset'))
    prepare_multiframe_scenes(all_scene_list, data_path)
    
    # random.shuffle(all_scene_list)
    # train_ratio = 0.8
    # boundary = int(train_ratio * len(all_scene_list))
    # train_split = all_scene_list[0:boundary]
    # train_split.sort()
    # test_split = all_scene_list[boundary:]
    # test_split.sort()

    # prepare_multiframe_scenes(train_split, data_path, 'train')
    # prepare_multiframe_scenes(test_split, data_path, 'test')

    # with open(os.path.join(data_path, 'multiframe', 'all_scene_list.txt'), 'w') as f:
    #     f.write('train_scene_list:\n')
    #     for scene in train_split:
    #         f.write(scene + '\n')
    #     f.write('\ntest_scene_list:\n')
    #     for scene in test_split:
    #         f.write(scene + '\n')

    for key, val in obstacle_attr.items():
      mean_size = val['size_sum'] / val['num']
      mean_bottom_height = val['bottom_height_sum'] / val['num']
      logger.info('{} mean size: [{:.2f}, {:.2f}, {:.2f}], mean bottom height: {:.2f}, number: {}'.format(key, mean_size[0], mean_size[1], mean_size[2], mean_bottom_height, val['num']))


def get_images_sets():
  all_scene_list = [
      '20201113T154121_j7-l4e-00011_4_70to90.bag',
      '20201124T151221_j7-l4e-00011_6_121to141.bag',
      '20201124T151221_j7-l4e-00011_27_0to20.bag',
      '20201112T154947_j7-l4e-00011_26_5to25.bag',
      '20201110T163005_j7-l4e-00011_6_93to113.bag',
      '20201112T102738_j7-l4e-00011_12_249to269.bag',
      '20201109T152801_j7-l4e-00011_18_146to166.bag',
      '20201112T102738_j7-l4e-00011_11_172to192.bag',
      '20201112T102738_j7-l4e-00011_11_83to103.bag',
      '20201111T163124_j7-l4e-00011_18_184to204.bag',
      '20201112T154947_j7-l4e-00011_25_191to211.bag',
      '20201113T154121_j7-l4e-00011_0_249to269.bag',
      '20201110T163005_j7-l4e-00011_7_38to58.bag',
      '20201113T154121_j7-l4e-00011_5_193to213.bag',
      '20201124T151221_j7-l4e-00011_25_119to139.bag',
      '20201124T151221_j7-l4e-00011_13_2to22.bag',
      '20201110T163005_j7-l4e-00011_5_219to239.bag',
      '20201124T151221_j7-l4e-00011_4_223to243.bag',
      '20201113T154121_j7-l4e-00011_2_16to36.bag',
      '20201111T163124_j7-l4e-00011_2_22to42.bag',
      '20201111T161204_j7-l4e-00011_3_30to50.bag',
      '20201124T151221_j7-l4e-00011_22_31to51.bag',
      '20201111T163124_j7-l4e-00011_17_198to218.bag',
      '20201110T163005_j7-l4e-00011_7_249to269.bag',
      '20201112T140545_j7-l4e-00011_6_149to169.bag',
      '20201112T154947_j7-l4e-00011_26_27to47.bag',
      '20201124T151221_j7-l4e-00011_27_131to151.bag',
      '20201124T151221_j7-l4e-00011_25_41to61.bag',
      '20201109T152801_j7-l4e-00011_29_251to271.bag',
      '20201109T152801_j7-l4e-00011_36_part000005.bag',
      '20201109T152801_j7-l4e-00011_9_198to218.bag',
      '20201112T102738_j7-l4e-00011_3_207to227.bag',
      '20201112T102738_j7-l4e-00011_14_23to43.bag',
      '20201113T154121_j7-l4e-00011_2_72to92.bag',
      '20201109T152801_j7-l4e-00011_33_167to187.bag',
      '20201113T154121_j7-l4e-00011_1_3to23.bag',
      '20201117T140821_j7-l4e-00011_21_231to251.bag',
      '20201117T140821_j7-l4e-00011_22_144to164.bag',
      '20201124T151221_j7-l4e-00011_3_49to69.bag',
      '20201124T151221_j7-l4e-00011_26_183to203.bag',
      '20201109T152801_j7-l4e-00011_36_part000009.bag',
      '20201111T163124_j7-l4e-00011_0_97to117.bag',
      '20201110T163005_j7-l4e-00011_6_72to92.bag',
      '20201124T151221_j7-l4e-00011_20_242to262.bag',
      '20201111T163124_j7-l4e-00011_0_3to23.bag',
      '20201109T152801_j7-l4e-00011_36_part000007.bag',
      '20201112T140545_j7-l4e-00011_6_0to20.bag',
      '20201124T151221_j7-l4e-00011_22_215to235.bag',
      '20201109T152801_j7-l4e-00011_12_16to36.bag',
      '20201109T152801_j7-l4e-00011_9_33to53.bag',
      '20201111T163124_j7-l4e-00011_18_69to89.bag',
      '20201117T140821_j7-l4e-00011_22_19to39.bag',
      '20201112T102738_j7-l4e-00011_12_188to208.bag',
      '20201113T154121_j7-l4e-00011_0_30to50.bag',
      '20201109T152801_j7-l4e-00011_32_75to95.bag',
      '20201117T140821_j7-l4e-00011_12_153to173.bag',
      '20201124T151221_j7-l4e-00011_28_138to158.bag',
      '20201109T152801_j7-l4e-00011_32_11to31.bag',
      '20201109T152801_j7-l4e-00011_9_256to276.bag',
      '20201109T152801_j7-l4e-00011_36_part000000.bag',
      '20201124T151221_j7-l4e-00011_27_49to69.bag',
      '20201113T154121_j7-l4e-00011_5_145to165.bag',
      '20201124T151221_j7-l4e-00011_27_208to228.bag',
      '20201109T152801_j7-l4e-00011_7_1to21.bag',
      '20201112T154947_j7-l4e-00011_26_74to94.bag',
      '20201124T151221_j7-l4e-00011_4_130to150.bag',
      '20201109T152801_j7-l4e-00011_36_part000008.bag',
      '20201112T140545_j7-l4e-00011_6_179to199.bag',
      '20201112T102738_j7-l4e-00011_12_106to126.bag',
      '20201111T163124_j7-l4e-00011_0_39to59.bag',
      '20201124T151221_j7-l4e-00011_25_0to20.bag',
      '20201109T152801_j7-l4e-00011_27_144to164.bag',
      '20201124T151221_j7-l4e-00011_26_40to60.bag',
      '20201112T140545_j7-l4e-00011_6_108to128.bag',
      '20201111T163124_j7-l4e-00011_9_9to29.bag',
      '20201113T154121_j7-l4e-00011_5_50to70.bag',
      '20201111T163124_j7-l4e-00011_5_99to119.bag',
      '20201112T102738_j7-l4e-00011_11_244to264.bag',
      '20201113T154121_j7-l4e-00011_6_29to49.bag',
      '20201112T102738_j7-l4e-00011_14_5to25.bag',
      '20201124T151221_j7-l4e-00011_28_0to20.bag',
      '20201117T140821_j7-l4e-00011_23_138to158.bag',
      '20201124T151221_j7-l4e-00011_3_189to209.bag',
      '20201113T154121_j7-l4e-00011_3_48to68.bag',
      '20201109T152801_j7-l4e-00011_9_136to156.bag',
      '20201113T154121_j7-l4e-00011_5_110to130.bag',
      '20201112T154947_j7-l4e-00011_19_0to20.bag',
      '20201109T152801_j7-l4e-00011_36_part000002.bag',
      '20201124T151221_j7-l4e-00011_28_26to46.bag',
      '20201110T163005_j7-l4e-00011_6_28to48.bag',
      '20201110T163005_j7-l4e-00011_5_140to160.bag',
      '20201110T163005_j7-l4e-00011_5_187to207.bag',
      '20201109T152801_j7-l4e-00011_36_part000001.bag',
      '20201113T154121_j7-l4e-00011_3_90to110.bag',
      '20201113T154121_j7-l4e-00011_6_0to20.bag',
      '20201124T151221_j7-l4e-00011_24_142to162.bag',
      '20201124T151221_j7-l4e-00011_28_75to95.bag',
      '20201111T163124_j7-l4e-00011_17_108to128.bag',
      '20201117T140821_j7-l4e-00011_20_161to181.bag',
      '20201110T163005_j7-l4e-00011_5_248to268.bag',
      '20201110T163005_j7-l4e-00011_7_132to152.bag',
      '20201109T152801_j7-l4e-00011_18_12to32.bag',
      '20201113T154121_j7-l4e-00011_0_207to227.bag',
      '20201112T154947_j7-l4e-00011_26_127to147.bag',
      '20201110T163005_j7-l4e-00011_5_36to56.bag',
      '20201113T154121_j7-l4e-00011_0_67to87.bag',
      '20201111T163124_j7-l4e-00011_9_51to71.bag',
      '20201111T161204_j7-l4e-00011_3_9to29.bag',
      '20201113T154121_j7-l4e-00011_4_210to230.bag',
      '20201111T163124_j7-l4e-00011_9_115to135.bag',
      '20201113T154121_j7-l4e-00011_3_134to154.bag',
      '20201112T102738_j7-l4e-00011_11_120to140.bag',
      '20201124T151221_j7-l4e-00011_29_1to21.bag',
      '20201111T163124_j7-l4e-00011_18_8to28.bag',
      '20201124T151221_j7-l4e-00011_26_241to261.bag',
      '20201109T152801_j7-l4e-00011_36_part000011.bag',
      '20201109T152801_j7-l4e-00011_18_57to77.bag',
      '20201111T161204_j7-l4e-00011_2_50to70.bag',
      '20201111T163124_j7-l4e-00011_5_38to58.bag',
      '20201110T163005_j7-l4e-00011_6_0to20.bag',
      '20201111T163124_j7-l4e-00011_2_163to183.bag',
      '20201112T102738_j7-l4e-00011_7_0to19.bag',
      '20201117T140821_j7-l4e-00011_23_81to101.bag',
      '20201112T102738_j7-l4e-00011_8_232to252.bag',
      '20201110T163005_j7-l4e-00011_7_205to225.bag',
      '20201109T152801_j7-l4e-00011_36_part000003.bag']

  random.shuffle(all_scene_list)
  train_ratio = 0.8
  boundary = int(train_ratio * len(all_scene_list))
  train_split = all_scene_list[0:boundary]
  train_split.sort()
  test_split = all_scene_list[boundary:]
  test_split.sort()

  select_frame_step = 4
  image_sets_path = os.path.join(args.data_path, 'multiframe', 'ImageSets')
  if not os.path.exists(image_sets_path):
    os.makedirs(image_sets_path)

  with open(os.path.join(image_sets_path, 'train.txt'), 'w') as f:
    for scene in train_split:
      idx = 0
      frame_list = os.listdir(os.path.join(args.data_path, 'multiframe', scene, 'pointcloud'))
      frame_list.sort()
      while idx < len(frame_list):
        f.write(scene + '/pointcloud/' + frame_list[idx] + '\n')
        idx += select_frame_step
    f.close()
  
  with open(os.path.join(image_sets_path, 'val.txt'), 'w') as f:
    for scene in test_split:
      idx = 0
      frame_list = os.listdir(os.path.join(args.data_path, 'multiframe', scene, 'pointcloud'))
      frame_list.sort()
      while idx < len(frame_list):
        f.write(scene + '/pointcloud/' + frame_list[idx] + '\n')
        idx += select_frame_step
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='directory to data path which should contains bag and label')
    parser.add_argument('--calib_name', default='', help='lidar calib name')
    parser.add_argument('--calib_date', default='', help='lidar calib date')
    parser.add_argument('--calib_dir', default='/opt/plusai/var/calib_db', help='directory to calib db')
    parser.add_argument('--lidar_topic', default='/unified/lidar_points')
    parser.add_argument('--odom_topic', default='/navsat/odom')
    parser.add_argument('--cfg_file', type=str, default='/home/jingsen/workspace/OpenPCDet/tools/cfgs/livox_models/pv_rcnn_multiframe.yaml')
    parser.add_argument('--visualize', action='store_true', default=False, help='visualize the multi-frame point cloud')
    parser.add_argument('--num_workers', default=6, help='num workers to process label data')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    log_file = os.path.join(args.data_path, 'data_preprocessing_log.txt')
    logger = create_logger(log_file, rank=0)

    # logger.info('=== Start extract point-cloud and annotations from origin bag and label files, this will take a long time ... ===')
    # preprocess_dataset()

    logger.info('\n\n=== Start process multiframe dataset ... ===')
    prepare_multiframe_dataset()

    logger.info('\n\n=== Start get image sets ... ===')
    get_images_sets()

    print('log file saved in {}'.format(log_file))