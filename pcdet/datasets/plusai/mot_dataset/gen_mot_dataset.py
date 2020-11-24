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
import bisect
import json
import pickle
import random
import quaternion
from tqdm import tqdm
import numpy as np
import sensor_msgs.point_cloud2 as pc2

def interpolate_pose(pose1, pose2, t1, t2, t_out):
  tau = (t_out-t1) / (t2-t1)
  trans = (1-tau) * pose1[0] + tau * pose2[0]
  quat = quaternion.slerp(np.quaternion(pose1[1][3], pose1[1][0], pose1[1][1], pose1[1][2]),
                          np.quaternion(pose2[1][3], pose2[1][0], pose2[1][1], pose2[1][2]),
                          t1, t2, t_out)
  return (trans, np.array([quat.x, quat.y, quat.z, quat.w]))

def get_best_pose(timestamp, poses):
  timestamps, poses = poses
  after_i = bisect.bisect_left(timestamps, timestamp)
  # print "timestamp is", timestamp, "after_i is", after_i, "top stamps are", timestamps[:2]
  before_i = max(0, after_i - 1)
  after_time = timestamps[after_i]
  before_time = timestamps[before_i]
  if after_time - before_time >= 0.02:
    print("warning, hole of size", after_time - before_time)
  if before_i == after_i:
    # print "beep"
    return poses[before_i]
  return interpolate_pose(poses[before_i], poses[after_i], before_time, after_time, timestamp)

def process_single_bag(bag_name, data_path):
  save_path = os.path.join(data_path, 'mot_dataset', bag_name)
  if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.mkdir(save_path + '/pointcloud')
    os.mkdir(save_path + '/label')

  bag_file = os.path.join(data_path, 'bag', bag_name)
  label_file = os.path.join(data_path, 'label', bag_name + '.json')
  bag = rosbag.Bag(bag_file)

  odom_list = []
  for topic, msg, _ in bag.read_messages(topics=args.odom_topic):
    timestamp = msg.header.stamp.to_sec()
    pos = np.array([msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z])
    quat = np.array([msg.pose.pose.orientation.x,
                     msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z,
                     msg.pose.pose.orientation.w])
    odom_list.append((timestamp, (pos, quat)))
  odom_list = sorted(odom_list)
  timestamps = [e[0] for e in odom_list]
  poses = [e[1] for e in odom_list]

  lidar_timestamps = []
  new_labeling = []
  for topic, msg, _ in bag.read_messages(topics=args.lidar_topic):
    timestamp = msg.header.stamp.to_sec()
    lidar_timestamps.append(timestamp)
    timestamp_str = '{:.6f}'.format(timestamp)
    lidar_pts_unified = pc2.read_points(msg)
    lidar_pts_unified = np.array(list(lidar_pts_unified), dtype=np.float32)[:, :4]
    intensity = lidar_pts_unified[:, 3].copy()
    lidar_pts_unified[:, 3] = 1.
    lidar_pts_unified = np.matmul(lidar_pts_unified, Tr_lidar_to_imu.T)
    lidar_pts_unified[:, 3] = intensity
    file_name = os.path.join(save_path, 'pointcloud', timestamp_str + '.bin')
    # print('Dump lidar frame {} ...'.format(file_name))
    lidar_pts_unified.tofile(file_name)
    pose = get_best_pose(timestamp, (timestamps, poses))
    new_labeling.append({'timestamp': timestamp_str,
                         'trans': pose[0],
                         'quat': pose[1],
                         'bag_name': bag_name,
                         'obstacle_list': []})

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

def preprocess_dataset():
  label_files = os.listdir(os.path.join(args.data_path, 'label'))
  bag_files = os.listdir(os.path.join(args.data_path, 'bag'))
  for label_file in tqdm(label_files):
    if not label_file.endswith('json'):
      continue
    bag_name = label_file[:-5]
    if bag_name not in bag_files:
      continue
    # print('Processing bag {} ...'.format(bag_name))
    process_single_bag(bag_name, args.data_path)

def transform_mtx(trans, quat):
  pose = np.eye(4)
  pose[0:3, 0:3] = quaternion.as_rotation_matrix(np.quaternion(quat[3], quat[0], quat[1], quat[2]))
  pose[:3, 3] = trans
  return pose

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
        elif obstacle[left_idx]:
          obstacle[i].update(obstacle[left_idx])
          obstacle[i].update(
            {'location': obstacle[left_idx]['location'] + obstacle[left_idx]['velocity'] * 0.1 * (i - left_idx)})     # 0.1 means 100ms
        elif obstacle[right_idx]:
          obstacle[i].update(obstacle[right_idx])
          obstacle[i].update(
            {'location': obstacle[right_idx]['location'] + obstacle[right_idx]['velocity'] * 0.1 * (i - right_idx)})  # 0.1 means 100ms
        left_idx = i

  obstacles = []
  for _, obs in obstacles_dict.items():
    process_single_instance(obs)
    obstacles.append(obs)
  return obstacles

def get_obstacle_class(obstacle):
  if obstacle['size'][0] < 5.0:
      return 'Car'
  elif obstacle['size'][0] < 11.0 and obstacle['size'][2] > 3.0:
      return 'Bus'
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

def prepare_multiframe_dataset():
    stack_frame_size = 3
    base_frame_index = 1    # stack frame is 0, 1, 2, all frames will be transformed to base frame coordinate
    data_path = args.data_path

    lidar_path = os.path.join(data_path, 'multiframe', 'training', 'pointcloud')
    if not os.path.exists(lidar_path):
      os.makedirs(lidar_path)
    label_path = os.path.join(data_path, 'multiframe', 'training', 'label')
    if not os.path.exists(label_path):
      os.makedirs(label_path)
    frame_idx = 0

    for scene_name in tqdm(os.listdir(os.path.join(data_path, 'mot_dataset'))):
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
                if not uuid in obstacles:
                  obstacles[uuid] = [{} for _ in range(stack_frame_size)]

                location = np.matmul(delta_pose[0:3, 0:3], np.array([obs['position']['x'], obs['position']['y'], obs['position']['z']]).T) + delta_pose[0:3, 3]
                velocity = np.matmul(delta_pose[0:3, 0:3], np.array([obs['velocity']['x'], obs['velocity']['y'], obs['velocity']['z']]).T)
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

    for key, val in obstacle_attr.items():
      mean_size = val['size_sum'] / val['num']
      mean_bottom_height = val['bottom_height_sum'] / val['num']
      print('{} mean size: [{:.2f}, {:.2f}, {:.2f}], mean bottom height: {:.2f}, number: {}'.format(key, mean_size[0], mean_size[1], mean_size[2], mean_bottom_height, val['num']))

def get_images_sets():
  train_ratio = 0.7
  image_sets_path = os.path.join(args.data_path, 'multiframe', 'ImageSets')
  if not os.path.exists(image_sets_path):
    os.makedirs(image_sets_path)

  frame_list = os.listdir(os.path.join(args.data_path, 'multiframe', 'training', 'pointcloud'))
  random.shuffle(frame_list)

  boundary = int(train_ratio * len(frame_list))
  train_split = frame_list[0:boundary]
  train_split.sort()
  test_split = frame_list[boundary:]
  test_split.sort()

  with open(os.path.join(image_sets_path, 'train.txt'), 'w') as f:
    for frame in train_split:
      f.write(frame.split('.')[0] + '\n')
    f.close()
  with open(os.path.join(image_sets_path, 'val.txt'), 'w') as f:
    for frame in test_split:
      f.write(frame.split('.')[0] + '\n')
    f.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', help='directory to data path which should contains bag and label')
  parser.add_argument('--calib_name', default='', help='lidar calib name')
  parser.add_argument('--calib_date', default='', help='lidar calib date')
  parser.add_argument('--calib_dir', default='/opt/plusai/var/calib_db', help='directory to calib db')
  parser.add_argument('--lidar_topic', default='/unified/lidar_points')
  parser.add_argument('--odom_topic', default='/navsat/odom')
  parser.add_argument('--visualize', action='store_true', default=False, help='visualize the multi-frame point cloud')
  args = parser.parse_args()

  # TODO: load lidar extrinsic from calib file
  Tr_lidar_to_imu = np.array([[9.7485195858863372e-01, -4.5840773586776476e-02, 2.1808808322147416e-01, 4.4884194774399999e+00],
                              [4.1536075689053195e-02, 9.9884171056766391e-01, 2.4284555619455646e-02, -1.9965142422800002e-02],
                              [-2.1894868262845490e-01, -1.4615340755379213e-02, 9.7562682116515986e-01, 2.8337476145100000e+00],
                              [0., 0., 0., 1.]], dtype=np.float32)

  # preprocess_dataset()
  prepare_multiframe_dataset()
  get_images_sets()