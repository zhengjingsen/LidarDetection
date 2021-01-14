'''
@Time       : 
@Author     : Jingsen Zheng
@File       : plusai_bag_dataset
@Brief      : 
'''

import glob
import rosbag
import pickle
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate

class UnifyLidar(object):
    def __init__(self, bag_info_cfg, bag):
        from pcdet.utils.calibration_plusai import load_lidar_calib
        car = bag_info_cfg.CAR
        calib_db_path = bag_info_cfg.CALIB_DB_PATH

        self.lidar_topic_list = []
        self.is_main_lidar = []
        self.lidar_extrinsic_list = []
        for lidar_cfg in bag_info_cfg.UNIFIED_LIDAR:
            self.lidar_topic_list.append(lidar_cfg['topic'])
            self.is_main_lidar.append(lidar_cfg['is_main_lidar'])
            self.lidar_extrinsic_list.append(load_lidar_calib(car,
                                                              lidar_cfg['calib_name'],
                                                              lidar_cfg['calib_date'],
                                                              calib_db_path))
        self.buffer_size = 10
        self.time_diff_thresh = 0.02
        self.frame_buffer = []
        self.data_iter = bag.read_messages(topics=self.lidar_topic_list)

    def is_frame_ready(self, frame):
        for is_ready in frame['is_ready']:
            if not is_ready:
                return False
        return True

    def add_msg(self, topic, msg):
        idx = self.lidar_topic_list.index(topic)
        timestamp = msg.header.stamp.to_sec()

        point_cloud = pc2.read_points(msg)
        point_cloud = np.array(list(point_cloud), dtype=np.float32)[:, :4]
        intensity = point_cloud[:, 3].copy()
        point_cloud[:, 3] = 1.
        point_cloud = np.matmul(point_cloud, self.lidar_extrinsic_list[idx].T)
        point_cloud[:, 3] = intensity

        cur_frame = None
        min_time_diff = 1e3
        for frame in self.frame_buffer:
            time_diff = abs(timestamp - frame['timestamp'])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                cur_frame = frame

        if min_time_diff > self.time_diff_thresh:
            cur_frame = {
                'timestamp': timestamp,
                'is_ready': [False for _ in self.lidar_topic_list],
                'pointcloud': [None for _ in self.lidar_topic_list]
            }
            self.frame_buffer.append(cur_frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)

        cur_frame['is_ready'][idx] = True
        cur_frame['pointcloud'][idx] = point_cloud
        if self.is_main_lidar[idx]:
            cur_frame.update({'timestamp': timestamp})
        if self.is_frame_ready(cur_frame):
            return (cur_frame['timestamp'], np.vstack(cur_frame['pointcloud']))
        else:
            return None

    def next(self):
        unified_lidar = None
        while unified_lidar is None:
            try:
                (topic, msg, _) = next(self.data_iter)
                unified_lidar = self.add_msg(topic, msg)
            except StopIteration as e:
                break
        return unified_lidar


class BagMultiframeDatasetUnifyLidar(DatasetTemplate):
    def __init__(self, dataset_cfg, bag_path, class_names, training=False, logger=None,
                 stack_frame_size=-1, model_input=True):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger)

        self.frame_idx = 0
        self.bag_path = bag_path
        self.max_time_step = 0.15
        self.end_flag = False
        self.model_input = model_input
        if str(bag_path).endswith('.bag'):
            self.bag = rosbag.Bag(bag_path, 'r')
            odom_list = []
            for topic, msg, _ in self.bag.read_messages(topics=dataset_cfg.BAG_INFO.ODOM_TOPIC):
                timestamp = msg.header.stamp.to_sec()
                pos = np.array([msg.pose.pose.position.x,
                                msg.pose.pose.position.y,
                                msg.pose.pose.position.z])
                quat = np.array([msg.pose.pose.orientation.x,
                                 msg.pose.pose.orientation.y,
                                 msg.pose.pose.orientation.z,
                                 msg.pose.pose.orientation.w])
                odom_list.append((timestamp, (pos, quat)))
            odom_list = sorted(odom_list, key=lambda x : x[0])
            self.timestamps = [e[0] for e in odom_list]
            self.poses = [e[1] for e in odom_list]

            if stack_frame_size > 0:
                self.stack_frame_size = stack_frame_size
            elif dataset_cfg.get('STACK_FRAME_SIZE', False):
                self.stack_frame_size = dataset_cfg.STACK_FRAME_SIZE
            else:
                self.stack_frame_size = 1
            self.base_frame_index = self.stack_frame_size // 2
            self.frame_list = []
            self.data_iter = UnifyLidar(dataset_cfg.BAG_INFO, self.bag)

            self.fill_frame_list()
        else:
            raise NotImplementedError

    def fill_frame_list(self):
        while len(self.frame_list) < self.stack_frame_size:
            unified_lidar = self.data_iter.next()
            if unified_lidar is not None:
                # check lidar topic continuity
                if len(self.frame_list) > 0:
                    last_timestamp = self.frame_list[-1][0]
                    if abs(last_timestamp - unified_lidar[0]) > self.max_time_step:
                        print('Some lidar topic maybe drop in bag {}, we will skip these un-continuous topic!'.format(self.bag_path.split('/')[-1]))
                        self.frame_list = []
                pose = common_utils.get_best_pose(unified_lidar[0], (self.timestamps, self.poses))
                self.frame_list.append((unified_lidar[0], pose, unified_lidar[1]))
            else:
                self.end_flag = True
                break

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_flag:
            self.bag.close()
            raise StopIteration

        base_frame = self.frame_list[self.base_frame_index]
        trans = base_frame[1][0]
        quat = base_frame[1][1]
        base_frame_inv_pose = np.linalg.inv(common_utils.transform_mtx(trans, quat))
        stack_pcds = []
        for idx, frame in enumerate(self.frame_list):
            cur_pointcloud = frame[2].copy()
            cur_pointcloud = np.concatenate((cur_pointcloud, np.ones((cur_pointcloud.shape[0], 1), dtype=np.float32) * idx),
                                            axis=-1)
            # transform point cloud and annotation to base_frame coordinate
            delta_pose = np.dot(base_frame_inv_pose, common_utils.transform_mtx(frame[1][0], frame[1][1]))
            cur_pointcloud[:, 0:3] = (np.matmul(delta_pose[0:3, 0:3], cur_pointcloud[:, 0:3].T) + delta_pose[0:3, 3:]).T
            stack_pcds.append(cur_pointcloud)
        point_cloud = np.vstack(stack_pcds)

        self.frame_list.pop(0)
        self.fill_frame_list()

        input_dict = {
            'points': point_cloud,
            'frame_id': self.frame_idx,
        }

        self.frame_idx += 1
        if self.model_input:
            input_dict = self.prepare_data(data_dict=input_dict)
        return base_frame[0], base_frame[1], input_dict


class BagMultiframeDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, bag_path, class_names, training=False, logger=None,
                 stack_frame_size=-1, model_input=True):
        from pcdet.utils.calibration_plusai import load_lidar_calib
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger)

        self.frame_idx = 0
        self.end_flag = True
        self.model_input = model_input

        bag_info_cfg = dataset_cfg.BAG_INFO
        self.Tr_lidar_to_imu = np.eye(4, dtype=np.float32)
        for lidar_cfg in bag_info_cfg.UNIFIED_LIDAR:
            if lidar_cfg['is_main_lidar']:
                self.Tr_lidar_to_imu = load_lidar_calib(bag_info_cfg.CAR,
                                                        lidar_cfg['calib_name'],
                                                        lidar_cfg['calib_date'],
                                                        bag_info_cfg.CALIB_DB_PATH)
                break

        if str(bag_path).endswith('.bag'):
            self.bag = rosbag.Bag(bag_path, 'r')
            odom_list = []
            for topic, msg, _ in self.bag.read_messages(topics=dataset_cfg.BAG_INFO.ODOM_TOPIC):
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
            self.timestamps = [e[0] for e in odom_list]
            self.poses = [e[1] for e in odom_list]

            if stack_frame_size > 0:
                self.stack_frame_size = stack_frame_size
            elif dataset_cfg.get('STACK_FRAME_SIZE', False):
                self.stack_frame_size = dataset_cfg.STACK_FRAME_SIZE
            else:
                self.stack_frame_size = 1
            self.base_frame_index = self.stack_frame_size // 2
            self.frame_list = []
            self.data_iter = self.bag.read_messages(topics=dataset_cfg.BAG_INFO.UNIFIED_LIDAR_TOPIC)
            for i in range(self.stack_frame_size):
                (topic, msg, _) = next(self.data_iter)
                self.frame_list.append(self.read_lidar_topic(msg))
            self.end_flag = False
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def read_lidar_topic(self, msg):
        timestamp = msg.header.stamp.to_sec()
        lidar_pts_unified = pc2.read_points(msg)
        lidar_pts_unified = np.array(list(lidar_pts_unified), dtype=np.float32)[:, :4]
        intensity = lidar_pts_unified[:, 3].copy()
        lidar_pts_unified[:, 3] = 1.
        lidar_pts_unified = np.matmul(lidar_pts_unified, self.Tr_lidar_to_imu.T)
        lidar_pts_unified[:, 3] = intensity
        pose = common_utils.get_best_pose(timestamp, (self.timestamps, self.poses))
        return (timestamp, pose, lidar_pts_unified)

    def __next__(self):
        if self.end_flag:
            raise StopIteration

        base_frame = self.frame_list[self.base_frame_index]
        trans = base_frame[1][0]
        quat = base_frame[1][1]
        base_frame_inv_pose = np.linalg.inv(common_utils.transform_mtx(trans, quat))
        stack_pcds = []
        for idx, frame in enumerate(self.frame_list):
            cur_pointcloud = frame[2].copy()
            cur_pointcloud = np.concatenate((cur_pointcloud, np.ones((cur_pointcloud.shape[0], 1), dtype=np.float32) * idx),
                                            axis=-1)
            # transform point cloud and annotation to base_frame coordinate
            delta_pose = np.dot(base_frame_inv_pose, common_utils.transform_mtx(frame[1][0], frame[1][1]))
            cur_pointcloud[:, 0:3] = (np.matmul(delta_pose[0:3, 0:3], cur_pointcloud[:, 0:3].T) + delta_pose[0:3, 3:]).T
            stack_pcds.append(cur_pointcloud)
        point_cloud = np.vstack(stack_pcds)

        try:
            (topic, msg, _) = next(self.data_iter)
            self.frame_list.append(self.read_lidar_topic(msg))
            self.frame_list.pop(0)
        except StopIteration as e:
            self.end_flag = True
            self.bag.close()

        input_dict = {
            'points': point_cloud,
            'frame_id': self.frame_idx,
        }

        self.frame_idx += 1
        if self.model_input:
            input_dict = self.prepare_data(data_dict=input_dict)
        return base_frame[0], base_frame[1], input_dict

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        self.split = 'test'
        if self.root_path.is_dir():
            data_file_list = glob.glob(str(root_path / f'*{self.ext}'))
        elif str(self.root_path).endswith(self.ext):
            data_file_list = [self.root_path]
        elif str(self.root_path).endswith('pkl'):
            with open(self.root_path, 'rb') as f:
                self.val_data_list = pickle.load(f)
                data_file_list = [self.root_path.parent / (info['point_cloud']['lidar_idx']) for info in self.val_data_list]
            self.split = 'val'

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
