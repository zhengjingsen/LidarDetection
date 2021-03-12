from functools import partial

import torch
import numpy as np
from ...utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            if data_dict.get('locations', None) is not None:
                data_dict['locations'] = data_dict['locations'][mask]
            if data_dict.get('rotations_y', None) is not None:
                data_dict['rotations_y'] = data_dict['rotations_y'][mask]

        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def transform_points_to_dynamic_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            from ...ops.PointCloudVoxel.pointcloud_voxel import PointCloudVoxel
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int32)
            self.voxel_size = config.VOXEL_SIZE
            self.max_points_num = config.MAX_POINT_NUMS
            self.max_num_of_voxels = config.MAX_NUMBER_OF_VOXELS
            voxel_generator = PointCloudVoxel(0,
                                              self.grid_size[0],
                                              self.grid_size[1],
                                              self.grid_size[2],
                                              self.point_cloud_range[0], self.point_cloud_range[3],
                                              self.point_cloud_range[1], self.point_cloud_range[4],
                                              self.point_cloud_range[2], self.point_cloud_range[5],
                                              0, 1, 1, 0, 0, 0, 1)

            return partial(self.transform_points_to_dynamic_voxels, voxel_generator=voxel_generator)

        # do my generate
        points = data_dict['points']
        points = torch.from_numpy(points)

        bev_coordinate = torch.zeros((self.max_points_num, 3), dtype=torch.float32)
        bev_local_coordinate = torch.zeros((self.max_points_num, 3), dtype=torch.float32)
        intensity = torch.zeros((self.max_points_num, 2), dtype=torch.float32)
        bev_mapping_pv = torch.zeros((self.max_points_num), dtype=torch.int32)
        bev_mapping_vf = torch.zeros((self.max_num_of_voxels, 3), dtype=torch.int32)

        voxel_generator.dynamicVoxelBEV(points, bev_coordinate, bev_local_coordinate, intensity,
                                        bev_mapping_pv, bev_mapping_vf)
        valid_point_nums = voxel_generator.getValidPointNums()
        valid_bev_voxel_nums = voxel_generator.getValidBEVVoxelNums()

        bev_coordinate = bev_coordinate[:valid_point_nums]
        bev_local_coordinate = bev_local_coordinate[:valid_point_nums]
        intensity = intensity[:valid_point_nums]
        bev_mapping_pv = bev_mapping_pv[:valid_point_nums]
        bev_mapping_vf = bev_mapping_vf[:valid_bev_voxel_nums]
        data_dict.update({
            'bev_coordinate': bev_coordinate,
            'bev_local_coordinate': bev_local_coordinate,
            'intensity': intensity,
            'bev_mapping_pv': bev_mapping_pv,
            'bev_mapping_vf': bev_mapping_vf,
        })
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            # pts_azimuth = np.abs(points[:, 1] / points[:, 0])
            # pts_near_flag = (pts_depth < 60.0) & (pts_azimuth < 0.58)
            pts_near_flag = pts_depth < config.SAMPLE_DISTANCE_THRESH
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            # print('num_points: ', num_points, ', total_num_points: ', len(points), ', num_far_points: ', len(far_idxs_choice))
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
