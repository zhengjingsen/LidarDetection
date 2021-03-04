from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        # return 'train' if self.training else 'leading_test'     # Modified for leading test
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes
            if data_dict.get('locations', None) is not None:
                data_dict['locations'] = data_dict['locations'][selected]
            if data_dict.get('rotations_y', None) is not None:
                data_dict['rotations_y'] = data_dict['rotations_y'][selected]

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        example_merged = defaultdict(list)
        for example in batch_list:
            for k, v in example.items():
                example_merged[k].append(v)
        ret = {}
        for key, elems in example_merged.items():
            if key in ['voxels', 'num_points', 'voxel_centers', 'seg_labels', 'part_labels', 'bbox_reg_labels']:
                ret[key] = np.concatenate(elems, axis=0)
            elif key in ['coordinates', 'points']:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            elif key in ['gt_boxes']:
                max_gt = 0
                batch_size = elems.__len__()
                for k in range(batch_size):
                    max_gt = max(max_gt, elems[k].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, elems[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :elems[k].__len__(), :] = elems[k]
                ret[key] = batch_gt_boxes3d
            elif key in ['bev_local_coordinate', 'fv_local_coordinate', 'intensity', 'bev_coordinate']:
                ret[key] = torch.cat(elems, dim=0)
            elif key in ['bev_mapping_vf', 'fv_mapping_vf']:
                for idx, elem in enumerate(elems):
                    cur_elem = torch.cat((idx * torch.ones((elem.shape[0], 1), dtype=torch.int32), elem), dim=1)
                    if idx == 0:
                        ret[key] = cur_elem
                    else:
                        ret[key] = torch.cat((ret[key], cur_elem), dim=0)

                ret[key] = ret[key].long()
            elif key in ['bev_mapping_pv', 'fv_mapping_pv']:
                cur_voxel_num = 0
                for idx, elem in enumerate(elems):
                    if idx == 0:
                        ret[key] = elem
                    else:
                        ret[key] = torch.cat((ret[key], elem + cur_voxel_num), dim=0)
                    if key == 'bev_mapping_pv':
                        cur_voxel_num += example_merged['bev_mapping_vf'][idx].shape[0]
                    else:
                        cur_voxel_num += example_merged['fv_mapping_vf'][idx].shape[0]
                ret[key] = ret[key].long()
            else:
                ret[key] = np.stack(elems, axis=0)
        ret['batch_size'] = batch_list.__len__()
        return ret
