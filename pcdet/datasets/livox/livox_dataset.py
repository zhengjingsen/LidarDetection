import copy
import pickle

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class LivoxDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.livox_infos = []
        self.include_livox_data(self.mode)

    def include_livox_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Livox dataset ...')
        livox_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                livox_infos.extend(infos)
        self.livox_infos.extend(livox_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Livox dataset: %d' % (len(livox_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'pointcloud' / ('%s.bin' % idx)
        assert lidar_file.exists()
        with open(lidar_file, 'rb') as f:
            lidar_data = np.fromfile(lidar_file).reshape(-1, 4)

        return lidar_data

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.pkl' % idx)
        try:
            assert label_file.exists()
        except AssertionError:
            print('[ERROR] get label failed:', label_file)
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)

        return labels

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': np.array([1920, 1080])}
            info['image'] = image_info

            calib_info = {'P2': np.eye(4), 'R0_rect': np.eye(4), 'Tr_velo_to_cam': np.eye(4)}
            info['calib'] = calib_info

            if has_label:
                obj_labels = self.get_label(sample_idx)

                annotations = {}
                # Fuse some categories
                anno_names = []
                for label in obj_labels:
                    if label['name'] in ['car', 'police_car']:
                        anno_names.append('Car')
                    elif label['name'] in ['bus', 'truck', 'Engineering_vehicles', 'trailer']:
                        anno_names.append('Truck')
                    else:
                        anno_names.append(label['name'])
                annotations['name'] = np.array(anno_names)

                annotations['truncated'] = np.array([0 for label in obj_labels])
                annotations['occluded'] = np.array([0 for label in obj_labels])
                annotations['alpha'] = np.array([0 for label in obj_labels])
                annotations['bbox'] = np.array([[1, 1, 1, 1] for label in obj_labels])
                annotations['dimensions'] = np.array([label['box3d_lidar'][3:6] for label in obj_labels],
                                                     dtype=np.float)  # lwh(lidar) format
                annotations['location'] = np.array([label['box3d_lidar'][0:3] for label in obj_labels], dtype=np.float)
                annotations['rotation_y'] = np.array([label['box3d_lidar'][6] for label in obj_labels], dtype=np.float)
                annotations['score'] = np.array([1 for label in obj_labels])
                annotations['difficulty'] = np.array([0 for label in obj_labels], np.int32)

                num_objects = len([label['name'] for label in obj_labels if label['name'] != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc_lidar = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                info['annos'] = annotations

                if count_inside_pts:
                    annotations['num_points_in_gt'] = np.array([label['num_points_in_gt'] for label in obj_labels])

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('livox_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes)
            pred_boxes_img = np.ones((pred_boxes_camera.shape[0], 4))

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.livox_infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.livox_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.livox_infos) * self.total_epochs

        return len(self.livox_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.livox_infos)

        info = copy.deepcopy(self.livox_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        calib = None

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pass

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape

        return data_dict


def create_livox_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = LivoxDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('livox_infos_%s.pkl' % train_split)
    val_filename = save_path / ('livox_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'livox_infos_trainval.pkl'
    test_filename = save_path / 'livox_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    dataset.set_split(train_split)
    livox_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(train_filename, 'wb') as f:
        pickle.dump(livox_infos_train, f)
    print('Livox info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    livox_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=False)
    with open(val_filename, 'wb') as f:
        pickle.dump(livox_infos_val, f)
    print('Livox info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(livox_infos_train + livox_infos_val, f)
    print('Livox info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    livox_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(livox_infos_test, f)
    print('Livox info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict

    dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
    create_livox_infos(
        dataset_cfg=dataset_cfg,
        class_names=['Car', 'Truck'],
        data_path=Path('/home/tong.wang/datasets/Livox/'),
        save_path=Path('/home/tong.wang/datasets/Livox/')
    )
