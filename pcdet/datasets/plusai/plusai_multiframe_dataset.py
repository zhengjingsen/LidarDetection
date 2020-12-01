import copy
import pickle

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class PlusAIMultiframeDataset(DatasetTemplate):
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
        print('root_path: ', self.root_path.resolve())
        self.root_split_path = self.root_path

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        # TODO: should read from dataset_cfg
        self.stack_frame_size = 3
        self.base_frame_idx = 1

        self.plusai_infos = []
        self.include_plusai_data(self.mode)

    def include_plusai_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading PlusAI dataset ...')
        plusai_infos = []
        
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                plusai_infos.extend(infos)
        self.plusai_infos.extend(plusai_infos)

        if self.logger is not None:
            self.logger.info('Total samples for PlusAI dataset: %d' % (len(plusai_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / idx
        assert lidar_file.exists()
        lidar_data = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

        return lidar_data

    def get_label(self, idx):
        [scene_name, _, frame] = idx.split('/')
        label_file = self.root_split_path / scene_name / 'label' / (frame[:-4] + '.pkl')
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
            # print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 5, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': np.array([1920, 1080])}
            info['image'] = image_info

            calib_info = {'P2': np.eye(4), 'R0_rect': np.eye(4), 'Tr_velo_to_cam': np.eye(4)}
            info['calib'] = calib_info

            if has_label:
                obj_labels = self.get_label(sample_idx)
                obj_labels = obj_labels['obstacles']
                # if len(obj_labels) == 0:
                #     print('Error occured, obstacle list is empty for sample_{}!'.format(sample_idx))
                #     info['annos'] = {}
                #     return info

                annotations = {}
                annotations['name'] = np.array([label[self.base_frame_idx]['class'] for label in obj_labels])
                annotations['truncated'] = np.array([0 for label in obj_labels])
                annotations['occluded'] = np.array([0 for label in obj_labels])
                annotations['alpha'] = np.array([0 for label in obj_labels])
                annotations['bbox'] = np.array([[1, 1, 1, 1] for label in obj_labels])
                annotations['dimensions'] = np.array([label[self.base_frame_idx]['size'] for label in obj_labels])  # lwh(lidar) format
                annotations['location'] = np.array([label[self.base_frame_idx]['location'] for label in obj_labels])
                annotations['rotation_y'] = np.array([label[self.base_frame_idx]['heading'] for label in obj_labels])
                annotations['score'] = np.array([1 for label in obj_labels])
                annotations['difficulty'] = np.array([0 for label in obj_labels], np.int32)

                # multi-frame data
                annotations['locations'] = np.array([[label['location'] for label in obj] for obj in obj_labels])
                annotations['rotations_y'] = np.array([[label['heading'] for label in obj] for obj in obj_labels])
                annotations['velocities'] = np.array([[label['velocity'] for label in obj] for obj in obj_labels])

                # num_objects = len([label['name'] for label in obj_labels if label['name'] != 'DontCare'])
                num_objects = len([name for name in annotations['name'] if name != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                if len(obj_labels) > 0:
                    loc = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                    loc_lidar = loc
                    l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                else:
                    print('obstacle size is zero in {}'.format(sample_idx))
                    gt_boxes_lidar = np.array([])
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                info['annos'] = annotations

                if count_inside_pts:
                    # annotations['num_points_in_gt'] = np.array([label['num_points_in_gt'] for label in obj_labels])
                    annotations['num_points_in_gt'] = np.array([20 for label in obj_labels])

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('plusai_dbinfos_%s.pkl' % split)

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
            if not num_obj > 0:
                continue

            cur_gt_boxes = gt_boxes.copy()
            point_indices = np.zeros([num_obj, points.shape[0]], dtype=np.int32)
            for i in range(self.stack_frame_size):
                cur_gt_boxes[:, 0:3] = annos['locations'].reshape(num_obj, -1)[:, 3*i:3*(i+1)]
                cur_gt_boxes[:, -1] = annos['rotations_y'].reshape(num_obj, -1)[:, i]
                point_indices += roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(cur_gt_boxes)).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                [scene_name, _, frame] = sample_idx.split('/')
                filename = '%s_%s_%s_%d.bin' % (scene_name, frame[:-4], names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                num_points_in_gt = gt_points.shape[0]
                if num_points_in_gt <= 0:
                    print('num points in {} is zero, skip this target!'.format(filename[0:-4]))
                    continue

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': num_points_in_gt,
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i],
                               'locations': annos['locations'][i], 'rotations_y': annos['rotations_y'][i]}
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
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
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
        if 'annos' not in self.plusai_infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.plusai_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.plusai_infos) * self.total_epochs

        return len(self.plusai_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.plusai_infos)

        info = copy.deepcopy(self.plusai_infos[index])
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
                'gt_boxes': gt_boxes_lidar,
                'locations': annos['locations'],
                'rotations_y': annos['rotations_y']
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape

        return data_dict


def create_plusai_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = PlusAIMultiframeDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('plusai_infos_%s.pkl' % train_split)
    val_filename = save_path / ('plusai_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'plusai_infos_trainval.pkl'
    test_filename = save_path / 'plusai_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    plusai_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(plusai_infos_train, f)
    print('PlusAI info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    plusai_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(plusai_infos_val, f)
    print('PlusAI info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(plusai_infos_train + plusai_infos_val, f)
    print('PlusAI info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # plusai_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(plusai_infos_test, f)
    # print('PlusAI info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

def check_lidar_data(pointcloud_path):
    import os
    import mayavi.mlab

    pointcloud_list = os.listdir(pointcloud_path)
    for pc in pointcloud_list:
        print(pc)
        pc_file = pointcloud_path / pc
        assert pc_file.exists()
        lidar_data = np.fromfile(str(pc_file), dtype=np.float32).reshape(-1, 5)

        fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1080, 1080))
        mayavi.mlab.points3d(lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2],
                             lidar_data[:, 4],  # Values used for Color
                             mode="point",
                             colormap='jet',  # 'bone', 'copper', 'gnuplot', 'spectral'
                             # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                             figure=fig,
                             )
        mayavi.mlab.show()

def visualize_mot_data(data_path):
    import os
    from pcdet.utils.data_viz import plot_gt_boxes
    bev_range = np.array([0, -20, -2, 150, 20, 5])
    data_path = os.path.join(data_path, 'multiframe', 'training')
    for frame in os.listdir(os.path.join(data_path, 'pointcloud')):
        pcd_file = os.path.join(data_path, 'pointcloud', frame)
        label_file = os.path.join(data_path, 'label', frame[0:-4]+'.pkl')
        point_cloud = np.fromfile(pcd_file, dtype=np.float32).reshape([-1, 5])
        with open(label_file, 'rb') as f:
            label = pickle.load(f)

        gt_boxes = np.array([np.concatenate([obj[0]['location'], obj[0]['size'], [obj[0]['heading']]]) for obj in label['obstacles']])
        plot_gt_boxes(point_cloud, gt_boxes, bev_range, name=frame[0:-4])

if __name__ == '__main__':
    import sys
    import yaml
    from pathlib import Path
    from easydict import EasyDict
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_plusai_infos':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_plusai_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Truck'],
            data_path=ROOT_DIR / 'data' / 'plusai' / 'multiframe',
            save_path=ROOT_DIR / 'data' / 'plusai' / 'multiframe'
        )
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'check_lidar_data':
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        check_lidar_data(ROOT_DIR / 'data' / 'plusai' / 'multiframe' / 'gt_database')
        # visualize_mot_data('/media/jingsen/data/Dataset/plusai/')