import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            if self.model_cfg.get('USE_MULTIFRAME_ENLARGED_GT_BOXES', False):
                from pcdet.utils.box_utils import boxes_to_corners_3d
                from pcdet.utils.common_utils import rotate_points_along_z

                batch_size, num_boxes, boxes_dim = data_dict['gt_boxes'].shape
                if num_boxes > 0:
                    stack_frame_size = data_dict['locations'].shape[2]
                    gt_boxes_corner = []
                    locations = data_dict['locations'].view(-1, stack_frame_size, 3)
                    rotations_y = data_dict['rotations_y'].view(-1, stack_frame_size)
                    gt_boxes = data_dict['gt_boxes'].view(-1, boxes_dim)
                    cur_gt_boxes = gt_boxes.clone()
                    for idx in range(stack_frame_size):
                        cur_gt_boxes[:, 0:3] = locations[:, idx, :]
                        cur_gt_boxes[:, -1] = rotations_y[:, idx]
                        gt_boxes_corner.append(boxes_to_corners_3d(cur_gt_boxes))
                    gt_boxes_corner = torch.cat(gt_boxes_corner, dim=1)
                    gt_boxes_corner -= gt_boxes[:, None, 0:3]
                    gt_boxes_corner_local = rotate_points_along_z(gt_boxes_corner, -gt_boxes[:, -2])
                    multi_length = gt_boxes_corner_local[:, :, 0].max(dim=1)[0] - gt_boxes_corner_local[:, :, 0].min(dim=1)[0]
                    multi_width = gt_boxes_corner_local[:, :, 1].max(dim=1)[0] - gt_boxes_corner_local[:, :, 1].min(dim=1)[0]
                    gt_boxes_enlarged = torch.cat(
                        [gt_boxes[:, 0:3], multi_length[:, None], multi_width[:, None], gt_boxes[:, 5:]],
                        dim=-1)
                    gt_boxes = gt_boxes_enlarged.view(batch_size, num_boxes, boxes_dim)
                else:
                    gt_boxes = data_dict['gt_boxes']
                data_dict['gt_boxes_enlarged'] = gt_boxes

            else:
                gt_boxes = data_dict['gt_boxes']
            targets_dict = self.assign_targets(
                gt_boxes=gt_boxes
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
