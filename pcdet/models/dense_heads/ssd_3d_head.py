import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from functools import partial
from .base_conv_bbox_head import BaseConvBboxHead
from ..model_utils.vote_module import VoteModule
from ...utils.utils import multi_apply
from ...utils import box_coder_utils, loss_utils, box_utils, common_utils
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...ops.pointnet2.pointnet2_batch.pointnet2_utils import furthest_point_sample

class SSD3DHead(nn.Module):
    r"""Bbox head of `3DSSD <https://arxiv.org/abs/2002.10187>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        in_channels (int): The number of input feature channel.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        act_cfg (dict): Config of activation in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
        vote_loss (dict): Config of candidate points regression loss.
    """
    def __init__(self,
                 model_cfg,
                 in_channels=256,
                 predict_boxes_when_training=True,
                 **kwargs):
        super(SSD3DHead, self).__init__()
        self.num_classes = kwargs['num_class']
        self.train_cfg = model_cfg.TRAIN_CFG
        self.test_cfg = model_cfg.TEST_CFG
        self.gt_per_seed = model_cfg.VOTE_MODULE_CFG.gt_per_seed
        self.num_proposal = model_cfg.VOTE_AGGREGATION_CFG.num_point

        self.objectness_loss =  loss_utils.CrossEntropyLoss(use_sigmoid=True, reduction='sum', loss_weight=1.0)
        self.dir_class_loss = loss_utils.CrossEntropyLoss(reduction='sum', loss_weight=1.0)
        self.center_loss = loss_utils.WeightedSmoothL1Loss(beta=1.0)
        self.dir_res_loss = loss_utils.WeightedSmoothL1Loss(beta=1.0)
        self.size_res_loss = loss_utils.WeightedSmoothL1Loss(beta=1.0)

        self.bbox_coder = getattr(box_coder_utils, model_cfg.BBOX_CODER_CFG.type)(
            model_cfg.BBOX_CODER_CFG.num_dir_bins)
        self.num_dir_bins = self.bbox_coder.num_dir_bins

        self.vote_module = VoteModule(**model_cfg.VOTE_MODULE_CFG)
        self.vote_aggregation = pointnet2_modules.PointnetSAModuleMSG(
            npoint=model_cfg.VOTE_AGGREGATION_CFG.num_point,
            radii=model_cfg.VOTE_AGGREGATION_CFG.radii,
            nsamples=model_cfg.VOTE_AGGREGATION_CFG.sample_nums,
            mlps=model_cfg.VOTE_AGGREGATION_CFG.mlp_channels,
            use_xyz=model_cfg.VOTE_AGGREGATION_CFG.use_xyz)

        # Bbox classification and regression
        self.conv_pred = BaseConvBboxHead(
            **model_cfg.PRED_LAYER_CFG,
            num_cls_out_channels=self._get_cls_out_channels(),
            num_reg_out_channels=self._get_reg_out_channels())

        self.corner_loss = loss_utils.WeightedSmoothL1Loss()
        self.vote_loss = loss_utils.WeightedSmoothL1Loss()
        self.num_candidates = model_cfg.VOTE_MODULE_CFG['num_points']
        self.sample_mod = 'spec'
        self.predict_boxes_when_training = predict_boxes_when_training

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        # Bbox classification and regression
        # (center residual (3), size regression (3)
        # heading class+residual (num_dir_bins*2)),
        return 3 + 3 + self.num_dir_bins * 2

    def _extract_input(self, batch_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        seed_points = batch_dict['sa_xyz'][-1]
        seed_features = batch_dict['sa_features'][-1]
        seed_indices = batch_dict['sa_indices'][-1]

        return seed_points, seed_features, seed_indices

    def init_weights(self):
        """Initialize weights of VoteHead."""
        pass

    def split_pred(self, cls_preds, reg_preds, base_xyz):
        """Split predicted features to specific parts.

        Args:
            cls_preds (torch.Tensor): Class predicted features to split.
            reg_preds (torch.Tensor): Regression predicted features to split.
            base_xyz (torch.Tensor): Coordinates of points.

        Returns:
            dict[str, torch.Tensor]: Split results.
        """
        results = {}
        results['obj_scores'] = cls_preds

        start, end = 0, 0
        reg_preds_trans = reg_preds.transpose(2, 1)

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['center_offset'] = reg_preds_trans[..., start:end]
        results['center'] = base_xyz.detach() + reg_preds_trans[..., start:end]
        start = end

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['size'] = reg_preds_trans[..., start:end]
        start = end

        # decode direction
        end += self.num_dir_bins
        results['dir_class'] = reg_preds_trans[..., start:end]
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end]
        start = end

        results['dir_res_norm'] = dir_res_norm
        results['dir_res'] = dir_res_norm * (2 * np.pi / self.num_dir_bins)

        return results

    def forward(self, batch_dict):
        """Forward pass.

        Note:
            The forward of VoteHead is devided into 4 steps:

                1. Generate vote_points from seed_points.
                2. Aggregate vote_points.
                3. Predict bbox and score.
                4. Decode predictions.

        Args:
            feat_dict (dict): Feature dict from backbone.
            sample_mod (str): Sample mode for vote aggregation layer.
                valid modes are "vote", "seed", "random" and "spec".

        Returns:
            dict: Predictions of vote head.
        """
        sample_mod = self.sample_mod
        assert sample_mod in ['vote', 'seed', 'random', 'spec']

        seed_points, seed_features, seed_indices = self._extract_input(
            batch_dict)

        # 1. generate vote_points from seed_points
        vote_points, vote_features, vote_offset = self.vote_module(
            seed_points, seed_features)
        results = dict(
            seed_points=seed_points,
            seed_indices=seed_indices,
            vote_points=vote_points,
            vote_features=vote_features,
            vote_offset=vote_offset)

        # 2. aggregate vote_points
        if sample_mod == 'spec':
            # Specify the new center in vote_aggregation
            aggregation_inputs = dict(
                xyz=seed_points,
                features=seed_features,
                new_xyz=vote_points)
        else:
            raise NotImplementedError(
                f'Sample mode {sample_mod} is not supported!')

        vote_aggregation_ret = self.vote_aggregation(**aggregation_inputs)
        aggregated_points, features, aggregated_indices = vote_aggregation_ret

        results['aggregated_points'] = aggregated_points
        results['aggregated_features'] = features
        results['aggregated_indices'] = aggregated_indices

        # 3. predict bbox and score
        cls_predictions, reg_predictions = self.conv_pred(features)

        # 4. decode predictions
        decode_res = self.split_pred(cls_predictions,
                                     reg_predictions,
                                     aggregated_points)

        results.update(decode_res)

        batch_size = batch_dict['batch_size']
        num_points_feature = batch_dict['points'].shape[-1]
        points = batch_dict['points'].view(batch_size, -1, num_points_feature)[..., 1:4]
        points = [points[i] for i in range(batch_size)]
        self.forward_ret_dict = {
            'bbox_preds': results,
            'points': points}

        if self.training:
            gt_bboxes_3d = [batch_dict['gt_boxes'][..., :-1][i] for i in range(batch_size)]
            gt_labels_3d = [batch_dict['gt_boxes'][..., -1][i].int() for i in range(batch_size)]
            self.forward_ret_dict.update({
                'gt_bboxes_3d': gt_bboxes_3d,
                'gt_labels_3d': gt_labels_3d})

        if not self.training or self.predict_boxes_when_training:
            batch_dict = self.get_bboxes(batch_dict)

        return batch_dict

    def loss(self):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of SSD3DHead.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of 3DSSD.
        """
        bbox_preds = self.forward_ret_dict['bbox_preds']
        points = self.forward_ret_dict['points']
        gt_bboxes_3d = self.forward_ret_dict['gt_bboxes_3d']
        gt_labels_3d = self.forward_ret_dict['gt_labels_3d']

        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   bbox_preds)
        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, positive_mask, negative_mask, centerness_weights,
         box_loss_weights, heading_res_loss_weight) = targets

        # calculate centerness loss
        centerness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            centerness_targets,
            weight=centerness_weights)

        # calculate center loss
        center_loss = self.center_loss(
            bbox_preds['center_offset'],
            center_targets,
            weights=box_loss_weights).sum()

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(1, 2),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        dir_res_loss = self.dir_res_loss(
            bbox_preds['dir_res_norm'],
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins),
            weights=heading_res_loss_weight).sum()

        # calculate size residual loss
        size_loss = self.size_res_loss(
            bbox_preds['size'],
            size_res_targets,
            weights=box_loss_weights).sum()

        # calculate corner loss
        one_hot_dir_class_targets = dir_class_targets.new_zeros(
            bbox_preds['dir_class'].shape)
        one_hot_dir_class_targets.scatter_(2, dir_class_targets.unsqueeze(-1),
                                           1)
        pred_bbox3d = self.bbox_coder.decode(
            dict(
                center=bbox_preds['center'],
                dir_res=bbox_preds['dir_res'],
                dir_class=one_hot_dir_class_targets,
                size=bbox_preds['size']))
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_corners3d = box_utils.boxes_to_corners_3d(pred_bbox3d)
        corner_loss = self.corner_loss(
            pred_corners3d,
            corner3d_targets.reshape(-1, 8, 3),
            weights=box_loss_weights.view(-1, 1, 1)).sum()

        # calculate vote loss
        vote_loss = self.vote_loss(
            bbox_preds['vote_offset'].transpose(1, 2),
            vote_targets,
            weights=vote_mask).sum()

        losses = dict(
            centerness_loss=centerness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_res_loss=size_loss,
            corner_loss=corner_loss,
            vote_loss=vote_loss)

        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    bbox_preds=None):
        """Generate targets of ssd3d head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of ssd3d head.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        # find empty example
        new_gt_bboxes_3d = []
        new_gt_labels_3d = []
        for index in range(len(gt_labels_3d)):
            valid_gt = gt_labels_3d[index] != 0
            cur_gt_bboxes_3d = gt_bboxes_3d[index][valid_gt]
            cur_gt_labels_3d = gt_labels_3d[index][valid_gt]
            if len(cur_gt_labels_3d) == 0:
                print('len(gt_labels_3d[index]) == 0')
                fake_box = cur_gt_bboxes_3d.new_zeros(
                    1, gt_bboxes_3d[index].shape[-1])
                cur_gt_bboxes_3d = fake_box
                cur_gt_labels_3d = cur_gt_labels_3d.new_zeros(1)
            new_gt_bboxes_3d.append(cur_gt_bboxes_3d)
            new_gt_labels_3d.append(cur_gt_labels_3d)

        gt_bboxes_3d = new_gt_bboxes_3d
        gt_labels_3d = new_gt_labels_3d

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        seed_points = [
            bbox_preds['seed_points'][i, :self.num_candidates].detach()
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, center_targets, size_res_targets, dir_class_targets,
         dir_res_targets, mask_targets, centerness_targets, corner3d_targets,
         vote_mask, positive_mask, negative_mask) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             aggregated_points, seed_points)

        center_targets = torch.stack(center_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)
        centerness_targets = torch.stack(centerness_targets).detach()
        corner3d_targets = torch.stack(corner3d_targets)
        vote_targets = torch.stack(vote_targets)
        vote_mask = torch.stack(vote_mask)

        center_targets -= bbox_preds['aggregated_points']

        centerness_weights = (positive_mask +
                              negative_mask).unsqueeze(-1).repeat(
                                  1, 1, self.num_classes).float()
        centerness_weights = centerness_weights / \
            (centerness_weights.sum() + 1e-6)
        vote_mask = vote_mask / (vote_mask.sum() + 1e-6)

        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)

        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        heading_res_loss_weight = heading_label_one_hot * \
            box_loss_weights.unsqueeze(-1)

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask, positive_mask,
                negative_mask, centerness_weights, box_loss_weights,
                heading_res_loss_weight)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           aggregated_points=None,
                           seed_points=None):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                candidate points layer.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        gt_corner3d = box_utils.boxes_to_corners_3d(gt_bboxes_3d)

        (center_targets, size_targets, dir_class_targets,
         dir_res_targets) = self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        points_mask, assignment = self._assign_targets_by_points_inside(
            gt_bboxes_3d, aggregated_points)

        vote_targets = center_targets.clone()
        center_targets = center_targets[assignment]
        size_res_targets = size_targets[assignment]
        mask_targets = torch.clamp((gt_labels_3d[assignment] - 1).long(), min=0)
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        corner3d_targets = gt_corner3d[assignment]

        top_center_targets = center_targets.clone()
        top_center_targets[:, 2] += size_res_targets[:, 2]
        dist = torch.norm(aggregated_points - top_center_targets, dim=1)
        dist_mask = dist < self.train_cfg.pos_distance_thr
        positive_mask = (points_mask.max(1)[0] > 0) * dist_mask
        negative_mask = (points_mask.max(1)[0] == 0)

        # Centerness loss targets
        canonical_xyz = aggregated_points - center_targets
        if self.bbox_coder.with_rot:
            canonical_xyz = common_utils.rotate_points_along_z(
                canonical_xyz.unsqueeze(1),
                -gt_bboxes_3d[:, 6][assignment]).squeeze(1)
        distance_front = torch.clamp(
            size_res_targets[:, 0] - canonical_xyz[:, 0], min=0)
        distance_back = torch.clamp(
            size_res_targets[:, 0] + canonical_xyz[:, 0], min=0)
        distance_left = torch.clamp(
            size_res_targets[:, 1] - canonical_xyz[:, 1], min=0)
        distance_right = torch.clamp(
            size_res_targets[:, 1] + canonical_xyz[:, 1], min=0)
        distance_top = torch.clamp(
            size_res_targets[:, 2] - canonical_xyz[:, 2], min=0)
        distance_bottom = torch.clamp(
            size_res_targets[:, 2] + canonical_xyz[:, 2], min=0)

        centerness_l = torch.min(distance_front, distance_back) / torch.max(
            distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(
            distance_left, distance_right)
        centerness_h = torch.min(distance_bottom, distance_top) / torch.max(
            distance_bottom, distance_top)
        centerness_targets = torch.clamp(
            centerness_l * centerness_w * centerness_h, min=0)
        centerness_targets = centerness_targets.pow(1 / 3.0)
        centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

        proposal_num = centerness_targets.shape[0]
        one_hot_centerness_targets = centerness_targets.new_zeros(
            (proposal_num, self.num_classes))
        one_hot_centerness_targets.scatter_(1, mask_targets.unsqueeze(-1), 1)
        centerness_targets = centerness_targets.unsqueeze(1) * one_hot_centerness_targets

        # Vote loss targets
        enlarged_gt_bboxes_3d = box_utils.enlarge_box3d(gt_bboxes_3d,
            [self.train_cfg.expand_dims_length * 2] * 3)
        # enlarged_gt_bboxes_3d.tensor[:, 2] -= self.train_cfg.expand_dims_length
        vote_mask, vote_assignment = self._assign_targets_by_points_inside(
            enlarged_gt_bboxes_3d, seed_points)

        vote_targets = vote_targets[vote_assignment] - seed_points
        vote_mask = vote_mask.max(1)[0] > 0

        return (vote_targets, center_targets, size_res_targets,
                dir_class_targets, dir_res_targets, mask_targets,
                centerness_targets, corner3d_targets, vote_mask, positive_mask,
                negative_mask)

    def get_bboxes(self, batch_dict):
        """Generate bboxes from sdd3d head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from sdd3d head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        bbox_preds = self.forward_ret_dict['bbox_preds']
        bbox3d = self.bbox_coder.decode(bbox_preds)
        obj_scores = bbox_preds['obj_scores'].transpose(1, 2)

        batch_dict.update({
            'batch_cls_preds': obj_scores,
            'batch_box_preds': bbox3d,
            'cls_preds_normalized': False
        })

        return batch_dict

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_boxes = bboxes_3d.shape[0]
        assignment = roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(dim=0),
                                                               bboxes_3d.unsqueeze(dim=0)).squeeze().long()
        assignment[assignment == -1] = num_boxes
        points_mask = assignment.new_zeros([points.shape[0], num_boxes + 1])
        points_mask.scatter_(dim=1, index=assignment.unsqueeze(dim=1), value=1)
        points_mask = points_mask[:, :-1]
        assignment[assignment == num_boxes] = num_boxes - 1

        return points_mask, assignment
