import numpy as np
import torch


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

class AnchorFreeBBoxCoder(object):
    """Anchor free bbox coder for 3D boxes.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        with_rot (bool): Whether the bbox is with rotation.
    """

    def __init__(self, num_dir_bins, num_sizes=0, mean_sizes=[], with_rot=True):
        super().__init__()
        assert len(mean_sizes) == num_sizes
        self.num_dir_bins = num_dir_bins
        self.with_rot = with_rot
        self.num_sizes = num_sizes
        self.mean_sizes = mean_sizes

    def encode(self, gt_bboxes_3d, gt_labels_3d):
        """Encode ground truth to prediction targets.

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes \
                with shape (n, 7).
            gt_labels_3d (torch.Tensor): Ground truth classes.

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d[:, 0:3]

        # generate bbox size target
        size_res_target = gt_bboxes_3d[:, 3:6] / 2

        # generate dir target
        box_num = gt_labels_3d.shape[0]
        if self.with_rot:
            (dir_class_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d[:, 6])
            dir_res_target /= (2 * np.pi / self.num_dir_bins)
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(box_num)

        return (center_target, size_res_target, dir_class_target,
                dir_res_target)

    def decode(self, bbox_out):
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - center: predicted bottom center of bboxes.
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size: predicted bbox size.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (batch, n, 7).
        """
        center = bbox_out['center']
        batch_size, num_proposal = center.shape[:2]

        # decode heading angle
        if self.with_rot:
            dir_class = torch.argmax(bbox_out['dir_class'], -1)
            dir_res = torch.gather(bbox_out['dir_res'], 2,
                                   dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(
                batch_size, num_proposal, 1)
        else:
            dir_angle = center.new_zeros(batch_size, num_proposal, 1)

        # decode bbox size
        bbox_size = torch.clamp(bbox_out['size'] * 2, min=0.1)

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

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

    def decode_corners(self, center, size_res, size_class):
        """Decode center, size residuals and class to corners. Only useful for
        axis-aligned bounding boxes, so angle isn't considered.

        Args:
            center (torch.Tensor): Shape [B, N, 3]
            size_res (torch.Tensor): Shape [B, N, 3] or [B, N, C, 3]
            size_class (torch.Tensor): Shape: [B, N] or [B, N, 1]
            or [B, N, C, 3]

        Returns:
            torch.Tensor: Corners with shape [B, N, 6]
        """
        if len(size_class.shape) == 2 or size_class.shape[-1] == 1:
            batch_size, proposal_num = size_class.shape[:2]
            one_hot_size_class = size_res.new_zeros(
                (batch_size, proposal_num, self.num_sizes))
            if len(size_class.shape) == 2:
                size_class = size_class.unsqueeze(-1)
            one_hot_size_class.scatter_(2, size_class, 1)
            one_hot_size_class_expand = one_hot_size_class.unsqueeze(
                -1).repeat(1, 1, 1, 3).contiguous()
        else:
            one_hot_size_class_expand = size_class

        if len(size_res.shape) == 4:
            size_res = torch.sum(size_res * one_hot_size_class_expand, 2)

        mean_sizes = size_res.new_tensor(self.mean_sizes)
        mean_sizes = torch.sum(mean_sizes * one_hot_size_class_expand, 2)
        size_full = (size_res + 1) * mean_sizes
        size_full = torch.clamp(size_full, 0)
        half_size_full = size_full / 2
        corner1 = center - half_size_full
        corner2 = center + half_size_full
        corners = torch.cat([corner1, corner2], dim=-1)
        return corners

    def angle2class(self, angle):
        """Convert continuous angle to a discrete class and a residual.

        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.

        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).

        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        angle_cls = shifted_angle // angle_per_class
        angle_res = shifted_angle - (
            angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """Inverse function to angle2class.

        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].

        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle
