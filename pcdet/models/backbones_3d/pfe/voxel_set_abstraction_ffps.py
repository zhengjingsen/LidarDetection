'''
@Time       : 
@Author     : Jingsen Zheng
@File       : voxel_set_abstraction_ffps
@Brief      : 
'''
import torch
import torch.nn as nn

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils


class VoxelSetAbstractionFFps(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        for src_name, cfg in SA_cfg.items():
            if src_name == 'bev':
                continue

            mlps = SA_cfg[src_name].MLPS
            if src_name == 'raw_points':
                for k in range(len(mlps)):
                    mlps[k] = [num_rawpoint_features - 3] + mlps[k]
            else:
                self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
                for k in range(len(mlps)):
                    mlps[k] = [mlps[k][0]] + mlps[k]
            cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

        c_in = 0
        for feature_sources in self.model_cfg.FEATURES_SOURCE:
            for src_name in feature_sources:
                if src_name == 'bev':
                    c_in += num_bev_features
                else:
                    mlps = SA_cfg[src_name].MLPS
                    c_in += sum([x[-1] for x in mlps])
                    if src_name == self.model_cfg.SAMPLE_FEATURE_SOURCE:
                        c_in += SA_cfg[src_name].MLPS[0][0] - 3

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.FP_layer = pointnet2_stack_modules.StackPointnetFPModule(mlp=[0])
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = pointnet2_stack_utils.bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_points_for_sample(self, batch_dict):
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points']
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            # src_points = common_utils.get_voxel_centers(
            #     batch_dict['voxel_coords'][:, 1:4],
            #     downsample_times=1,
            #     voxel_size=self.voxel_size,
            #     point_cloud_range=self.point_cloud_range
            # )
            src_points = batch_dict['voxels'][:, 0, 0:3]
            src_points = torch.cat([batch_dict['voxel_coords'][:, 0:1], src_points], dim=1)
        else:
            raise NotImplementedError

        points = src_points[:, 1:4]
        points_feature = src_points[:, 4:]
        batch_cnt = points.new_zeros(batch_dict['batch_size']).int()
        for i in range(batch_dict['batch_size']):
            batch_cnt[i] = (src_points[:, 0] == i).sum()

        return points, points_feature, batch_cnt

    def get_sampled_points(self, points, points_feature, batch_cnt, num_keypoints):
        keypoints_list = []
        keypoints_feature_list = []

        cur_idx = 0
        for cur_cnt in batch_cnt.cpu().numpy():
            sampled_points = points[cur_idx:cur_idx+cur_cnt, :]
            sampled_points_feature = points_feature[cur_idx:cur_idx+cur_cnt, :]
            if sampled_points_feature.shape[1] < 3:
                features_for_fps = sampled_points.unsqueeze(dim=0).contiguous()
            else:
                # features_for_fps = torch.cat([sampled_points, sampled_points_feature], dim=1).unsqueeze(dim=0).contiguous()
                features_for_fps = sampled_points_feature.unsqueeze(dim=0).contiguous()
            num_keypoints_fs = num_keypoints // 3 * 2
            cur_pt_idxs_fs = pointnet2_stack_utils.feature_furthest_point_sample(
                features_for_fps, num_keypoints_fs
            ).long()
            cur_pt_idxs_ds = pointnet2_stack_utils.furthest_point_sample(
                sampled_points.unsqueeze(dim=0).contiguous(), num_keypoints - num_keypoints_fs
            ).long()
            cur_pt_idxs = torch.cat([cur_pt_idxs_fs, cur_pt_idxs_ds], dim=1)

            if sampled_points.shape[0] < num_keypoints:
                empty_num = num_keypoints - sampled_points.shape[0]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[cur_pt_idxs[0]]
            keypoints_list.append(keypoints)

            keypoints_feature = sampled_points_feature[cur_pt_idxs[0]]
            keypoints_feature_list.append(keypoints_feature)

            cur_idx += cur_cnt

        keypoints = torch.cat(keypoints_list, dim=0)  # (M1 + M2 + ..., 3)
        keypoints_feature = torch.cat(keypoints_feature_list, dim=0) # (M1+ M2 + ..., C)
        batch_cnt = keypoints.new_zeros(batch_cnt.shape[0]).int().fill_(num_keypoints)
        return keypoints, keypoints_feature, batch_cnt

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        batch_size = batch_dict['batch_size']
        unknown, _, unknown_batch_cnt = self.get_points_for_sample(batch_dict)

        cur_coords = batch_dict['multi_scale_3d_features']['x_conv1'].indices
        known = common_utils.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=self.downsample_times_map['x_conv1'],
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        known_batch_cnt = known.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            known_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
        known_feats = batch_dict['multi_scale_3d_features']['x_conv1'].features.contiguous()

        point_features = self.FP_layer(unknown, unknown_batch_cnt, known, known_batch_cnt, known_feats=known_feats)
        points = unknown
        batch_cnt = unknown_batch_cnt
        for num_keypoints, feature_sources in zip(self.model_cfg.NUM_KEYPOINTS, self.model_cfg.FEATURES_SOURCE):
            # sample keypoints from last keypoints
            keypoints, keypoints_features, keypoints_batch_cnt = self.get_sampled_points(
                points, point_features, batch_cnt, num_keypoints)

            point_features_list = []
            for _, src_name in enumerate(feature_sources):
                if src_name == 'raw_points':
                    raw_points = batch_dict['points']
                    xyz = raw_points[:, 1:4]
                    xyz_batch_cnt = xyz.new_zeros(batch_size).int()
                    for bs_idx in range(batch_size):
                        xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
                    features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None
                elif src_name != 'bev':
                    cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
                    xyz = common_utils.get_voxel_centers(
                        cur_coords[:, 1:4],
                        downsample_times=self.downsample_times_map[src_name],
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                    xyz_batch_cnt = xyz.new_zeros(batch_size).int()
                    for bs_idx in range(batch_size):
                        xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
                    features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()
                else:
                    continue

                k = self.SA_layer_names.index(src_name)
                pooled_points, pooled_features = self.SA_layers[k](
                    xyz=xyz.contiguous(),
                    xyz_batch_cnt=xyz_batch_cnt,
                    new_xyz=keypoints,
                    new_xyz_batch_cnt=keypoints_batch_cnt,
                    features=features,
                )
                point_features_list.append(pooled_features)

            points = keypoints
            point_features_list.append(keypoints_features)
            point_features = torch.cat(point_features_list, dim=1)
            batch_cnt = keypoints.new_zeros(batch_size).int().fill_(num_keypoints)

        keypoints = points.view(-1, num_keypoints, 3)
        point_features_list = [point_features.view(batch_size, num_keypoints, -1)]
        if 'bev' in feature_sources:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            point_features_list.append(point_bev_features)

        point_features = torch.cat(point_features_list, dim=2)

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict
