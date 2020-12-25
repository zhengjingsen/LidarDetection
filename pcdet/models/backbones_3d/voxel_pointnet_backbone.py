'''
@Time       : 
@Author     : Jingsen Zheng
@File       : voxel_pointnet_backbone
@Brief      : 
'''
from functools import partial

import spconv
import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...utils import common_utils

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class VoxelPointnetBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = [0.1, 0.1, 0.2]
        self.point_cloud_range = [2, -16, -2, 162, 16, 6]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(128, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(128, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

        # pointnet params
        self.downsample_times_map = {'input': 1, 'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8}

        self.SA_layer1 = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[0.4, 0.8],
            nsamples=[16, 16],
            mlps=[[2, 16, 16], [2, 16, 16]],
            use_xyz=True,
            pool_method='max_pool',
        )

        self.SA_layer2 = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[0.8, 1.2],
            nsamples=[16, 32],
            mlps=[[32, 32, 32], [32, 32, 32]],
            use_xyz=True,
            pool_method='max_pool',
        )

        self.SA_layer3 = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[1.2, 2.4],
            nsamples=[16, 32],
            mlps=[[64, 64, 64], [64, 64, 64]],
            use_xyz=True,
            pool_method='max_pool',
        )

        self.SA_layer4 = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[2.4, 4.8],
            nsamples=[16, 32],
            mlps=[[64, 64, 64], [64, 64, 64]],
            use_xyz=True,
            pool_method='max_pool',
        )

        self.SA_layer5 = pointnet2_stack_modules.StackSAModuleMSG(
            radii=[4.8],
            nsamples=[32],
            mlps=[[128, 128, 128]],
            use_xyz=True,
            pool_method='max_pool',
        )

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.FP_layer1 = pointnet2_stack_modules.StackPointnetFPModule(mlp=[32, 16])
        self.FP_layer2 = pointnet2_stack_modules.StackPointnetFPModule(mlp=[64, 32])
        self.FP_layer3 = pointnet2_stack_modules.StackPointnetFPModule(mlp=[128, 64])
        self.FP_layer4 = pointnet2_stack_modules.StackPointnetFPModule(mlp=[128, 64])

        self.num_point_features_before_fusion = 128
        self.num_keypoints = 4096

    def get_sampled_points(self, batch_dict):
        src_points = batch_dict['voxels'][:, 0, 0:3]
        src_points = torch.cat([batch_dict['voxel_coords'][:, 0:1], src_points], dim=1)

        points = src_points[:, 1:4]
        points_feature = src_points[:, 4:]
        batch_size = batch_dict['batch_size']
        batch_cnt = points.new_zeros(batch_size).int()
        for i in range(batch_size):
            batch_cnt[i] = (src_points[:, 0] == i).sum()

        keypoints_list = []
        keypoints_feature_list = []
        cur_idx = 0
        for cur_cnt in batch_cnt.cpu().numpy():
            sampled_points = points[cur_idx:cur_idx + cur_cnt, :]
            sampled_points_feature = points_feature[cur_idx:cur_idx + cur_cnt, :]

            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                sampled_points.unsqueeze(dim=0).contiguous(), self.num_keypoints
            ).long()

            if sampled_points.shape[0] < self.num_keypoints:
                empty_num = self.num_keypoints - sampled_points.shape[0]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

            keypoints = sampled_points[cur_pt_idxs[0]]
            keypoints_list.append(keypoints)

            keypoints_feature = sampled_points_feature[cur_pt_idxs[0]]
            keypoints_feature_list.append(keypoints_feature)

            cur_idx += cur_cnt

        keypoints = torch.cat(keypoints_list, dim=0)  # (M1 + M2 + ..., 3)
        keypoints_feature = torch.cat(keypoints_feature_list, dim=0)  # (M1+ M2 + ..., C)
        batch_cnt = keypoints.new_zeros(batch_cnt.shape[0]).int().fill_(self.num_keypoints)
        return keypoints, keypoints_feature, batch_cnt

    def get_pooled_feature(self, keypoints, keypoints_batch_cnt, x_conv, down_factor, SA_layer):
        voxel_coords = x_conv.indices
        voxel_centers = common_utils.get_voxel_centers(
            voxel_coords[:, 1:4],
            downsample_times=down_factor,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        batch_size = keypoints_batch_cnt.shape[0]
        batch_cnt = keypoints_batch_cnt.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            batch_cnt[bs_idx] = (voxel_coords[:, 0] == bs_idx).sum()

        pooled_points, pooled_features = SA_layer(
            xyz=voxel_centers,
            xyz_batch_cnt=batch_cnt,
            new_xyz=keypoints,
            new_xyz_batch_cnt=keypoints_batch_cnt,
            features=x_conv.features.contiguous()
        )

        return pooled_points, pooled_features

    def get_pooled_feature_raw_points(self, keypoints, keypoints_batch_cnt, batch_dict, SA_layer):
        batch_size = batch_dict['batch_size']
        raw_points = batch_dict['points']
        xyz = raw_points[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()
        features = raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None

        pooled_points, pooled_features = SA_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=keypoints,
            new_xyz_batch_cnt=keypoints_batch_cnt,
            features=features.contiguous()
        )

        return pooled_points, pooled_features

    def conv(self, x, down_factor, xyz, xyz_batch_cnt, features, FP_layer, conv_layer):
        cur_coords = x.indices
        new_xyz = common_utils.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=down_factor,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        batch_size = xyz_batch_cnt.shape[0]
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            new_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

        new_features = FP_layer(new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, known_feats=features.contiguous())
        x.features = torch.cat([x.features, new_features], dim=-1)

        x = conv_layer(x)
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        xyz, _, xyz_batch_cnt = self.get_sampled_points(batch_dict)

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        _, pooled_features = self.get_pooled_feature_raw_points(xyz, xyz_batch_cnt, batch_dict, self.SA_layer1)
        x_conv1 = self.conv(x, 1, xyz, xyz_batch_cnt, pooled_features, self.FP_layer1, self.conv1)

        _, pooled_features = self.get_pooled_feature(xyz, xyz_batch_cnt, x_conv1, 1, self.SA_layer2)
        x_conv2 = self.conv(x_conv1, 2, xyz, xyz_batch_cnt, pooled_features, self.FP_layer2, self.conv2)

        _, pooled_features = self.get_pooled_feature(xyz, xyz_batch_cnt, x_conv2, 2, self.SA_layer3)
        x_conv3 = self.conv(x_conv2, 4, xyz, xyz_batch_cnt, pooled_features, self.FP_layer3, self.conv3)

        _, pooled_features = self.get_pooled_feature(xyz, xyz_batch_cnt, x_conv3, 4, self.SA_layer4)
        x_conv4 = self.conv(x_conv3, 8, xyz, xyz_batch_cnt, pooled_features, self.FP_layer4, self.conv4)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        _, pooled_features = self.get_pooled_feature(xyz, xyz_batch_cnt, x_conv4, 8, self.SA_layer5)

        batch_idx = torch.arange(batch_size, device=xyz.device).view(-1, 1).repeat(1, self.num_keypoints).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), xyz), dim=1)
        batch_dict['point_features_before_fusion'] = pooled_features
        point_features = self.vsa_point_feature_fusion(pooled_features)

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)

        return batch_dict
