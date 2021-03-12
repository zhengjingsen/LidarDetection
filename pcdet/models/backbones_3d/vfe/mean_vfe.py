import torch
import torch.nn as nn
from ....ops.scatter.scatter_max import scatter_max, scatter_mean

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        return batch_dict


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


def dense(batch_size, spatial_shape, feature_dim, indices, features, channels_first=True):
    output_shape = [batch_size] + list(spatial_shape) + [feature_dim]
    res = scatter_nd(indices.long(), features, output_shape)
    if not channels_first:
        return res
    ndim = len(spatial_shape)
    trans_params = list(range(0, ndim + 1))
    trans_params.insert(1, ndim + 1)
    return res.permute(*trans_params).contiguous()


class ScatterMaxCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_index, output, output_index):
        scatter_max(input, input_index, output, output_index, True)
        ctx.size = input.size()
        ctx.save_for_backward(output_index)

        return output

    @staticmethod
    def backward(ctx, output_grad):
        output_index = ctx.saved_tensors[0]
        grad_input = output_grad.new_zeros(ctx.size)
        # print("test grad")
        # print("max : ", output_index.max())
        # print("min : ", output_index.min())
        # print("points counts : ", ctx.size[0])
        grad_input.scatter_(0, output_index, output_grad)

        return grad_input, None, None, None


def scatterMax(input, input_index, voxel_nums, train):
    '''
        only accept two dimension tensor, and do maxpooing in first dimension
    '''
    output = input.new_full((voxel_nums, input.shape[1]), torch.finfo(input.dtype).min)
    output_index = input_index.new_empty((voxel_nums, input.shape[1]))

    if train:
        output = ScatterMaxCuda.apply(input, input_index, output, output_index)
    else:
        output = scatter_max(input, input_index, output, output_index, False)

    return output


def scatterMean(input, input_index, voxel_nums):
    output = input.new_full((voxel_nums, input.shape[1]), 0.0)
    input_mean = input.new_empty(input.shape)

    scatter_mean(input, input_index, output, input_mean)
    return output


class VoxelFeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        raise NotImplementedError

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class MeanVFEDV(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.output_channels = 5

        # self.FC = nn.Sequential(
        #     nn.Linear(11, self.output_channels, bias=False),
        #     nn.BatchNorm1d(self.output_channels, eps=1e-3, momentum=0.01),
        #     nn.ReLU(inplace=True)
        # )

    def get_output_feature_dim(self):
        return self.output_channels

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """

        batch_size = batch_dict['batch_size']
        bev_coordinate = batch_dict['bev_coordinate']
        bev_local_coordinate = batch_dict['bev_local_coordinate']
        intensity = batch_dict['intensity']
        bev_mapping_pv = batch_dict['bev_mapping_pv']
        # throw z position
        bev_mapping_vf = batch_dict['bev_mapping_vf'][:, :3].contiguous()

        # point_mean = scatterMean(bev_coordinate, bev_mapping_pv, bev_mapping_vf.shape[0])
        # feature = torch.cat(
        #     (bev_coordinate, intensity, (bev_coordinate - point_mean), bev_local_coordinate),
        #     dim=1).contiguous()
        #
        # bev_fc_output = self.FC(feature)
        # bev_maxpool = scatterMax(bev_fc_output, bev_mapping_pv, bev_mapping_vf.shape[0], True)

        feature = scatterMean(torch.cat([bev_coordinate, intensity], dim=1).contiguous(), bev_mapping_pv, bev_mapping_vf.shape[0])
        batch_dict['voxel_features'] = feature.contiguous()
        batch_dict['voxel_coords'] = batch_dict['bev_mapping_vf']
        return batch_dict
