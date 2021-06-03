'''
@Time       : 
@Author     : Jingsen Zheng
@File       : pointnet2_module
@Brief      : 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from torch.autograd import Function, Variable
import time
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda as pointnet2

class SampleAndGather(Function):
    @staticmethod
    def symbolic(g, feature_t: torch.Tensor, xyz_t: torch.Tensor, npoint: int):
        return g.op('SampleAndGather', feature_t, xyz_t, npoint_i=npoint)

    @staticmethod
    def forward(ctx, feature_t: torch.Tensor, xyz_t: torch.Tensor, npoint: int) \
            -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint points and features
        :param ctx:
        :param feature_t: (B, N, C) where N > npoint
        :param xyz_t: (B, 3, N)
        :param npoint: int, number of points to be sampled
        : return:
            output: (B, 3, npoint)
        """
        assert feature_t.is_contiguous()
        assert xyz_t.is_contiguous()

        B, N, C = feature_t.size()
        idx = torch.cuda.IntTensor(B, npoint).fill_(0)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        start = time.time()
        pointnet2.feature_furthest_point_sampling_wrapper(B, N, npoint, C, feature_t, temp, idx)
        print("idx: ", idx[-1, -1])
        end = time.time()
        print('sampling elapsed time: ', (end - start)*1000.0)
        ctx.mark_non_differentiable(idx)

        start = time.time()
        _, C, _ = xyz_t.size()
        output = torch.cuda.FloatTensor(B, C, npoint)
        pointnet2.gather_points_wrapper(B, C, N, npoint, xyz_t, idx, output)
        print("output: ", output[0, :, -1])
        end = time.time()
        print('gather elapsed time: ', (end - start)*1000.0)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return None, grad_features, None

sample_and_gather = SampleAndGather.apply

from torch.onnx.symbolic_helper import parse_args
class QueryAndGroup(Function):
    @staticmethod
    @parse_args('v', 'v', 'v', 'f', 'i')
    def symbolic(g, xyz: torch.Tensor, new_xyz: torch.Tensor, features_with_xyz: torch.Tensor,
                 radius: float, nsample: int):
        return g.op('QueryAndGroup', xyz, new_xyz, features_with_xyz, radius_f=radius, nsample_i=nsample)

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, new_xyz: torch.Tensor, features_with_xyz: torch.Tensor,
                radius: float, nsample: int) -> torch.Tensor:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features_with_xyz: (B, 3 + C, N) descriptors of the features
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :return:
            grouped_xyz: (B, 3, npoint, nsample)
            grouped_features: (B, C, npoint, nsample)
        """
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        assert features_with_xyz.is_contiguous()

        B, C, N = features_with_xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        ctx.mark_non_differentiable(idx)

        # group features
        grouped_features = torch.cuda.FloatTensor(B, C, npoint, nsample) # (B, 3 + C, npoint, nsample)
        pointnet2.group_points_wrapper(B, C, N, npoint, nsample, features_with_xyz, idx, grouped_features)

        ctx.for_backwards = (idx, N)

        return grouped_features

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.cuda.FloatTensor(B, C, N).zero_()
        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return None, None, grad_features, None, None

query_and_group = QueryAndGroup.apply


class Points_Sampler(nn.Module):
    """Points sampling.

    Args:
        num_point (list[int]): Number of sample points.
        fps_mod_list (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
    """

    def __init__(self,
                 num_point: List[int],
                 fps_mod_list: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1]):
        super(Points_Sampler, self).__init__()
        # FPS would be applied to different fps_mod in the list,
        # so the length of the num_point should be equal to
        # fps_mod_list and fps_sample_range_list.
        assert len(num_point) == len(fps_mod_list) == len(
            fps_sample_range_list)
        self.num_point = num_point
        self.fps_mod_list = fps_mod_list
        self.fps_sample_range_list = fps_sample_range_list

    def forward(self, points_xyz, points_xyz_t, features_with_xyz):
        """forward.
        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            points_xyz_t (Tensor): (B, 3, N) xyz coordinates of the features.
            features_with_xyz (Tensor): (B, 3+C, N) Descriptors of the features.

        Return：
            Tensor: (B, npoint, sample_num) Indices of sampled points.
        """
        output = []
        last_fps_end_index = 0

        for fps_sample_range, fps_mod, npoint in zip(
                self.fps_sample_range_list, self.fps_mod_list, self.num_point):
            assert fps_sample_range < points_xyz.shape[1]

            if fps_sample_range == -1 and last_fps_end_index == 0:
                    sample_points_xyz = points_xyz
                    sample_points_xyz_t = points_xyz_t
                    sample_features = features_with_xyz
            else:
                if fps_sample_range == -1:
                    fps_sample_range = points_xyz.size(1)

                sample_points_xyz = \
                    points_xyz[:, last_fps_end_index:fps_sample_range]
                sample_points_xyz_t = \
                    points_xyz_t[:, :, last_fps_end_index:fps_sample_range]
                sample_features = \
                    features_with_xyz[:, :, last_fps_end_index:fps_sample_range]

            if fps_mod == 'F-FPS' or fps_mod == 'FS':
                output.append(sample_and_gather(sample_features.transpose(1, 2).contiguous(),
                                                sample_points_xyz_t.contiguous(),
                                                npoint))
            if fps_mod == 'D-FPS' or fps_mod == 'FS':
                output.append(sample_and_gather(sample_points_xyz.contiguous(),
                                                sample_points_xyz_t.contiguous(),
                                                npoint))

            last_fps_end_index += fps_sample_range

        if len(output) == 1:
            output = output[0]
        else:
            output = torch.cat(output, dim=2)

        return output


class PointnetSAModuleMSG(nn.Module):
    def __init__(self, mlps: List[List[int]], npoint: int, nsamples: List[int], radii: List[float],
                 pool_method='max_pool', use_xyz: bool = True,
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1]):
        super().__init__()
        assert len(mlps) == len(nsamples)

        self.nsamples = nsamples
        self.radius = radii
        self.num_balls = len(mlps)

        self.mlps = nn.ModuleList()
        self.groupers = nn.ModuleList()
        sa_out_channel = 0
        for i in range(len(mlps)):
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            sa_out_channel += mlp_spec[-1]


        if isinstance(npoint, int):
            npoint = [npoint]
        self.point_sampler = Points_Sampler(npoint, fps_mod, fps_sample_range_list)

        self.pool_method = pool_method

    def forward(self, xyz, features):
        """
        :param xyz:  #(B, N, 3) tensor of the xyz coordinates of the features
        :param features:  #(B, C, N) tensor of the descriptors of the the features
        """
        xyz_t = xyz.transpose(1, 2)
        features_with_xyz = torch.cat([xyz_t, features], dim=1).contiguous()  # (B, 3+C, N)

        new_xyz_t = self.point_sampler(xyz, xyz_t, features_with_xyz)
        new_xyz = new_xyz_t.transpose(1, 2).contiguous()

        new_features_list = []
        new_xyz_t = new_xyz_t.unsqueeze(dim=3)
        for k in range(self.num_balls):
            group_features = query_and_group(xyz, new_xyz, features_with_xyz, self.radius[k], self.nsamples[k])
            feature_channel_num = group_features.size(1)
            new_features = torch.cat([group_features[:, 0:3, ...] - new_xyz_t,
                                      group_features[:, 3:feature_channel_num, ...]], dim=1)

            new_features = self.mlps[k](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, self.nsamples[k]]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1).squeeze(dim=3)

        return new_xyz, new_features
