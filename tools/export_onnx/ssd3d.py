'''
@Time       : 
@Author     : Jingsen Zheng
@File       : ssd3d
@Brief      : 
'''

import torch
import torch.nn as nn

import argparse
import numpy as np
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models.model_utils.conv_module import ConvModule
from tools.export_onnx.pointnet2_module import PointnetSAModuleMSG, Points_Sampler


class PointNet2SAMSG(nn.Module):
    """PointNet2 with Multi-scale grouping.

    Args:
        input_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radii (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        aggregation_channels (tuple[int]): Out channels of aggregation
            multi-scale grouping features.
        fps_mods (tuple[int]): Mod of FPS for each SA module.
        fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
            points which each SA module samples.
        dilated_group (tuple[bool]): Whether to use dilated ball query for
        out_indices (Sequence[int]): Output from which stages.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_sa = len(model_cfg.SA_CHANNELS)
        self.out_indices = model_cfg.OUT_INDICES
        assert max(self.out_indices) < self.num_sa
        assert len(model_cfg.NUM_POINTS) == len(model_cfg.RADII) == len(model_cfg.NUM_SAMPLES) == len(
            model_cfg.SA_CHANNELS) == len(model_cfg.AGGREGATION_CHANNELS)

        self.SA_modules = nn.ModuleList()
        self.aggregation_mlps = nn.ModuleList()
        sa_in_channel = input_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(model_cfg.SA_CHANNELS[sa_index])
            sa_out_channel = 0
            for radius_index in range(len(model_cfg.RADII[sa_index])):
                cur_sa_mlps[radius_index] = [sa_in_channel] + list(
                    cur_sa_mlps[radius_index])
                sa_out_channel += cur_sa_mlps[radius_index][-1]

            cur_fps_mod = model_cfg.FPS_MODS[sa_index]
            cur_fps_sample_range_list = model_cfg.FPS_SAMPLE_RANGE_LISTS[sa_index]

            self.SA_modules.append(
                PointnetSAModuleMSG(mlps=cur_sa_mlps,
                                    npoint=model_cfg.NUM_POINTS[sa_index],
                                    nsamples=model_cfg.NUM_SAMPLES[sa_index],
                                    radii=model_cfg.RADII[sa_index],
                                    fps_mod=cur_fps_mod,
                                    fps_sample_range_list=cur_fps_sample_range_list))
            skip_channel_list.append(sa_out_channel)
            self.aggregation_mlps.append(
                ConvModule(
                    sa_out_channel,
                    model_cfg.AGGREGATION_CHANNELS[sa_index],
                    conv_cfg=dict(type='Conv1d'),
                    norm_cfg=dict(type='BatchNorm1d'),
                    kernel_size=1,
                    bias=True))
            sa_in_channel = model_cfg.AGGREGATION_CHANNELS[sa_index]

        self.num_point_features = self.model_cfg.AGGREGATION_CHANNELS[-1]
        self.num_point_features_before_fusion = self.model_cfg.AGGREGATION_CHANNELS[-1]

    def _split_point_feats(self, points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        point_num_features = points.size(-1)
        if point_num_features > 3:
            features = points[..., 3:point_num_features].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features

    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, torch.Tensor]: Outputs of the last SA module.

                - sa_xyz (torch.Tensor): The coordinates of sa features.
                - sa_features (torch.Tensor): The features from the
                    last Set Aggregation Layers.
                - sa_indices (torch.Tensor): Indices of the \
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        sa_xyz = [xyz]
        sa_features = [features]

        out_sa_xyz = []
        out_sa_features = []

        for i in range(self.num_sa):
            cur_xyz, cur_features = self.SA_modules[i](
                sa_xyz[i], sa_features[i])
            cur_features = self.aggregation_mlps[i](cur_features)
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            if i in self.out_indices:
                out_sa_xyz.append(sa_xyz[-1])
                out_sa_features.append(sa_features[-1])

        return sa_xyz[-1], sa_features[-1]


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config of model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint to start from')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)

    return args, cfg

def main():
    model = PointNet2SAMSG(cfg.MODEL.BACKBONE_3D, 4)
    # model = Points_Sampler([2048], ['FS'])

    with torch.no_grad():
        checkpoint = torch.load(args.ckpt)
        model_state_disk = checkpoint['model_state']

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key[12:] in model.state_dict() and model.state_dict()[key[12:]].shape == model_state_disk[key].shape:
                update_model_state[key[12:]] = val

        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()

        # ###################################### Convert VFE model to ONNX ######################################
        # VFE input: [max_num_pillars, max_num_points_per_pillar, point_features]
        # model_input = (
        #     torch.randn([1, 4, 4096, 32], dtype=torch.float32, device=torch.device('cuda:0')),
        #     torch.randn([1, 4, 4096, 32], dtype=torch.float32, device=torch.device('cuda:0')),
        #     torch.randn([1, 4, 4096, 64], dtype=torch.float32, device=torch.device('cuda:0')))

        # model_input = (
        #     torch.randn([1, 96000, 4], dtype=torch.float32, device=torch.device('cuda:0')))
        pointcloud = np.fromfile(str('/media/jingsen/data/Dataset/plusai/mot_dataset/20201109T152801_j7-l4e-00011_18_57to77.bag/pointcloud/1604911901.000515.bin'),
                                 dtype=np.float32).reshape(1, -1, 4)

        # # points_xyz = torch.randn([1, 120000, 3], dtype=torch.float32, device=torch.device('cuda:0'))
        # points_xyz = torch.from_numpy(pointcloud[..., :3]).cuda()
        # # features = torch.randn([1, 1, 120000], dtype=torch.float32, device=torch.device('cuda:0'))
        # features = torch.from_numpy(pointcloud[..., 3]).reshape(1, 1, -1).cuda()
        # points_xyz_t = points_xyz.transpose(1, 2).contiguous()
        # features_with_xyz = torch.cat([points_xyz_t, features], dim=1).contiguous()
        # model_input = (points_xyz, points_xyz_t, features_with_xyz)
        # model_input_names = ['points_xyz', 'points_xyz_t', 'features_with_xyz']
        # model_output_names = ['output_sa_xyz']

        model_input = torch.from_numpy(pointcloud[:, :16384, :]).cuda()
        model_input_names = ['point_cloud']
        model_output_names = ['output_sa_xyz', 'output_sa_features']
        output_onnx_file = 'ssd3d.onnx'
        torch.onnx.export(model, model_input, output_onnx_file, verbose=True,
                          input_names=model_input_names, output_names=model_output_names)
        print("[SUCCESS] SSD3D model is converted to ONNX.")

        result = model(model_input)
        print(result[1])


import onnx
import onnxruntime
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils
from pcdet.utils import common_utils
from pcdet.datasets.plusai.plusai_bag_dataset import DemoDataset
from pcdet.models import build_network, load_data_to_gpu
def model_validation():
    # ###################################### Validate model ONNX/PyTorch ######################################
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(''))
    torch_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    logger = common_utils.create_logger()
    torch_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    torch_model.cuda()
    torch_model.eval()

    print("Validating VFE ONNX model ...")
    model_input = (
        torch.randn([1, 4096, 4], dtype=torch.float32, device=torch.device('cuda:0')))
    batch_dict = {'batch_size': 1,
                  'points': model_input}
    batch_dict = torch_model.backbone_3d(batch_dict)
    out_torch_xyz = batch_dict['sa_xyz'][0]
    out_torch_features = batch_dict['sa_features'][0]

    # onnx_model = onnx.load("ssd3d.onnx")
    # onnx.checker.check_model(onnx_model)
    # onnx_session = onnxruntime.InferenceSession("ssd3d.onnx")
    # onnx_input_name = onnx_session.get_inputs()[0].name
    # onnx_output_name = [onnx_session.get_outputs()[0].name, onnx_session.get_outputs()[1].name]
    # out_onnx = onnx_session.run(onnx_output_name,
    #                             {onnx_input_name: model_input.detach().cpu().numpy()})

    model = PointNet2SAMSG(cfg.MODEL.BACKBONE_3D, 4)
    with torch.no_grad():
        checkpoint = torch.load(args.ckpt)
        model_state_disk = checkpoint['model_state']

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key[12:] in model.state_dict() and model.state_dict()[key[12:]].shape == model_state_disk[key].shape:
                update_model_state[key[12:]] = val

        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()

    ## run origin model
    # xyz, features = (
    #     torch.randn([1, 4096, 3], dtype=torch.float32, device=torch.device('cuda:0')),
    #     torch.randn([1, 1, 4096], dtype=torch.float32, device=torch.device('cuda:0')))
    # xyz_flipped = xyz.transpose(1, 2).contiguous()
    # indices = torch_model.backbone_3d.SA_modules[1].points_sampler(xyz, features)
    # new_xyz = pointnet2_utils.gather_operation(
    #     xyz_flipped,
    #     indices
    # )

    ## run export onnx model
    out_onnx_xyz, out_onnx_features = model(model_input)
    # features_with_xyz = torch.cat([xyz_flipped, features], dim=1).contiguous()  # (B, 3+C, N)
    # new_onnx_xyz = model.SA_modules[1].point_sampler(xyz, xyz_flipped, features_with_xyz)
    #
    # np.testing.assert_allclose(new_xyz.detach().cpu().numpy(),
    #                            new_onnx_xyz.detach().cpu().numpy(), rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(out_torch_xyz.detach().cpu().numpy(),
                               out_onnx_xyz.detach().cpu().numpy(), rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(out_torch_features.detach().cpu().numpy(),
                               out_onnx_features.detach().cpu().numpy(), rtol=1e-03, atol=1e-05)
    print("[SUCCESS] ONNX model validated.")


if __name__ == '__main__':
    args, cfg = parse_config()
    main()
    # model_validation()
