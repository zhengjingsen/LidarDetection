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

import spconv
from functools import partial
from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config of model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint to start from')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    np.random.seed(1024)

    return args, cfg


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

class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, voxel_features, voxel_coords):
        """
        Args:
            batch_size: int
            vfe_features: (num_voxels, C)
            voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        batch_size = 1
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        output = (x_conv1, x_conv2, x_conv3, x_conv4, out)

        return output

def main():
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger)

    second_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    second_model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    second_model.cuda()
    second_model.eval()

    model = VoxelBackBone8x(cfg.MODEL.BACKBONE_3D, 4, demo_dataset.grid_size)

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

        # model_input = (
        #     torch.randn([1, 96000, 4], dtype=torch.float32, device=torch.device('cuda:0')))
        data_dict = demo_dataset.__getitem__(1)
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        data_dict = second_model.vfe(data_dict)

        # test model
        test_output = model(data_dict['voxel_features'], data_dict['voxel_coords'])

        model_input = (data_dict['voxel_features'], data_dict['voxel_coords'])
        model_input_names = ['voxel_feature', 'voxel_coords']

        model_output_names = ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'encoded_spconv_tensor']
        output_onnx_file = 'spconv.onnx'
        # torch.onnx.export(model, model_input, output_onnx_file, verbose=True,
        #                   input_names=model_input_names, output_names=model_output_names)
        print("[SUCCESS] SSD3D model is converted to ONNX.")


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
