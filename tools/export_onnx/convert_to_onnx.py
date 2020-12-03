import argparse
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from tools.export_onnx.vfe_module import PillarVFE
from tools.export_onnx.rpn_module import RPN


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
    args, cfg = parse_config()

    vfe_model = PillarVFE(model_cfg=cfg.MODEL.VFE,
                          num_point_features=4)

    rpn_model = RPN(model_cfg=cfg.MODEL,
                    num_class=2,
                    class_names=['Car', 'Truck'],
                    grid_size=np.array([2000, 288, 1]),
                    point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE)

    with torch.no_grad():
        checkpoint = torch.load(args.ckpt)
        model_state_disk = checkpoint['model_state']

        vfe_update_model_state = {}
        rpn_update_model_state = {}
        for key, val in model_state_disk.items():
            if key[4:] in vfe_model.state_dict() and vfe_model.state_dict()[key[4:]].shape == model_state_disk[key].shape:
                vfe_update_model_state[key[4:]] = val
            if key in rpn_model.state_dict() and rpn_model.state_dict()[key].shape == model_state_disk[key].shape:
                rpn_update_model_state[key] = val

        vfe_state_dict = vfe_model.state_dict()
        vfe_state_dict.update(vfe_update_model_state)
        vfe_model.load_state_dict(vfe_state_dict)
        vfe_model.cuda()
        vfe_model.eval()

        rpn_state_dict = rpn_model.state_dict()
        rpn_state_dict.update(rpn_update_model_state)
        rpn_model.load_state_dict(rpn_state_dict)
        rpn_model.cuda()
        rpn_model.eval()

        # ###################################### Convert VFE model to ONNX ######################################
        # VFE input: [max_num_pillars, max_num_points_per_pillar, point_features]
        vfe_input = torch.randn(
            [30000, 32, 10], dtype=torch.float32, device=torch.device('cuda:0'))

        vfe_input_names = ['vfe_input']
        vfe_output_names = ['pillar_features']
        output_onnx_file = 'vfe.onnx'
        torch.onnx.export(vfe_model, vfe_input, output_onnx_file, verbose=True,
                          input_names=vfe_input_names, output_names=vfe_output_names)
        print("[SUCCESS] VFE model is converted to ONNX.")

        # ###################################### Convert RPN model to ONNX ######################################
        # RPN input: NCHW
        rpn_input = torch.ones(
            [1, 64, 288, 2000], dtype=torch.float32, device=torch.device('cuda:0'))
        rpn_input_names = ['spatial_features']
        rpn_output_names = ['cls_preds', 'box_preds', 'dir_cls_preds']
        output_onnx_file = 'rpn.onnx'
        torch.onnx.export(rpn_model, rpn_input, output_onnx_file, verbose=True,
                          input_names=rpn_input_names, output_names=rpn_output_names)
        print("[SUCCESS] RPN model is converted to ONNX.")


if __name__ == '__main__':
    main()
