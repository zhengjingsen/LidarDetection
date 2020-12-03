import argparse
from pathlib import Path

import numpy as np
import torch
import onnxruntime
import onnx

from pcdet.config import cfg, cfg_from_yaml_file
from tools.export_onnx.vfe_module import PillarVFE
from tools.export_onnx.rpn_module import RPN


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of model')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
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

        # ###################################### Validate VFE model ONNX/PyTorch ######################################
        print("Validating VFE ONNX model ...")
        vfe_input = torch.randn([30000, 32, 10], dtype=torch.float32, device=torch.device('cuda:0'))
        vfe_out_torch = vfe_model(vfe_input)

        vfe_onnx_model = onnx.load("vfe.onnx")
        onnx.checker.check_model(vfe_onnx_model)
        onnx_vfe_session = onnxruntime.InferenceSession("vfe.onnx")
        onnx_vfe_input_name = onnx_vfe_session.get_inputs()[0].name
        onnx_vfe_output_name = [onnx_vfe_session.get_outputs()[0].name]
        vfe_out_onnx = onnx_vfe_session.run(onnx_vfe_output_name,
                                            {onnx_vfe_input_name: vfe_input.detach().cpu().numpy()})

        np.testing.assert_allclose(vfe_out_torch.detach().cpu().numpy(), vfe_out_onnx[0], rtol=1e-03, atol=1e-05)
        print("[SUCCESS] VFE ONNX model validated.")

        # ####################################### Validate RPN model ONNX/PyTorch ######################################
        print("Validating RPN ONNX model ...")
        rpn_input = torch.randn([1, 64, 288, 2000], dtype=torch.float32, device=torch.device('cuda:0'))
        rpn_out_torch = rpn_model(rpn_input)

        rpn_onnx_model = onnx.load("rpn.onnx")
        onnx.checker.check_model(rpn_onnx_model)
        onnx_rpn_session = onnxruntime.InferenceSession("rpn.onnx")
        onnx_rpn_input_name = onnx_rpn_session.get_inputs()[0].name
        onnx_rpn_output_name = [onnx_rpn_session.get_outputs()[0].name,
                                onnx_rpn_session.get_outputs()[1].name,
                                onnx_rpn_session.get_outputs()[2].name]
        rpn_out_onnx = onnx_rpn_session.run(onnx_rpn_output_name,
                                            {onnx_rpn_input_name: rpn_input.detach().cpu().numpy()})

        np.testing.assert_allclose(rpn_out_torch[0].detach().cpu().numpy(), rpn_out_onnx[0], rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(rpn_out_torch[1].detach().cpu().numpy(), rpn_out_onnx[1], rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(rpn_out_torch[2].detach().cpu().numpy(), rpn_out_onnx[2], rtol=1e-03, atol=1e-05)
        print("[SUCCESS] RPN ONNX model validated.")


if __name__ == '__main__':
    main()
