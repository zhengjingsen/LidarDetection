'''
@Time       : 
@Author     : Jingsen Zheng
@File       : demo_3d.py
@Brief      : 
'''
import argparse
import torch
from pathlib import Path
import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.plusai.plusai_bag_dataset import DemoDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils.laserdetvis import LaserDetVis


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


logger = common_utils.create_logger()

class VisualizeDets(LaserDetVis):
  def __init__(self, model, dataset):
    super(VisualizeDets, self).__init__(show_img=False)
    self.model = model
    self.dataset = dataset

    self.data_idx = np.arange(len(self.dataset)).tolist()
    np.random.shuffle(self.data_idx)

    self.offset = 0
    self.update()

  def update(self):
    idx = self.offset % len(self.dataset)
    # idx = self.data_idx[idx]

    with torch.no_grad():
      data_dict = self.dataset.__getitem__(idx)

      logger.info(f'Visualized sample index: \t{idx + 1}')
      data_dict = self.dataset.collate_batch([data_dict])
      load_data_to_gpu(data_dict)
      pred_dicts, _ = self.model.forward(data_dict)

      # img_path = os.path.join(self.root_path, example['image_path'])
      # img = cv2.imread(img_path)
      # Show
      gt_objs = None
      if self.dataset.split == 'val':
          gt_objs = self.dataset.val_data_list[idx]['annos']['gt_boxes_lidar']
      self.update_view(idx,
                       points=data_dict['points'][:, 1:].cpu().numpy(),
                       objs=pred_dicts[0]['pred_boxes'].cpu(),
                       ref_scores=pred_dicts[0]['pred_scores'].cpu().numpy(),
                       ref_labels=pred_dicts[0]['pred_labels'].cpu().numpy(),
                       gt_objs=gt_objs,
                       # img=img
                       )

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    if self.show_img:
      self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.offset += 1
      self.update()
    elif event.key == 'B':
      self.offset -= 1
      self.update()
    elif event.key == 'C':
      self.intensity_mode = not self.intensity_mode
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()

def main():
    args, cfg = parse_config()
    logger.info('-----------------Quick Demo 3D of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")
    # run the visualizer
    vis = VisualizeDets(model, demo_dataset)
    vis.run()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
