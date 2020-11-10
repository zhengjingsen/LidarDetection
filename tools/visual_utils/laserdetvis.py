#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
from .visualize_utils import check_numpy_to_torch, rotate_points_along_z
from vispy import color

class LaserDetVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, show_img=False):
    self.show_img = show_img
    self.canvas_size = (1920, 1920)
    self.running = True

    self.reset()

  def is_running(self):
    return self.running

  def key_press(self, event):
    raise NotImplementedError

  '''
  Parameters: 
    points: N x 3 or N x 4
  '''
  def add_points(self, points):
    self.points = points[:, 0:3]
    if points.shape[1] == 4:
      intensity = points[:, 3]
      intensity = ((intensity - intensity.min()) /
                   (intensity.max() - intensity.min()) *
                   128 + 127).astype(np.uint8)
      viridis_map = self.get_mpl_colormap("viridis")
      self.viridis_colors = viridis_map[intensity]

  """ 
  Takes an object and a projection matrix (P) and projects the 3d
    bounding box into the image plane.
    Returns:
      corners_2d: (8,2) array in left image coord.
      corners_3d: (8,3) array in in rect camera coord.
  """
  def compute_box_3d(self, obj):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(obj)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[None, 3:6].repeat(8, 1) * template[:, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[None, 6]).view(8, 3)
    corners3d += boxes3d[0:3]

    connect = np.array([[0, 1], [1, 5], [0, 4], [4, 5],
                        [1, 2], [0, 3], [5, 6], [4, 7],
                        [2, 3], [2, 6], [3, 7], [6, 7]], dtype=np.int32)

    return corners3d if is_numpy else corners3d.numpy(), connect

  '''
  Parameters: 
    objs: N x 7 (x, y, z, w, l, h, yaw)
  '''
  def add_objs(self, objs):
    obj_vertices = []
    obj_vert_connect = []
    obj_label_pos = []
    for i in range(len(objs)):
      vertices, connect = self.compute_box_3d(objs[i])
      obj_vertices.append(vertices)
      obj_vert_connect.append(connect + i * len(vertices))
      obj_label_pos.append((vertices[-1, 0], vertices[-1, 1], vertices[-1, 2]))

    obj_vertices = np.concatenate(obj_vertices, axis=0)
    obj_vert_connect = np.concatenate(obj_vert_connect, axis=0)

    return obj_vertices, obj_vert_connect, obj_label_pos

  '''
  Parameters: 
    data : ndarray
        ImageVisual data. Can be shape (M, N), (M, N, 3), or (M, N, 4).
  '''
  def add_image(self, img):
    self.image = img

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True, size=self.canvas_size)
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.scan_view.camera = vispy.scene.TurntableCamera(elevation=30,
                                                        azimuth=-90,
                                                        distance=30,
                                                        translate_speed=30,
                                                        up='+z')
    # grid
    self.grid = self.canvas.central_widget.add_grid()
    self.grid.add_widget(self.scan_view)

    self.scan_vis = visuals.Markers(parent=self.scan_view.scene)
    self.scan_vis.antialias = 0
    # self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)

    self.line = visuals.Line(width=1, method='gl', parent=self.scan_view.scene)
    self.text = visuals.Text(color='green', font_size=600, bold=True, parent=self.scan_view.scene)
    self.gt_line = visuals.Line(width=1000, parent=self.scan_view.scene)
    # self.sem_view.camera.link(self.scan_view.camera)

    if self.show_img:
      # img canvas size

      # new canvas for img
      self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                    size=(1242, 375))
      # grid
      self.img_grid = self.img_canvas.central_widget.add_grid()
      # interface (n next, b back, q quit, very simple)
      self.img_canvas.events.key_press.connect(self.key_press)
      self.img_canvas.events.draw.connect(self.draw)

      # add a view for the depth
      self.img_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.img_view, 0, 0)
      self.img_vis = visuals.Image(cmap='viridis')
      self.img_view.add(self.img_vis)

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def update_view(self, idx, points=None, objs=None, gt_objs=None,
                  ref_scores=None, ref_labels=None,
                  img=None):
    # then change names
    title = "scan " + str(idx)
    self.canvas.title = title

    # then do all the point cloud stuff

    # plot scan
    if points is not None:
      self.add_points(points)
    if self.viridis_colors is not None:
      self.scan_vis.set_data(self.points,
                             # face_color=self.viridis_colors[..., ::-1],
                             # edge_color=self.viridis_colors[..., ::-1],
                             face_color='white',
                             edge_color='white',
                             size=1)
    else:
      self.scan_vis.set_data(self.points, size=1)

    # plot objs
    if objs is None:
      self.line.set_data()
    else:
      obj_vertices, obj_vert_connect, obj_label_pos = self.add_objs(objs)
      self.line.set_data(pos=obj_vertices, color='green',
                         connect=obj_vert_connect)
      if ref_scores is not None and ref_labels is not None:
        labels = []
        for i in range(len(ref_labels)):
          labels.append('{:.2f}'.format(ref_scores[i]))
        self.text.text = labels
        self.text.pos = obj_label_pos

    if gt_objs is None:
      self.gt_line.set_data()
    else:
      gt_obj_vertices, gt_obj_vert_connect, _ = self.add_objs(gt_objs)
      self.gt_line.set_data(pos=gt_obj_vertices, color='green',
                            connect=gt_obj_vert_connect)

    # plot image
    if self.show_img:
      self.img_canvas.title = title
      if img is not None:
        self.add_image(img)
      self.img_vis.set_data(self.image)
      self.img_vis.update()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.show_img:
      if self.img_canvas.events.key_press.blocked():
        self.img_canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    if self.show_img:
      self.img_canvas.close()

    vispy.app.quit()
    self.running = False

  def run(self):
    vispy.app.run()
