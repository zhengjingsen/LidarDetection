'''
@Time       : 
@Author     : Jingsen Zheng
@File       : tracker
@Brief      : 
'''
import torch
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment    # deprecated
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanBoxTracker
from pcdet.utils.box_utils import boxes3d_nearest_bev_iou

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.01):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 7
  trackers:    M x 7


  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if (len(trackers)==0):
    return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

  iou_matrix = boxes3d_nearest_bev_iou(torch.from_numpy(detections),
                                       torch.from_numpy(trackers)).numpy()

  # matched_indices = linear_assignment(-iou_matrix)      # hougarian algorithm, compatible to linear_assignment in sklearn.utils

  row_ind, col_ind = linear_sum_assignment(-iou_matrix)      # hougarian algorithm
  matched_indices = np.stack((row_ind, col_ind), axis=1)

  unmatched_detections = []
  for d, det in enumerate(detections):
    if (d not in matched_indices[:, 0]): unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if (t not in matched_indices[:, 1]): unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if (iou_matrix[m[0], m[1]] < iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else: matches.append(m.reshape(1, 2))
  if (len(matches) == 0):
    matches = np.empty((0, 2),dtype=int)
  else: matches = np.concatenate(matches, axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):        # A baseline of 3D multi-object tracking
  def __init__(self, config, max_age=2, min_hits=3):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.reorder = [0, 1, 2, 6, 3, 4, 5]
    self.reorder_back = [0, 1, 2, 4, 5, 6, 3]

  def update_tracking(self, dets_all):
    """
    Params:
      dets_all: dict
      dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
      info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets = dets_all[0]['pred_boxes'].cpu().detach().numpy().astype(np.float32)     # dets: N x 7, float numpy array
    scores = dets_all[0]['pred_scores'].cpu().detach().numpy()
    labels = dets_all[0]['pred_labels'].cpu().detach().numpy()
    info = np.concatenate([scores[:, np.newaxis], labels[:, np.newaxis]], axis=-1)

    # reorder the data to put x,y,z in front to be compatible with the state transition matrix
    # where the constant velocity model is defined in the first three rows of the matrix
    dets = dets[:, self.reorder]          # reorder the data to [[x,y,z,theta,l,w,h], ...]
    self.frame_count += 1

    trks = np.zeros((len(self.trackers), 7), dtype=np.float32)         # N x 7 , # get predicted locations from existing trackers.
    to_del = []
    info_tracked_objects = {'object_ids': [], 'object_types': [], 'det_scores': [], 'pred_boxes': []}
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1, 1))
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
      if (np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets[:, self.reorder_back], trks[:, self.reorder_back])

    # update matched trackers with assigned detections
    for t, trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
        trk.update(dets[d, :][0], info[d, :][0])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
      trk = KalmanBoxTracker(dets[i, :], info[i, :])
      self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()      # bbox location
      d = d[self.reorder_back]      # change format from [x,y,z,theta,l,w,h] to [x,y,z,l,w,h,theta]

      if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
        info_tracked_objects['object_ids'].append(trk.id + 1)
        info_tracked_objects['object_types'].append(trk.info[1])
        info_tracked_objects['det_scores'].append(trk.info[0])
        info_tracked_objects['pred_boxes'].append(d)
      i -= 1

      # remove dead tracklet
      if (trk.time_since_update >= self.max_age):
        self.trackers.pop(i)

    info_tracked_objects['object_ids'] = np.array(info_tracked_objects['object_ids'], dtype=np.int)
    info_tracked_objects['object_types'] = np.array(info_tracked_objects['object_types'], dtype=np.int)
    info_tracked_objects['det_scores'] = np.array(info_tracked_objects['det_scores'], dtype=np.float)
    info_tracked_objects['pred_boxes'] = np.array(info_tracked_objects['pred_boxes'], dtype=np.float)

    return info_tracked_objects
