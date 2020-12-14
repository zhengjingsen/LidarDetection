import math
import copy
import torch
import numpy as np
from pykalman import KalmanFilter
from pcdet.utils.box_utils import boxes3d_nearest_bev_iou

class DetectedObject(object):
    def __init__(self, pred_box, name, score):
        self.loc = pred_box[:3]
        self.dims = pred_box[3:6]
        self.rotz = pred_box[6]
        self.type = name
        self.score = score

    def __repr__(self):
        object_info = f"[Detected Object] location: {self.loc} | size: {self.dims} | rotz: {self.rotz} | score: {self.score}"
        return object_info


class ObjectTracker(object):
    def __init__(self, det_object, track_id):
        self.loc = det_object.loc
        self.dims = det_object.dims
        self.rotz = det_object.rotz
        self.type = det_object.type
        self.reliability = det_object.score
        self.velo = [0.0, 0.0]

        self.real_path = [self.loc[:2]]
        self.predicted_path = [self.loc[:2]]
        self.dims_history = [det_object.dims]

        self.track_id = track_id
        self.new_object = True
        self.updated = False
        self.predicted = False
        self.age = 1
        self.lose_tracking = 0

        self.kf = KalmanFilter(transition_matrices=np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]),
                               observation_matrices=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                               transition_covariance=0.03 * np.eye(4))
        self.filtered_state_means = np.array([0.0, 0.0, 0.0, 0.0])
        self.filtered_state_covariances = np.eye(4)
        self.delta_t = 0.1
        self.max_dims_track_range = 7
        self.length_diff_ratio_thres = 0.2
        self.no_dims_filter_range = 8

    def update(self, det_object):
        if abs(det_object.dims[0] - self.dims[0]) / self.dims[0] > self.length_diff_ratio_thres \
                and self.loc[0] > self.no_dims_filter_range:
            self.loc[0] = det_object.loc[0] - det_object.dims[0] / 2 + self.dims[0] / 2
            self.loc[1:3] = det_object.loc[1:3]
        else:
            self.loc = det_object.loc
        self.rotz = det_object.rotz
        self.type = det_object.type
        self.reliability = det_object.score
        self.velo = [(self.loc[0] - self.real_path[-1][0]) / self.delta_t,
                     (self.loc[1] - self.real_path[-1][1]) / self.delta_t]

        self.real_path.append(det_object.loc[:2])
        self.dims_history.append(det_object.dims)
        self.dimension_filter()

        self.new_object = False
        self.updated = True
        self.predicted = False
        self.age += 1
        self.lose_tracking = 0

        current_measurement = np.array([self.loc[0], self.loc[1]], dtype=np.float)
        self.filtered_state_means, self.filtered_state_covariances = self.kf.filter_update(self.filtered_state_means,
                                                                                           self.filtered_state_covariances,
                                                                                           current_measurement)

    def update_with_prediction(self):
        self.age += 1
        self.new_object = False
        self.updated = False
        self.predicted = True

        # self.loc[0] += self.filtered_state_means[2] * self.delta_t
        # self.loc[1] += self.filtered_state_means[3] * self.delta_t
        self.loc[0] += self.velo[0] * self.delta_t
        self.loc[1] += self.velo[1] * self.delta_t

        self.filtered_state_means, self.filtered_state_covariances = self.kf.filter_update(self.filtered_state_means,
                                                                                           self.filtered_state_covariances,
                                                                                           self.loc[:2])

        # self.real_path.append(None)
        self.real_path.append(self.loc[:2])
        self.predicted_path.append(self.loc[:2])

    def dimension_filter(self):
        self.dims = copy.deepcopy(self.dims_history[-1])
        if self.loc[0] <= self.no_dims_filter_range:
            return

        length_dims_track_range = min(len(self.dims_history), 2 * self.max_dims_track_range)
        width_dims_track_range = min(len(self.dims_history), self.max_dims_track_range)
        if len(self.dims_history) >= 3:
            median_dim_l = np.median([dims[0] for dims in self.dims_history[-length_dims_track_range:]])
            median_dim_w = np.median([dims[1] for dims in self.dims_history[-width_dims_track_range:]])
            if abs(median_dim_l - self.dims[0]) / self.dims[0] > self.length_diff_ratio_thres:
                self.dims[0] = median_dim_l
                self.dims[1] = median_dim_w

    def angle_filter(self):
        pass

    def __repr__(self):
        object_status = 'Unknown'
        if self.updated:
            object_status = 'Updated'
        if self.predicted:
            object_status = 'Predicted'
        object_info = f"[Object {self.track_id}] location: {self.loc} | size: {self.dims} \
                        | status: {object_status} | age: {self.age} | lose tracking: {self.lose_tracking}"
        return object_info


class TrackingManager(object):
    def __init__(self, config):
        self.tracker_list = []
        self.detected_object_list = []
        self.class_names = config.CLASS_NAMES
        self.track_id = 0

        self.dist_thres_longitudinal = 5
        self.dist_thres_lateral = 0.8
        self.start_tracking_score_thres = 0.3
        self.lose_tracking_thres = 3
        self.age_thres_of_object = 3
        self.age_thres_for_prediction = 2
        self.side_range_limit = 16
        self.filter_missdetection_thres = 2

    def create_det_object_list(self, pred_dicts):
        det_boxes = pred_dicts[0]['pred_boxes'].cpu().detach().numpy()
        det_scores = pred_dicts[0]['pred_scores'].cpu().detach().numpy()
        det_labels = pred_dicts[0]['pred_labels'].cpu().detach().numpy()
        num_det_objects = det_boxes.shape[0]

        self.detected_object_list = []
        self.detected_object_boxes = []
        for det_object_idx in range(num_det_objects):
            if det_boxes[det_object_idx, 1] > self.side_range_limit or \
                    det_boxes[det_object_idx, 1] < -self.side_range_limit:
                continue
            det_object = DetectedObject(det_boxes[det_object_idx, :], det_labels[det_object_idx],
                                        det_scores[det_object_idx])
            self.detected_object_boxes.append(det_boxes[det_object_idx, :])
            self.detected_object_list.append(det_object)
        self.detected_object_boxes = np.array(self.detected_object_boxes, dtype=np.float32)

    def get_tracked_object_boxes(self):
        if not len(self.tracker_list):
            return np.array([[0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)  # DEBUG

        tracked_boxes = []
        for tracked_object in self.tracker_list:
            # if tracked_object.age < self.age_thres_of_object:
            #     continue
            obj_loc = tracked_object.loc
            obj_dims = tracked_object.dims
            obj_rotz = tracked_object.rotz
            tracked_box = np.concatenate((obj_loc, obj_dims, obj_rotz[..., np.newaxis]))
            tracked_boxes.append(tracked_box)

        return np.array(tracked_boxes, dtype=np.float32)

    def get_tracked_objects(self):
        info_tracked_objects = {'object_ids': [], 'object_types': [], 'pred_boxes': []}

        for tracked_object in self.tracker_list:
            # if tracked_object.age < self.age_thres_of_object:
            #     continue
            obj_loc = tracked_object.loc
            obj_dims = tracked_object.dims
            obj_rotz = tracked_object.rotz
            tracked_box = np.concatenate((obj_loc, obj_dims, obj_rotz[..., np.newaxis]))
            info_tracked_objects['pred_boxes'].append(tracked_box)
            info_tracked_objects['object_ids'].append(tracked_object.track_id)
            info_tracked_objects['object_types'].append(tracked_object.type)

        info_tracked_objects['pred_boxes'] = np.array(info_tracked_objects['pred_boxes'], dtype=np.float)
        info_tracked_objects['object_ids'] = np.array(info_tracked_objects['object_ids'], dtype=np.int)

        return info_tracked_objects

    def update_tracking(self, pred_dicts):
        # Parse the prediction results
        self.create_det_object_list(pred_dicts)

        if self.detected_object_list is None:
            # Update with prediction
            for tracked_object in self.tracker_list:
                if tracked_object.age >= self.age_thres_for_prediction:
                    tracked_object.update_with_prediction()
                tracked_object.lose_tracking += 1

            return self.get_tracked_objects()

        # Update the tracked objects with detection results
        # for tracked_object in self.tracker_list:
        #     tracked_object.updated = False
        #     min_dist_lateral = 10
        #     closest_det_object = None
        #
        #     for det_object in self.detected_object_list[:]:
        #         objects_dist_lateral = abs(tracked_object.loc[1] - det_object.loc[1])
        #         if objects_dist_lateral < min_dist_lateral:
        #             min_dist_lateral = objects_dist_lateral
        #             closest_det_object = det_object
        #
        #     # Can't find a close object
        #     if not closest_det_object:
        #         continue
        #
        #     closest_objects_dist_longitudinal = abs(tracked_object.loc[0] - closest_det_object.loc[0])
        #     if closest_objects_dist_longitudinal <= self.dist_thres_longitudinal:
        #         tracked_object.update(closest_det_object)
        #         self.detected_object_list.remove(closest_det_object)

        bev_iou = boxes3d_nearest_bev_iou(torch.from_numpy(self.get_tracked_object_boxes()[:, 0:7]),
                                          torch.from_numpy(self.detected_object_boxes[:, 0:7])).numpy()
        # sorted_bev_iou = np.sort(bev_iou, axis=-1)
        sort_indices = bev_iou.argsort(axis=-1)
        associated_flag = [False for _ in self.detected_object_list]
        for idx in range(len(self.tracker_list)):
            tracked_object = self.tracker_list[idx]
            tracked_object.updated = False
            max_iou_index = sort_indices[idx, -1]
            max_iou = bev_iou[idx, max_iou_index]
            max_iou_detection = self.detected_object_list[max_iou_index]
            second_max_iou_index = sort_indices[idx, -2] if sort_indices.shape[1] > 1 else max_iou_index
            second_max_iou = bev_iou[idx, second_max_iou_index]
            second_max_iou_detection = self.detected_object_list[second_max_iou_index]
            if max_iou < 0.1:
                continue
            elif max_iou > second_max_iou * 2:
                tracked_object.update(max_iou_detection)
                associated_flag[max_iou_index] = True
            else:
                if abs(max_iou_detection.loc[1] - tracked_object.loc[1]) <= abs(second_max_iou_detection.loc[1] - tracked_object.loc[1]):
                    closest_det_object = max_iou_detection
                    closest_det_index = max_iou_index
                else:
                    closest_det_object = second_max_iou_detection
                    closest_det_index = second_max_iou_index
                tracked_object.update(closest_det_object)
                associated_flag[closest_det_index] = True

        # Update the not updated tracked objects with prediction
        for tracked_object in self.tracker_list:
            if tracked_object.updated:
                continue
            if tracked_object.age >= self.age_thres_for_prediction:
                tracked_object.update_with_prediction()
            tracked_object.lose_tracking += 1

        # Create new trackers for not correlated detected objects
        for idx, det_object in enumerate(self.detected_object_list):
            if associated_flag[idx]:
                continue
            if det_object.score >= self.start_tracking_score_thres:
                new_tracker = ObjectTracker(det_object, self.track_id)
                self.tracker_list.append(new_tracker)
                self.track_id += 1

        # Remove untracked objects
        for tracked_object in self.tracker_list:
            if tracked_object.updated:
                continue
            if  tracked_object.new_object:
                tracked_object.new_object = False
                continue
            if tracked_object.lose_tracking >= self.lose_tracking_thres or \
                    tracked_object.age - tracked_object.lose_tracking <= self.filter_missdetection_thres:
                self.tracker_list.remove(tracked_object)

        return self.get_tracked_objects()
