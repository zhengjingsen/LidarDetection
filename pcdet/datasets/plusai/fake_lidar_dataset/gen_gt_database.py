import pickle
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np
from autolab_core import RigidTransform

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


LABELS = [
    "None",
    "Car",
    "Truck",
    "Bus",
    "Pedestrian",
    "Bike",
    "Moto",
    "Fence",
    "Van",
    "Animal",
    "Cone",
    "RoadBlock",
    "Generic",
    "Unknown",
    "Unknown",
    "Unknown"
]


class Object3D(object):
    def __init__(self, lidar_label):
        # extract label, truncation, occlusion
        self.type = LABELS[lidar_label['attribute']['label_type']]  # 'Car', 'Pedestrian', ...
        self.truncation = 0.0  # truncated pixel ratio [0..1]
        self.occlusion = 0  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = -1  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = -10  # left
        self.ymin = -10  # top
        self.xmax = -10  # right
        self.ymax = -10  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = lidar_label['dimensions']['height']  # box height
        self.w = lidar_label['dimensions']['width']  # box width
        self.l = lidar_label['dimensions']['length']  # box length (in meters)
        self.t = (lidar_label['bottom_center']['x'],
                  lidar_label['bottom_center']['y'],
                  lidar_label['bottom_center']['z'])  # location (x,y,z)
        self.rz = lidar_label['yaw']  # yaw angle [-pi..pi]


class Calibration(object):
    def __init__(self,
                 calib_infos,
                 img_width=1000,
                 img_height=1000):
        self.car_heading = calib_infos["car_heading"]
        self.car_position = calib_infos["car_position"]
        self.lidar_heading = calib_infos["device_heading"]
        self.lidar_position = calib_infos["device_position"]
        self.lidar_info = calib_infos["pointcloud_info"]
        self.cams_info = calib_infos["images"]
        if "radar_points" in calib_infos:
            self.radar_pts = calib_infos["radar_points"]
        self.img_width = img_width
        self.img_height = img_height

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

    def lidar2world(self, pc):
        pc_offsets = np.array(self.lidar_info["offset"], dtype=np.float64)
        return pc + pc_offsets

    # 3d to 3d
    def world2car(self, pc):
        rotation_quaternion = np.asarray([self.car_heading['w'], self.car_heading['x'],
                                          self.car_heading['y'], self.car_heading['z']])
        translation = np.asarray([self.car_position['x'], self.car_position['y'], self.car_position['z']])
        T_qua2rota = RigidTransform(rotation_quaternion, translation)
        Trans = T_qua2rota.translation
        Rot = T_qua2rota.rotation
        Mat = np.zeros((4, 4), dtype=np.float64)
        Mat[0:3, 0:3] = Rot
        Mat[3, 3] = 1
        Mat[0:3, 3] = Trans
        return np.matmul(np.linalg.inv(Mat), self.cart2hom(pc).T).T[:, 0:3]

    def car2world(self, pc):
        rotation_quaternion = np.asarray([self.car_heading['w'], self.car_heading['x'],
                                          self.car_heading['y'], self.car_heading['z']])
        translation = np.asarray([self.car_position['x'], self.car_position['y'], self.car_position['z']])
        T_qua2rota = RigidTransform(rotation_quaternion, translation)
        Trans = T_qua2rota.translation
        Rot = T_qua2rota.rotation
        Mat = np.zeros((4, 4), dtype=np.float64)
        Mat[0:3, 0:3] = Rot
        Mat[3, 3] = 1
        Mat[0:3, 3] = Trans
        return np.matmul(Mat, self.cart2hom(pc).T).T[:, 0:3]

    def world2dev(self, pc):
        rotation_quaternion = np.asarray([self.lidar_heading['w'], self.lidar_heading['x'],
                                          self.lidar_heading['y'], self.lidar_heading['z']])
        translation = np.asarray([self.lidar_position['x'], self.lidar_position['y'], self.lidar_position['z']])
        T_qua2rota = RigidTransform(rotation_quaternion, translation)
        Trans = T_qua2rota.translation
        Rot = T_qua2rota.rotation
        Mat = np.zeros((4, 4), dtype=np.float64)
        Mat[0:3, 0:3] = Rot
        Mat[3, 3] = 1
        Mat[0:3, 3] = Trans
        return np.matmul(np.linalg.inv(Mat), self.cart2hom(pc).T).T[:, 0:3]

    def dev2world(self, pc):
        rotation_quaternion = np.asarray([self.lidar_heading['w'], self.lidar_heading['x'],
                                          self.lidar_heading['y'], self.lidar_heading['z']])
        translation = np.asarray([self.lidar_position['x'], self.lidar_position['y'], self.lidar_position['z']])
        T_qua2rota = RigidTransform(rotation_quaternion, translation)
        Trans = T_qua2rota.translation
        Rot = T_qua2rota.rotation
        Mat = np.zeros((4, 4), dtype=np.float64)
        Mat[0:3, 0:3] = Rot
        Mat[3, 3] = 1
        Mat[0:3, 3] = Trans
        return np.matmul(Mat, self.cart2hom(pc).T).T[:, 0:3]

    def world2cam(self, pc, cam_channel='front_left'):
        cur_cam_info = None
        for info in self.cams_info:
            if info["cam_id"] == cam_channel:
                cur_cam_info = info
                break
        if cur_cam_info is None:
            raise ValueError("Camera channel %s is not supported now!" % cam_channel)
        rotation_quaternion = np.asarray([cur_cam_info['heading']['w'], cur_cam_info['heading']['x'],
                                          cur_cam_info['heading']['y'], cur_cam_info['heading']['z']], dtype=np.float64)
        translation = np.asarray([cur_cam_info['position']['x'], cur_cam_info['position']['y'],
                                  cur_cam_info['position']['z']], dtype=np.float64)
        T_qua2rota = RigidTransform(rotation_quaternion, translation)
        Trans = T_qua2rota.translation
        Rot = T_qua2rota.rotation
        Mat = np.zeros((4, 4), dtype=np.float64)
        Mat[0:3, 0:3] = Rot
        Mat[3, 3] = 1
        Mat[0:3, 3] = Trans
        return np.matmul(np.linalg.inv(Mat), self.cart2hom(pc).T).T[:, 0:3]

    def cam2world(self, pc, cam_channel='front_left'):
        cur_cam_info = None
        for info in self.cams_info:
            if info["cam_id"] == cam_channel:
                cur_cam_info = info
                break
        if cur_cam_info is None:
            raise ValueError("Camera channel %s is not supported now!" % cam_channel)
        rotation_quaternion = np.asarray([cur_cam_info['heading']['w'], cur_cam_info['heading']['x'],
                                          cur_cam_info['heading']['y'], cur_cam_info['heading']['z']])
        translation = np.asarray(
            [cur_cam_info['position']['x'], cur_cam_info['position']['y'], cur_cam_info['position']['z']])
        T_qua2rota = RigidTransform(rotation_quaternion, translation)
        Trans = T_qua2rota.translation
        Rot = T_qua2rota.rotation
        Mat = np.zeros((4, 4), dtype=np.float64)
        Mat[0:3, 0:3] = Rot
        Mat[3, 3] = 1
        Mat[0:3, 3] = Trans
        return np.matmul(Mat, self.cart2hom(pc).T).T[:, 0:3]

    def lidar2dev(self, pc):
        pc_world = self.lidar2world(pc)
        pc_dev = self.world2dev(pc_world)
        return pc_dev

    def lidar2car(self, pc):
        pc_world = self.lidar2world(pc)
        pc_car = self.world2car(pc_world)
        return pc_car

    def lidar2cam(self, pc, cam_channel='front_left'):
        pc_world = self.lidar2world(pc)
        pc_cam = self.world2cam(pc_world,cam_channel=cam_channel)
        return pc_cam

    def car2cam(self, pc):
        mat = np.array([[-0.081515, -0.078592, 0.993569, 4.792509],
                        [-0.996604, -0.005272, -0.082181, 0.739551],
                        [0.011697, -0.996893, -0.077895, 1.927075],
                        [0.000000, 0.000000, 0.000000, 1.000000]], dtype=np.float64)
        return np.matmul(np.linalg.inv(mat), self.cart2hom(pc).T).T[:, 0:3]

    def label_lidar2dev(self, objects3d):
        boxes3d_dev = []
        objects3d_dev = []
        if objects3d is not None:
            for obj in objects3d:
                ctr_world = np.array(obj.t, dtype=np.float64)
                ctr_world[2] += obj.h / 2.0
                box_size = (obj.l, obj.w, obj.h)
                rotz = -obj.rz
                box3d = get_3d_box(box_size, rotz, ctr_world)
                box3d_dev = self.world2dev(box3d)
                boxes3d_dev.append(box3d_dev)
                obj.t = (box3d_dev[0, :] + box3d_dev[6, :]) / 2.0
                obj.t[2] -= obj.h / 2.0
                p0 = box3d_dev[0, 0:2]
                p3 = box3d_dev[3, 0:2]
                p3p0 = p0 - p3
                angle = np.arccos(p3p0[0] / np.sqrt(np.sum(np.square(p3p0))))
                if p3p0[0] > 0:
                    angle = -angle
                obj.rz = angle
                objects3d_dev.append(obj)
        else:
            print("[Warning]: Empty scene!")

        return boxes3d_dev, objects3d_dev

    # 3d to 2d
    def cam2pixel(self, pc, cam_channel='front_left'):
        cur_cam_info = None
        for info in self.cams_info:
            if info["cam_id"] == cam_channel:
                cur_cam_info = info
                break
        if cur_cam_info is None:
            raise ValueError("Camera channel %s is not supported now!" % cam_channel)
        k1 = np.array(cur_cam_info['k1']).astype(np.float64)
        k2 = np.array(cur_cam_info['k2']).astype(np.float64)
        k3 = np.array(cur_cam_info['k3']).astype(np.float64)
        p1 = np.array(cur_cam_info['p1']).astype(np.float64)
        p2 = np.array(cur_cam_info['p2']).astype(np.float64)
        fx = np.array(cur_cam_info['fx']).astype(np.float64)
        fy = np.array(cur_cam_info['fy']).astype(np.float64)
        cx = np.array(cur_cam_info['cx']).astype(np.float64)
        cy = np.array(cur_cam_info['cy']).astype(np.float64)
        pc = pc[pc[:, 2] > 0, :]  # filt the points behind the camera
        x_norm = pc[:, 0] / pc[:, 2]
        y_norm = pc[:, 1] / pc[:, 2]
        r2 = np.square(x_norm) + np.square(y_norm)
        r4 = r2 * r2
        r6 = r2 * r4
        x_dist = x_norm*(1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm*x_norm)
        y_dist = y_norm*(1 + k1*r2 + k2*r4 + k3*r6) + p1*(r2 + 2*y_norm*y_norm) + 2*p2*x_norm*y_norm
        x_pixel = fx * x_dist + cx
        y_pixel = fy * y_dist + cy
        pc_pixel = np.concatenate((np.reshape(x_pixel, (-1, 1)), np.reshape(y_pixel, (-1, 1))), axis=-1)  # (n, 2)
        pc_pixel_fov = pc_pixel[(pc_pixel[:, 0] >= 0) & (pc_pixel[:, 0] < self.img_width) &
                                (pc_pixel[:, 1] >= 0) & (pc_pixel[:, 1] < self.img_width), :]
        return pc_pixel, pc_pixel_fov

    def car2pixel(self, pc, cam_channel='front_left'):
        pc_world = self.car2world(pc)
        pc_cam = self.world2cam(pc_world, cam_channel=cam_channel)
        pc_pixel = self.cam2pixel(pc_cam, cam_channel=cam_channel)
        return pc_pixel

    def dev2pixel(self, pc, cam_channel='front_left'):
        pc_world = self.dev2world(pc)
        pc_cam = self.world2cam(pc_world, cam_channel=cam_channel)
        pc_pixel = self.cam2pixel(pc_cam, cam_channel=cam_channel)
        return pc_pixel

    def lidar2pixel(self, pc, cam_channel='front_left'):
        pc_world = self.lidar2world(pc)
        pc_cam = self.world2cam(pc_world)
        pc_pixel, pc_pixel_fov = self.cam2pixel(pc_cam, cam_channel=cam_channel)
        return pc_pixel, pc_pixel_fov

    def lidar2pixel_v2(self, pc, cam_channel='front_left'):
        pc_car = self.lidar2car(pc)
        pc_cam = self.car2cam(pc_car)
        pc_pixel, pc_pixel_fov = self.cam2pixel(pc_cam, cam_channel=cam_channel)
        return pc_pixel, pc_pixel_fov

    # Rotation angle
    def rot_world2car(self, rotz):
        w = self.car_heading['w']
        x = self.car_heading['x']
        y = self.car_heading['y']
        z = self.car_heading['z']

        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

        angle_r = r * 180 / math.pi
        angle_p = p * 180 / math.pi
        angle_y = y * 180 / math.pi

        rotz_car_coord = rotz - angle_y
        return rotz_car_coord

    def get_object_rotz(self, object_3d):
        ctr_world = np.array(object_3d.t, dtype=np.float64)
        ctr_world[2] += object_3d.h / 2.0
        box_size = (object_3d.l, object_3d.w, object_3d.h)
        rotz = - object_3d.rz
        corners_3d = get_3d_box(box_size, rotz, ctr_world)
        corners_3d_car_coord = self.world2car(corners_3d)
        p0 = corners_3d_car_coord[0, 0:2]
        p3 = corners_3d_car_coord[3, 0:2]
        p3p0 = p0 - p3
        angle = np.arccos(p3p0[0] / np.sqrt(np.sum(np.square(p3p0))))
        if p3p0[0] > 0:
            angle = -angle
        rz = angle

        return rz


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def rotz(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  s,  0],
                         [-s, c,  0],
                         [0,  0,  1]])

    R = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_lidar(file_path):
    lidar_points = np.fromfile(file_path, dtype=np.float32)
    lidar_points = np.reshape(lidar_points, (-1, 4))[:, :3]

    return lidar_points


def create_groundtruth_database(root_path=None, used_classes=None):
    database_save_path = Path(root_path) / 'gt_database'
    db_info_save_path = Path(root_path) / 'plusai_gt_dbinfos.pkl'

    database_save_path.mkdir(parents=True, exist_ok=True)
    all_db_infos = {}

    index_file = Path(root_path) / 'all.txt'
    with open(index_file, 'r') as f:
        dir_names = f.read().splitlines()

    frame_idx = 0
    for dir_name in tqdm(dir_names):
        root_json_file = Path(root_path) / dir_name / (dir_name + '.json')
        with open(root_json_file, 'r') as f:
            root_json = json.load(f)
        num_frames = len(root_json['labeling'])
        # num_frames = 1  # for DEBUG

        for k in tqdm(range(num_frames)):
            # Load object json file
            object_json_filename = root_json['labeling'][k]['filename']
            object_json_file = Path(root_path) / object_json_filename
            with open(object_json_file) as f:
                object_json = json.load(f)

            # Get calibration
            calib_info = {'car_heading': object_json['car_heading'],
                          'car_position': object_json['car_position'],
                          'device_heading': object_json['device_heading'],
                          'device_position': object_json['device_position'],
                          'pointcloud_info': object_json['pointcloud_info'],
                          'images': object_json['images']
                          }
            calib = Calibration(calib_info)

            # Get lidar point cloud
            lidar_path = Path(root_path) / object_json['pointcloud_info']['pointcloud_url']
            lidar_pts = get_lidar(lidar_path)
            lidar_pts_car_coord = calib.lidar2car(lidar_pts)

            # Get ground truth annotations
            gt_names = []
            gt_boxes = []
            for annos in root_json['labeling'][k]['annotations_3d']:
                obj = Object3D(annos)
                # Distinguish the categories of different size of Cars by length
                if obj.l < 5.0:
                    obj.type = 'Car'
                elif obj.l < 8.0:
                    obj.type = 'Truck'
                else:
                    obj.type = 'Tram'
                gt_names.append(obj.type)

                # Calculate location in vehicle coordinate
                location_world = np.array([obj.t], dtype=np.float)
                location_car_coord = np.squeeze(calib.world2car(location_world))
                dimensions = np.array([obj.l, obj.w, obj.h], dtype=np.float)
                rotz_car_coord = calib.get_object_rotz(obj)
                rotz_car_coord = np.array([rotz_car_coord], dtype=np.float)

                # Ground truth box: [x, y, z, l, w, h, rz]
                gt_box_car_coord = np.concatenate([location_car_coord, dimensions, rotz_car_coord], axis=0)
                gt_boxes.append(gt_box_car_coord)

            gt_boxes = np.array(gt_boxes, dtype=np.float)
            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(lidar_pts_car_coord), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%06d_%s_%d.bin' % (frame_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = lidar_pts_car_coord[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]

                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': frame_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]

            frame_idx += 1

    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
        print("Ground truth database info saved in %s." % db_info_save_path)


if __name__ == "__main__":
    create_groundtruth_database(root_path="/home/ethan/Workspace/dataset/PlusAI")

