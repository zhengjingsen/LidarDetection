import cv2
import numpy as np


def plot_feature_map(features, channel=None):
    """Visualize the feature maps in neural network.

    :param features: pytorch tensor, [C, H, W]
    :param channel: select which channel to be visualized, default = None means all channels in avg
    :return: None
    """

    # Convert torch tensor to numpy array
    features = features.cpu().detach().numpy()
    assert len(features.shape) == 3

    if channel is None:
        feature_map = np.mean(features, axis=1)
    else:
        feature_map = features[channel, ...]

    cv2.imwrite("feature_map.jpg", feature_map)
    print("Feature map saved.")

# plot function for multi-frame visualization
def plot_multiframe_boxes(points, boxes, bev_range,
                          gt_boxes=None, resolution=0.1,
                          scores=None, labels=None,
                          info=None):
    """ Visualize the boxes.

    :param points: lidar points, [N, 3~5]
    :param boxes: [N, stack_frame_size, [x, y, z, l, w, h, r]]
    :param gt_boxes: gt boxes, [N, stack_frame_size, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    :return: color image
    """

    stack_frame_size = boxes.shape[1]
    # drop out out range points
    points = points[(points[:, 0] > bev_range[0]) & (points[:, 0] < bev_range[3]) &
                    (points[:, 1] > bev_range[1]) & (points[:, 1] < bev_range[4]) &
                    (points[:, 2] > bev_range[2]) & (points[:, 2] < bev_range[5])]

    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / resolution + 1)
    pixels_y = int((bev_range[4] - bev_range[1]) / resolution + 1)
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)

    # Plot the point cloud
    loc_x = ((bev_range[3] - points[:, 0]) / resolution).astype(int)
    loc_y = ((bev_range[4] - points[:, 1]) / resolution).astype(int)
    if points.shape[1] == 5:
        color = np.ones((points.shape[0], 3), dtype=np.uint8) * 32
        # fraction = points[:, -1] / stack_frame_size
        # color[:, 1] = fraction * 255
        # color[:, 2] = (1 - fraction) * 255
        color[points[:, -1] == 0] = [180, 0, 0]
        color[points[:, -1] == 1] = [0, 180, 0]
        color[points[:, -1] == 2] = [0, 0, 180]
        canvas[loc_x, loc_y] = color
    else:
        canvas[loc_x, loc_y] = [0, 255, 255]

    def plot_boxes(boxes, color, scores, labels, thickness=1):
        assert boxes.shape[1] == len(color)
        for idx in range(boxes.shape[1]):
            cur_color = color[idx]
            for i, box in enumerate(boxes):
                box2d = get_corners_2d(box[idx])
                box2d[:, 0] = bev_range[3] - box2d[:, 0]
                box2d[:, 1] = bev_range[4] - box2d[:, 1]
                # Plot box
                cv2.line(canvas, (int(box2d[0, 1] / resolution), int(box2d[0, 0] / resolution)),
                         (int(box2d[1, 1] / resolution), int(box2d[1, 0] / resolution)), cur_color, thickness)
                cv2.line(canvas, (int(box2d[1, 1] / resolution), int(box2d[1, 0] / resolution)),
                         (int(box2d[2, 1] / resolution), int(box2d[2, 0] / resolution)), cur_color, thickness)
                cv2.line(canvas, (int(box2d[2, 1] / resolution), int(box2d[2, 0] / resolution)),
                         (int(box2d[3, 1] / resolution), int(box2d[3, 0] / resolution)), cur_color, thickness)
                cv2.line(canvas, (int(box2d[3, 1] / resolution), int(box2d[3, 0] / resolution)),
                         (int(box2d[0, 1] / resolution), int(box2d[0, 0] / resolution)), cur_color, thickness)
                # Plot heading
                heading_points = rot_line_90(box2d[0], box2d[1])
                cv2.line(canvas, (int(heading_points[0, 1] / resolution), int(heading_points[0, 0] / resolution)),
                         (int(heading_points[1, 1] / resolution), int(heading_points[1, 0] / resolution)), cur_color, thickness)
                if scores is not None and labels is not None and idx == stack_frame_size // 2:
                    cv2.putText(canvas, str(scores[i]) + ', ' + str(labels[i]),
                                (int(box2d[0, 1] / resolution), int(box2d[0, 0] / resolution)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, color=cur_color, thickness=2)

    # Plot the detect boxes
    if boxes is not None:
        dt_colors = [[255, 100, 100], [100, 255, 100], [100, 100, 255]]
        plot_boxes(boxes, dt_colors, scores, labels, 1)

    # Plot the gt boxes
    gt_colors = [[128, 0, 0], [0, 128, 0], [0, 0, 128]]
    if gt_boxes is not None:
        plot_boxes(gt_boxes, gt_colors, None, None, 1)

    if info is not None:
        cv2.putText(canvas, info, (10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=gt_colors[1], thickness=1)

    return canvas


def plot_gt_boxes(points, gt_boxes, bev_range, name=None, ret=False,
                  resolution=0.1):
    """ Visualize the ground truth boxes.
    :param points: lidar points, [N, 3]
    :param gt_boxes: gt boxes, [N, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    :return: None
    """

    # Configure the resolution
    steps = resolution
    
    points = points[(points[:, 0] > bev_range[0]) & (points[:, 0] < bev_range[3]) &
                    (points[:, 1] > bev_range[1]) & (points[:, 1] < bev_range[4]) &
                    (points[:, 2] > bev_range[2]) & (points[:, 2] < bev_range[5])]
    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / steps) + 1
    pixels_y = int((bev_range[4] - bev_range[1]) / steps) + 1
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
    canvas.fill(0)

    # Plot the point cloud
    loc_x = ((points[:, 0] - bev_range[0]) / steps).astype(int)
    loc_y = ((points[:, 1] - bev_range[1]) / steps).astype(int)
    canvas[loc_x, loc_y] = [0, 255, 255]

    # for idx in range(points.shape[0]):
    #     time0 = time()
    #     point = points[idx, :]
    #     time1 = time()
    #     print("checkpoint1:", time1 - time0)
    #     if bev_range[0] <= point[0] <= bev_range[3] and \
    #        bev_range[1] <= point[1] <= bev_range[4] and \
    #        bev_range[2] <= point[2] <= bev_range[5]:
    #         time2 = time()
    #         print("checkpoint2:", time2 - time1)
    #         loc_x = int((point[0] - bev_range[0]) / steps)
    #         loc_y = int((point[1] - bev_range[1]) / steps)
    #         time3 = time()
    #         print("checkpoint3:", time3 - time2)
    #         canvas[loc_x, loc_y] = [0, 255, 255]
    #         time4 = time()
    #         print("checkpoint4:", time4 - time3)

    # Plot the gt boxes
    gt_color = (0, 255, 0)
    for box in gt_boxes:
        box2d = get_corners_2d(box)
        box2d[:, 0] -= bev_range[0]
        box2d[:, 1] -= bev_range[1]
        # Plot box
        cv2.line(canvas, (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)),
                 (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)),
                 (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)),
                 (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)),
                 (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)), gt_color, 3)
        # Plot heading
        heading_points = rot_line_90(box2d[0], box2d[1])
        cv2.line(canvas, (int(heading_points[0, 1] / steps), int(heading_points[0, 0] / steps)),
                 (int(heading_points[1, 1] / steps), int(heading_points[1, 0] / steps)), gt_color, 3)

    # Rotate the canvas to correct direction
    # canvas = cv2.rotate(canvas, cv2.cv2.ROTATE_90_CLOCKWISE)
    canvas = cv2.flip(canvas, 0)
    canvas = cv2.flip(canvas, 1)

    cv2.putText(canvas, "Green: Ground Truth", (10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=gt_color, thickness=1)

    if ret:
        return canvas
    else:
        cv2.imwrite("gt_box_%s.jpg" % name, canvas)
        print("Image %s saved." % name)


def plot_gt_det_cmp(points, gt_boxes, det_boxes, bev_range, name=None):
    """Visualize all gt boxes and det boxes for comparison.

    :param points: lidar points, [N, 3]
    :param gt_boxes: gt boxes, [N, [x, y, z, l, w, h, r]]
    :param det_boxes: det boxes, [N, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    :return:
    """

    # Configure the resolution
    steps = 0.1

    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / steps)
    pixels_y = int((bev_range[4] - bev_range[1]) / steps)
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
    canvas.fill(0)

    # Plot the point cloud
    loc_x = ((points[:, 0] - bev_range[0]) / steps).astype(int)
    loc_y = ((points[:, 1] - bev_range[1]) / steps).astype(int)
    canvas[loc_x, loc_y] = [0, 255, 255]

    # Plot the gt boxes
    gt_color = (0, 255, 0)
    for box in gt_boxes:
        box2d = get_corners_2d(box)
        box2d[:, 0] -= bev_range[0]
        box2d[:, 1] -= bev_range[1]
        # Plot box
        cv2.line(canvas, (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)),
                 (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)),
                 (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)),
                 (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)), gt_color, 3)
        cv2.line(canvas, (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)),
                 (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)), gt_color, 3)
        # Plot heading
        heading_points = rot_line_90(box2d[0], box2d[1])
        cv2.line(canvas, (int(heading_points[0, 1] / steps), int(heading_points[0, 0] / steps)),
                 (int(heading_points[1, 1] / steps), int(heading_points[1, 0] / steps)), gt_color, 3)

    # Plot the det boxes
    det_color = (0, 0, 255)
    for box in det_boxes:
        box2d = get_corners_2d(box)
        box2d[:, 0] -= bev_range[0]
        box2d[:, 1] -= bev_range[1]
        # Plot box
        cv2.line(canvas, (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)),
                 (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)), det_color, 3)
        cv2.line(canvas, (int(box2d[1, 1] / steps), int(box2d[1, 0] / steps)),
                 (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)), det_color, 3)
        cv2.line(canvas, (int(box2d[2, 1] / steps), int(box2d[2, 0] / steps)),
                 (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)), det_color, 3)
        cv2.line(canvas, (int(box2d[3, 1] / steps), int(box2d[3, 0] / steps)),
                 (int(box2d[0, 1] / steps), int(box2d[0, 0] / steps)), det_color, 3)
        # Plot heading
        heading_points = rot_line_90(box2d[0], box2d[1])
        cv2.line(canvas, (int(heading_points[0, 1] / steps), int(heading_points[0, 0] / steps)),
                 (int(heading_points[1, 1] / steps), int(heading_points[1, 0] / steps)), det_color, 3)

    # Rotate the canvas to correct direction
    canvas = cv2.flip(canvas, 0)
    canvas = cv2.flip(canvas, 1)

    cv2.putText(canvas, "Green: Ground Truth", (10, 35), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=gt_color, thickness=1)
    cv2.putText(canvas, "Red: Detection", (10, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=det_color, thickness=1)

    cv2.imwrite("leading_boxes_%s.jpg" % name, canvas)
    print("Image %s saved." % name)


def rotz(t):
    """Rotation about the z-axis.

    :param t: rotation angle
    :return: rotation matrix
    """

    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s,  c]])


def rot_line_90(point1, point2):
    """Rotate a line around its center for 90 degree in BEV plane.

    :param point1: End point1, [x, y]
    :param point2: End point2, [x, y]
    :return: rot_line
    """

    center_x = (point1[0] + point2[0]) / 2
    center_y = (point1[1] + point2[1]) / 2
    rot_point1 = np.dot(rotz(np.pi / 2), [point1[0] - center_x, point1[1] - center_y])
    rot_point2 = np.dot(rotz(np.pi / 2), [point2[0] - center_x, point2[1] - center_y])
    rot_point1 += [center_x, center_y]
    rot_point2 += [center_x, center_y]

    return np.array([rot_point1, rot_point2])


def get_corners_2d(box):
    """Takes an bounding box and calculate the 2D corners in BEV plane.

    0 --- 1
    |     |        x
    |     |        ^
    |     |        |
    3 --- 2  y <---o

    :param box: 3D bounding box, [x, y, z, l, w, h, r]
    :return: corners_2d: (4,2) array in left image coord.
    """

    # compute rotational matrix around yaw axis
    rz = box[6]
    R = rotz(rz)

    # 2d bounding box dimensions
    l = box[3]
    w = box[4]

    # 2d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_2d = np.dot(R, np.vstack([x_corners, y_corners]))
    corners_2d[0, :] = corners_2d[0, :] + box[0]
    corners_2d[1, :] = corners_2d[1, :] + box[1]

    return corners_2d.T


if __name__ == "__main__":
    bv_range = [0, -25.0, 50.0, 25.0]
    random_points = np.random.random((1000, 3)) * 50
    random_points[:, 1] -= 25.0
    gt_boxes = np.array([[10.0, 1.0, 0.0, 4.1, 1.7, 1.5, 0],
                         [3.0, -5.0, 0.0, 3.8, 1.67, 1.4, np.pi/4],
                         [20.0, 13.0, 0.0, 8.1, 2.7, 4.5, np.pi/2]])
    plot_gt_boxes(random_points, gt_boxes, bv_range)

    # ================================================================
    det_boxes = np.array([[10.2, 1.05, 0.0, 4.15, 1.68, 1.5, 0.01],
                         [3.05, -5.04, 0.0, 3.84, 1.69, 1.4, np.pi / 4 - 0.04],
                         [20.3, 13.2, 0.0, 8.3, 2.69, 4.5, np.pi / 2 + 0.08]])
    plot_gt_det_cmp(random_points, gt_boxes, det_boxes, bv_range)
