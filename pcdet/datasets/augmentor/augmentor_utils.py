import numpy as np

from ...utils import common_utils


def random_flip_along_x(gt_boxes, points, locations = None, rotations_y=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        locations: (N, S, 3)    S means stack frame size in multi-frame mode
        rotations_y: (N, S)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    if locations is not None and rotations_y is not None:
        if enable:
            locations[:, :, 1] = -locations[:, :, 1]
            rotations_y = -rotations_y
        return gt_boxes, points, locations, rotations_y

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, locations = None, rotations_y=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        locations: (N, S, 3)    S means stack frame size in multi-frame mode
        rotations_y: (N, S)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    if locations is not None and rotations_y is not None:
        if enable:
            locations[:, :, 0] = -locations[:, :, 0]
            rotations_y = -(rotations_y + np.pi)
        return gt_boxes, points, locations, rotations_y

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, locations = None, rotations_y=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
        locations: (N, S, 3)    S means stack frame size in multi-frame mode
        rotations_y: (N, S)
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    if locations is not None and rotations_y is not None:
        N = locations.shape[0]
        locations = common_utils.rotate_points_along_z(locations, np.array([noise_rotation] * N))
        rotations_y += noise_rotation
        return gt_boxes, points, locations, rotations_y

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, locations = None, rotations_y=None):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
        locations: (N, S, 3)    S means stack frame size in multi-frame mode
        rotations_y: (N, S)
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    if locations is not None and rotations_y is not None:
        locations *= noise_scale
        return gt_boxes, points, locations, rotations_y

    return gt_boxes, points
