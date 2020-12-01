'''
@Time       : 
@Author     : Jingsen Zheng
@File       : calibration_plusai
@Brief      : 
'''

import os
import cv2
import numpy as np

def load_lidar_calib(car, calib_name, calib_date, calib_db_path):
    calib_file_name = car + '_' + calib_date + '_' + calib_name + '.yml'
    calib_file_name = os.path.join(calib_db_path, calib_file_name)
    if not os.path.isfile(calib_file_name):
        print('Calib file {} not found!'.format(calib_file_name))
        raise FileNotFoundError

    calib = cv2.FileStorage(calib_file_name, cv2.FILE_STORAGE_READ)
    Tr_lidar_to_imu = calib.getNode('Tr_lidar_to_imu')
    return Tr_lidar_to_imu.mat().astype(np.float32)