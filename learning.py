'''
@Time       : 
@Author     : Jingsen Zheng
@File       : learning
@Brief      : 
'''

# print("result is {}".format(1 in [2], [1, 2, 3]))

import os
import sys
import numpy as np

if __name__ == '__main__':
    data_path = '/media/jingsen/data/Dataset/plusai/mot_dataset/'
    val_list_file = '/media/jingsen/data/Dataset/plusai/mot_dataset/ImageSets/val.txt'

    sample_id_list = [x.strip() for x in open(val_list_file).readlines()]
    cur_bag_name = ''
    with open(os.path.join(data_path, 'val_single_frame.txt'), 'w') as f:
        for sample in sample_id_list:
            bag_name, _, pointcloud_idx = sample.split('/')
            if cur_bag_name != bag_name:
                cur_bag_name = bag_name
                bag_frame_list = os.listdir(os.path.join(data_path, cur_bag_name, 'pointcloud'))
                bag_frame_list.sort()
            f.write(os.path.join(bag_name, 'pointcloud', bag_frame_list[int(pointcloud_idx[:-4]) + 1]) + '\n')
        f.close()


