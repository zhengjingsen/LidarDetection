import pickle
import rosbag
import json
import math


if __name__ == "__main__":
    bag_file = '/home/ethan/Workspace/unified_lidar2.bag'
    pkl_file = '/home/ethan/Workspace/result.pkl'
    bag = rosbag.Bag(bag_file, 'r')
    # info = bag.get_type_and_topic_info()
    # print(info[1].keys())

    with open(pkl_file, 'rb') as f:
        det_result = pickle.load(f)

    json_dict = {'objects': []}
    frame_idx = 0
    object_id = 0

    odom_tmp = []
    for topic, msg, _ in bag.read_messages(topics=["/unified/lidar_points", "/navsat/odom"]):
        if topic == "/navsat/odom":
            odom_tmp = [msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]

        if topic == "/unified/lidar_points":
            timestr = '%0.9f' % msg.header.stamp.to_sec()
            timestr = timestr.split('.')

            for obj_idx in range(det_result[frame_idx][0]['pred_boxes'].shape[0]):
                object_info = {'bounds': [{'Tr_imu_to_world': {'qw': odom_tmp[0], 'qx': odom_tmp[1],
                                                               'qy': odom_tmp[2], 'qz': odom_tmp[3],
                                                               'x': odom_tmp[4], 'y': odom_tmp[5],
                                                               'z': odom_tmp[6]},
                                           'timestamp': int(timestr[0]),
                                           'timestamp_nano': int(timestr[1]),
                                           'velocity': {'x': 0, 'y': 0, 'z': 0}}
                                          ],
                               'size': {},
                               'uuid': str(object_id)
                               }
                obj_loc = det_result[frame_idx][0]['pred_boxes'][obj_idx, :3].cpu().detach().numpy().tolist()
                obj_dim = det_result[frame_idx][0]['pred_boxes'][obj_idx, 3:6].cpu().detach().numpy().tolist()
                obj_rz = det_result[frame_idx][0]['pred_boxes'][obj_idx, 6].cpu().detach().numpy().tolist()

                # Rotate the object center
                loc_x = obj_loc[0] * math.cos(-obj_rz) - obj_loc[1] * math.sin(-obj_rz)
                loc_y = obj_loc[0] * math.sin(-obj_rz) + obj_loc[1] * math.cos(-obj_rz)

                object_info['bounds'][0].update(
                    {'center': {'x': loc_x, 'y': loc_y, 'z': obj_loc[2]},
                     'direction': {'x': 0, 'y': 0, 'z': 0},
                     'heading': obj_rz,
                     'is_front_car': 0,
                     'position': {'x': obj_loc[0], 'y': obj_loc[1], 'z': obj_loc[2]},
                     'size': {'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]},
                     }
                )
                object_info['size'].update({'x': obj_dim[0], 'y': obj_dim[1], 'z': obj_dim[2]})
                json_dict['objects'].append(object_info)
                object_id += 1
            frame_idx += 1
            odom_tmp = []

    json_txt = json.dumps(json_dict, indent=4)
    with open('unified_lidar2.bag.json', 'w') as f:
        f.write(json_txt)
        print("JSON file saved.")
    bag.close()
