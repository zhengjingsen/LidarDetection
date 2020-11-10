import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2

if __name__ == "__main__":
    bag_file = '/home/ethan/Workspace/unified_lidar2.bag'
    bag = rosbag.Bag(bag_file, 'r')
    # info = bag.get_type_and_topic_info()
    # print(info[1].keys())

    bag_data_lidar_unified = bag.read_messages(topics='/unified/lidar_points')

    frame_idx = 0
    for topic, msg_lidar_unified, _ in bag_data_lidar_unified:
        lidar_pts_unified = pc2.read_points(msg_lidar_unified)
        lidar_pts_unified = np.array(list(lidar_pts_unified))[:, :4]
        lidar_pts_unified[:, 3] = 1
        tf_matrix_lidar_unified = np.array([9.7664633748321206e-01, 2.3700882187393947e-02,
                                            2.1354233630479907e-01, 4.4884194774399999e+00,
                                            -2.7655994825909448e-02, 9.9949637395898994e-01,
                                            1.5552890021958202e-02, -1.9965142422800002e-02,
                                            -2.1306615826035888e-01, -2.1095415271440141e-02,
                                            9.7680992196315319e-01, 2.8337476145100000e+00,
                                            0., 0., 0., 1.]).reshape([4, 4])

        lidar_pts_unified = np.matmul(lidar_pts_unified, tf_matrix_lidar_unified.T)
        lidar_pts_unified = lidar_pts_unified[:, :3]
        filename = "/home/ethan/Workspace/dataset/test_scene/%06d.bin" % frame_idx
        with open(filename, 'wb') as f:
            lidar_pts_unified.tofile(f)
            print("test_scene %06d saved in %s." % (frame_idx, filename))
        # np.savetxt("/home/ethan/Workspace/dataset/test_scene/%06d.txt" % frame_idx, lidar_pts_unified)
        frame_idx += 1

    bag.close()
