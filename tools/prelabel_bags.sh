#!/bin/bash

labels=(20210115T150849_j7-feidian_11_175to195.bag)
bags_path="../data/plusai/temp/"
result_path="../data/plusai/inference_result/"

for bag in ${labels[*]}
do
if [ ! -f temp/$bag ];then
    aws s3 cp s3://labeling/benchmark/obstacle_tracking/data/$bag ${bags_path} --endpoint-url=http://172.16.0.3
fi
done


# for label in ${labels[*]}
# do
# #if [ ! -f label/${label}.json ];then
#     aws s3 cp s3://labeling/benchmark/obstacle_tracking/label/${label}.json label/ --endpoint-url=http://172.16.0.3
# #fi
# done

python inference_bag2json.py \
  --cfg_file cfgs/livox_models/pv_rcnn_multiframe.yaml \
  --ckpt ../checkpoints/livox_models/pv_rcnn_multiframe/checkpoint_epoch_80.pth \
  --bag_file ${bags_path} \
  --save_path ${result_path}