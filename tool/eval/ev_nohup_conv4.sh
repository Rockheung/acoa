# HOW TO USE
# go to the location which you wnat to save evaluaion result and 
# $ nohup sh ~/Downloads/acoa/tool/eval/ev_fc8_fc7_fc6_conv5_conv4.sh class_3 2 ~/acoa_dataset/weight/5_class_13/train_fc8_fc7_fc6_conv5_conv4/ &
# $ nohup sh {run this script} {tf class folder} {hierachy level} {train_dir path} &
# and you can monitor with this command
# $ tail -f nohup.out
ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset/$1
TRAIN_DIR=$2
EVAL_DIR=${PWD}
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
CKPT=${ACOASET}/train_fc8/ckpt-291715
#CKPT=${ACOASET}/train_full/ckpt55
SLIM_PATH=$HOME/Downloads/acoa/slim_origin
python ${SLIM_PATH}/eval_image_classifier.py \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=acoa \
    --dataset_split_name=validation \
    --model_name=vgg_16 \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --eval_interval_secs 60 \
    --batch_size 16
