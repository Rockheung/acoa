# HOW TO USE
# go to the location which you want to save train weight and 
# $ nohup sh ~/Downloads/acoa/tool/eval/ev_fc8_fc7_fc6_conv5_conv4.sh class_3 2 ~/acoa_dataset/weight/5_class_13/train_fc8_fc7_fc6_conv5_conv4/ &
# $ nohup sh {run this script} {tf class folder} {hierachy level}
# and you can monitor with this command
# $ tail -f nohup.out
ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset/$1
TRAIN_DIR=${PWD}
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
SLIM_PATH=$HOME/Downloads/acoa/slim_Exp
python ${SLIM_PATH}/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=acoa \
    --dataset_split_name=train \
    --model_name=$2 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=32 \
    --save_interval_secs=600 \
    --save_summaries_secs=60 \
    --learning_rate=0.01 \
    --learning_rate_decay_factor=0.95 \
    --weight_decay=0.00004 \
    --log_every_n_steps=10 \
    --optimizer=adam \
    --checkpoint_exclude_scopes=vgg_16/fc8 \
    --trainable_scopes=vgg_16/fc8 \
    --max_number_of_steps 100000
