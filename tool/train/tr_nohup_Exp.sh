# HOW TO USE
# go to the location which you want to save train weight and 
# $ nohup sh ~/Downloads/acoa/tool/train/tr_nohup_Exp.sh Bottom vgg_16_fc_xk_xk
# $ nohup sh {this script} {tf class folder}
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
    --learning_rate_decay_factor=0.95 \
    --weight_decay=0.00004 \
    --log_every_n_steps=10 \
    --optimizer=adam \
    --checkpoint_exclude_scopes=vgg_16/fc8,vgg_16/fc7,vgg_16/fc6 \
    --trainable_scopes=vgg_16/fc8,vgg_16/fc7,vgg_16/fc6,vgg_16/conv5,vgg_16/conv4 \
    --max_number_of_steps 100000
