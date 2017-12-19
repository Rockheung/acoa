# HOW TO USE
# go to the location which you want to save train weight and 
# $ nohup sh ~/ACOA/acoa/tool/train/tr_nohup_mobile.sh Shoe mobilenet_v1 mobile_train
# $ nohup sh {this script} {tf class folder}
# and you can monitor with this command 
# $ tail -f nohup.out
#    --trainable_scopes=vgg_16/fc9 \
ACOASET=$HOME/ACOA
DATASET_DIR=${ACOASET}/dataset/$1
TRAIN_DIR=${ACOASET}/Experiment/$3
CHECKPOINT_PATH=${ACOASET}/Experiment/nets/mobilenet_v1/mobilenet_v1_1.0_224.ckpt
SLIM_PATH=$HOME/ACOA/acoa/slim_Exp
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
    --learning_rate_decay_type='exponential' \
    --learning_rate_decay_factor=0.999 \
    --weight_decay=0.00004 \
    --log_every_n_steps=10 \
    --optimizer=adam \
    --checkpoint_exclude_scopes=MobilenetV1/Logits/Conv2d_1c_1x1 \
    --max_number_of_steps 1000000
