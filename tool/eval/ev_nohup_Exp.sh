# HOW TO USE
# go to the location which you wnat to save evaluaion result and 
# $ nohup sh ~/Downloads/acoa/tool/eval/ev_nohup_Exp.sh Bottom ~/acoa_dataset/weight/Bottom/Exp/train vgg_16_fc_xk_xk &
# $ nohup sh {this script} {tf class folder} {train_dir path} &
# and you can monitor with this command
# $ tail -f nohup.out
ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset/$1
TRAIN_DIR=$2
EVAL_DIR=${PWD}
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
CKPT=${ACOASET}/train_fc8/ckpt-291715
#CKPT=${ACOASET}/train_full/ckpt55
SLIM_PATH=$HOME/Downloads/acoa/slim_Exp
python ${SLIM_PATH}/eval_image_classifier.py \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=acoa \
    --dataset_split_name=validation \
    --model_name=$3 \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --batch_size 16 \
    --eval_interval_secs 60
