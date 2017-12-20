# HOW TO USE
# go to the location which you wnat to save evaluaion result and 
# $ nohup sh ~/ACOA/acoa/tool/eval/ev_nohup_Exp_mobile.sh Shoe mobilenet_v1 mobile_train &
# $ nohup sh {this script} {tf class folder} {train_dir path} &
# and you can monitor with this command
# $ tail -f nohup.out
ACOASET=$HOME/ACOA
DATASET_DIR=${ACOASET}/dataset/$1
TRAIN_DIR=${ACOASET}/Experiment/$3
CHECKPOINT_PATH=${ACOASET}/Experiment/nets/vgg_16.ckpt
SLIM_PATH=$HOME/ACOA/acoa/slim_Exp
EVAL_DIR=${ACOASET}/Experiment/mobile_eval
python ${SLIM_PATH}/eval_image_classifier.py \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=acoa \
    --dataset_split_name=validation \
    --model_name=$2 \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --batch_size 16 \
    --eval_interval_secs 60
