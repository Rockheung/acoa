ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset/class_7
TRAIN_DIR=${ACOASET}/weight/1_coarse/train_fc8
EVAL_DIR=${ACOASET}/weight/1_coarse/eval_fc8
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
CKPT=${ACOASET}/train_fc8/ckpt-291715
#CKPT=${ACOASET}/train_full/ckpt55
SLIM_PATH=$HOME/Downloads/acoa/slim_v2
python ${SLIM_PATH}/eval_image_classifier.py \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=acoa \
    --dataset_split_name=validation \
    --model_name=vgg_16 \
    --checkpoint_path=${TRAIN_DIR} \
    --eval_dir=${EVAL_DIR} \
    --batch_size 16 \
    --eval_interval_secs 60 \
    --hierarchy_level 1 \
    --per_process_gpu_memory_fraction 0.9
