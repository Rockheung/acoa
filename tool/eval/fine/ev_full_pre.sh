ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset/class_7
TRAIN_DIR=${ACOASET}/weight/11_class_25/train_full_pre
EVAL_DIR=${ACOASET}/weight/11_class_25/eval_full_pre
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
    --batch_size 8 \
    --eval_interval_secs 60 \
    --hierarchy_level 2 \
    --per_process_gpu_memory_fraction 0.9
