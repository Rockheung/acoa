ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset/class_7
TRAIN_DIR=${ACOASET}/train_fc8_fc7_fc6_conv5_conv4
EVAL_DIR=${ACOASET}/eval_fc8_fc7_fc6_conv5_conv4
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
    --hierarchy_level 1 \
    --per_process_gpu_memory_fraction 0.9
