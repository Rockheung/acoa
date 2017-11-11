ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset/class_7
TRAIN_DIR=${ACOASET}/train_fc8
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
SLIM_PATH=$HOME/Downloads/acoa/slim_v2
python ${SLIM_PATH}/train_image_classifier_with_evaluation.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=acoa \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=32 \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --learning_rate_decay_factor=0.95 \
    --weight_decay=0.00004 \
    --log_every_n_steps=10 \
    --optimizer=adam \
    --checkpoint_exclude_scopes=vgg_16/fc8 \
    --trainable_scopes=vgg_16/fc8 \
    --per_process_gpu_memory_fraction=0.9 \
    --hierarchy_level=1
