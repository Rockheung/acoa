ACOASET=/home/rockheung/acoa_dataset
DATASET_DIR=${ACOASET}/dataset
TRAIN_DIR=${ACOASET}/train_full
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
SLIM_PATH=/home/rockheung/Downloads/models/research/slim
python ${SLIM_PATH}/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=apparel \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --save_interval_secs=900 \
    --save_summaries_secs=60 \
    --log_every_n_steps=10
