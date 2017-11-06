ACOASET=/home/rockheung/acoa_dataset
DATASET_DIR=${ACOASET}/dataset
TRAIN_DIR=${ACOASET}/train_full_pre
CHECKPOINT_PATH=${ACOASET}/train_fc8
SLIM_PATH=/home/rockheung/Downloads/models/research/slim
python ${SLIM_PATH}/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=apparel \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=32 \
    --learning_rate=0.001 \
    --save_interval_secs=900 \
    --save_summaries_secs=60 \
    --log_every_n_steps=10
