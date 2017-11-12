ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset
TRAIN_DIR=${ACOASET}/train_fc8_fc7_fc6_conv5_conv4
EVAL_DIR=${ACOASET}/eval_fc8_fc7_fc6_conv5_conv4
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
CKPT=${ACOASET}/ckpt50
SLIM_PATH=$HOME/Downloads/acoa/slim
tensorboard \
    --logdir=${TRAIN_DIR}
