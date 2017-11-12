ACOASET=$HOME/acoa_dataset
DATASET_DIR=${ACOASET}/dataset
TRAIN_DIR=${ACOASET}/train_full_pre
EVAL_DIR=${ACOASET}/eval_full_pre
CHECKPOINT_PATH=${ACOASET}/checkpoints/vgg_16.ckpt
CKPT=${ACOASET}/ckpt50
SLIM_PATH=$HOME/Downloads/acoa/slim
tensorboard \
    --logdir=${TRAIN_DIR} \
    --host=0.0.0.0
