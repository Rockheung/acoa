export slim_path=/home/dan/ACOA/acoa/slim_addon
export exp_path=/home/dan/ACOA/Experiment

python eval_image_classifier.py \
    --dataset_dir=/home/dan/ACOA/tfrecord/ \
    --dataset_name=acoa \
    --dataset_split_name=validation \
    --model_name=vgg_16 \
    --checkpoint_path=/home/dan/ACOA/Experiment/nets_fuck \
    --eval_dir=$exp_path/eval_dir \
    --batch_size=64