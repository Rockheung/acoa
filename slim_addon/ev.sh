export slim_path=/home/dan/ACOA/acoa/slim_addon
export exp_path=/home/dan/ACOA/Experiment

python $slim_path/eval_image_classifier.py \
    --dataset_dir=/home/dan/ACOA/tfrecord/ \
    --dataset_name=acoa \
    --dataset_split_name=validation \
    --model_name=addonnet \
    --checkpoint_path=/home/dan/ACOA/Experiment/addon_weight_2 \
    --eval_dir=$exp_path/eval_dir \
    --batch_size=64 \
    --hierarchy_level=2