export slim_path=/home/dan/ACOA/acoa/slim_addon
export exp_path=/home/dan/ACOA/Experiment


python $slim_path/train_image_classifier_addon.py --train_dir=/home/dan/ACOA/Experiment/addon_weight \
										   --dataset_name=acoa \
										   --dataset_split_name=train \
										   --dataset_dir=/home/dan/ACOA/tfrecord/ \
										   --model_name=addonnet \
										   --checkpoint_path=/home/dan/ACOA/train_fc8_fc7_fc6_conv5_conv4/model.ckpt-125041 \
										   --batch_size=32 \
										   --learning_rate_decay_factor=0.95 \
										   --weight_decay=0.00004 \
										   --log_every_n_steps=10 \
										   --optimizer=adam \
										   --save_summaries_secs=60 \
										   --save_interval_secs=1800 \
										   --hierarchy_level=2 \
										   --checkpoint_exclude_scopes=vgg_16/addon \
    									   --trainable_scopes=vgg_16/addon