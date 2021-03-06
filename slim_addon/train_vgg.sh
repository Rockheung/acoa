export slim_path=/home/dan/ACOA/acoa/slim_addon
export exp_path=/home/dan/ACOA/Experiment
# python $slim_path/train_image_classifier_addon.py --train_dir=/home/dan/ACOA/Experiment/addon_weight_3 \
# 										   --dataset_name=acoa \
# 										   --dataset_split_name=train \
# 										   --dataset_dir=/home/dan/ACOA/tfrecord/ \
# 										   --model_name=addonnet \
# 										   --checkpoint_path=/home/dan/ACOA/train_fc8_fc7_fc6_conv5_conv4/model.ckpt-125041 \
# 										   --batch_size=32 \
# 										   --learning_rate=0.01 \
# 										   --learning_rate_decay_type='exponential' \
# 										   --learning_rate_decay_factor=0.999 \
# 										   --end_learning_rate=0.002 \
# 										   --log_every_n_steps=10 \
# 										   --optimizer=adam \
# 										   --save_summaries_secs=60 \
# 										   --save_interval_secs=120 \
# 										   --hierarchy_level=2 \
# 										   --weight_decay=0.00004 \
# 										   --checkpoint_exclude_scopes=vgg_16/addon \
# 										   --trainable_scopes=vgg_16/addon

									   
python $slim_path/train_image_classifier_addon.py --train_dir=/home/dan/ACOA/Experiment/addon_weight_2 \
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
										   --save_interval_secs=60 \
										   --hierarchy_level=2 \
										   --checkpoint_exclude_scopes=vgg_16/addon \
    									   --trainable_scopes=vgg_16/addon