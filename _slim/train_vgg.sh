export slim_path=/home/dan/ACOA/acoa/slim_addon
export exp_path=/home/dan/ACOA/Experiment


python train_image_classifier.py --train_dir=/home/dan/ACOA/Experiment/nets_fuck \
										   --dataset_name=acoa \
										   --dataset_split_name=train \
										   --dataset_dir=/home/dan/ACOA/tfrecord/ \
										   --model_name=vgg_16 \
										   --checkpoint_path=/home/dan/ACOA/Experiment/nets/vgg_16.ckpt \
										   --batch_size=32 \
										   --learning_rate_decay_factor=0.95 \
										   --weight_decay=0.00004 \
										   --log_every_n_steps=10 \
										   --optimizer=adam \
										   --save_summaries_secs=60 \
										   --save_interval_secs=60 \
										   --checkpoint_exclude_scopes=vgg_16/fc8