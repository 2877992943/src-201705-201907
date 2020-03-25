#!/bin/bash


cp ./transformer_layers_yr.py /usr/local/lib/python2.7/dist-packages/tensor2tensor/layers/transformer_layers.py



source configuration.sh
echo 'train...'
echo $PROBLEM

echo $USR_DIR
echo $MODEL
echo $PARAMSET
echo $MODEL_PATH
echo $DATA_PATH
#export CUDA_VISIBLE_DEVICES=0,1,2
t2t-trainer \
  --t2t_usr_dir $USR_DIR \
  --data_dir=$DATA_PATH \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$PARAMSET \
  --output_dir=$MODEL_PATH \
  --save_checkpoints_secs 1200 \
  --train_steps=1250000 \
  --hparams="batch_size=2000,max_length=1000,ffn_layer=bayes" \
  --min_eval_frequency=1000000 \
  --eval_steps=1000000 \
  --eval_throttle_seconds=360000 \
  --save_checkpoints_steps=5000 \
  --keep_checkpoint_max=5 \
  #--hparams="self_attention_type=quaternion_dot_product,ffn_layer=raw_dense_relu_dense"
  #--hparams="self_attention_type=quaternion_dot_product"
  #--worker_gpu=3
  #--worker_gpu_memory_fraction 0.95 \
  #--schedule=train \

