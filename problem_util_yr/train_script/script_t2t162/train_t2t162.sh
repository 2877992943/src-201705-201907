#!/bin/bash
source configuration.sh
echo 'train...'
echo $PROBLEM 
echo $USR_DIR
echo $MODEL
echo $PARAMSET
echo $MODEL_PATH
echo $DATA_PATH
export CUDA_VISIBLE_DEVICES=1
t2t-trainer \
  --t2t_usr_dir $USR_DIR \
  --data_dir=$DATA_PATH \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$PARAMSET \
  --output_dir=$MODEL_PATH \
  --worker_gpu_memory_fraction 0.95 \
  --save_checkpoints_secs 1200 \
  --train_steps=250000 \
  --hparams="batch_size=8192,max_length=1000" \
  --worker_gpu_memory_fraction 0.95 \
  --schedule=train

