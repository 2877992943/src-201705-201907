#!/bin/bash


source configuration.sh

echo 'gen'
pwd
ls

echo $PROBLEM 
echo $USR_DIR
echo $DATA_PATH
echo $MODEL_PATH

TEST_PATH=$ROOT_PATH/script_testdecode_sh

BEAM_SIZE=1
ALPHA=0.6
extra_length=20
decode_length=20
return_beams=False
t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_PATH \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=transformer_ae_base \
  --output_dir=$MODEL_PATH \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA,extra_length=$extra_length,return_beams=$return_beams,write_beam_scores=True" \
  --decode_interactive=True
