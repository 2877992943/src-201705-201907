#!/bin/bash
ORIGINAL_PATH=`pwd`
SHELL_PATH=`dirname $0`
cd $SHELL_PATH/../
ROOT_PATH=`pwd`

echo $ROOT_PATH

MODEL_PATH=$ROOT_PATH/model
SRC_PATH=$ROOT_PATH/src
DATA_PATH=$ROOT_PATH/data
DICT_PATH=$DATA_PATH/dict
CORPUS_PATH=$DATA_PATH/corpus
TRAINING_PATH=$DATA_PATH/training

#   判断 $folder 是否存在
if [ ! -d $MODEL_PATH ]; then
  mkdir $MODEL_PATH
  echo "model dir not exist ,make it now"
fi



MODEL=transformer
PROBLEM='xbschunk_problem'
PARAMSET=transformer_base_single_gpu
#PARAMSET=transformer_base_single_gpu
USR_DIR=$SRC_PATH

echo $MODEL_PATH
echo $SRC_PATH
echo $DATA_PATH
echo $DICT_PATH
echo $CORPUS_PATH
echo $TRAINING_PATH
echo $PARAMSET
echo $USR_DIR
cd $ORIGINAL_PATH





