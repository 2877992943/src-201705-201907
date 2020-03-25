#!/bin/bash
echo 'Starting NLP service...'
RUN_PATH=`pwd`
# script dir
SCRIPT_PATH=`dirname $0`
source $SCRIPT_PATH/configuration.sh
echo '-----'
echo $PROBLEM

export CUDA_VISIBLE_DEVICES=2
 
# docker workdir is problem_xbs_chunk/, not current dir(/script )
python $SCRIPT_PATH/impl_batchinfer_ProblemDecoder_relation_xbs.py --model_dir=$MODEL_PATH --data_dir=$DATA_PATH --usr_dir=$SRC_PATH --inputFile=$SCRIPT_PATH/tmp --outputFile=$SCRIPT_PATH/tmp



 




