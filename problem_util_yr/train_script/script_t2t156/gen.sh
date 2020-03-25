#!/bin/bash
source configuration.sh

echo 'gen'
pwd
ls
# delet old data
rm $DATA_PATH/*problem*

echo 'generate data...'
echo $PROBLEM 
echo $USR_DIR
echo $DATA_PATH




t2t-datagen --data_dir=$DATA_PATH --problem=$PROBLEM --t2t_usr_dir=$USR_DIR > source_target.txt



