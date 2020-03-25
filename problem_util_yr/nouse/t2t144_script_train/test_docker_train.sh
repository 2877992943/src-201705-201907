#!/bin/bash
source configuration.sh
#echo $ROOT_PATH
local_root_path='/home/yangrui/t2t144'
virtual_root_path=/home_yangrui
#export PYTHONPATH=$virtual_root_path/problem_util_yr # non exist in local machine
export PYTHONPATH=/custom_tool # tool /package needed to be called when train

problem_file=problem_languageModel_t2t144
this_problem_path=$virtual_root_path/$problem_file

docker run --runtime=nvidia -i -v $local_root_path:$virtual_root_path -v $local_root_path:$PYTHONPATH -e PYTHONPATH -w $virtual_root_path rxthinking:tensor2tensor_1.4.4 t2t-trainer \
  --t2t_usr_dir $this_problem_path/src \
  --data_dir=$this_problem_path/data \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$PARAMSET \
  --output_dir=$this_problem_path/model \
  --worker_gpu_memory_fraction 0.5


