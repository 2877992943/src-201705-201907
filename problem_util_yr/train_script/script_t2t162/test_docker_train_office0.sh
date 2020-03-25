#!/bin/bash
source configuration.sh
#echo $ROOT_PATH
local_root_path='/home/yangrui'
virtual_root_path=/vroot
#export PYTHONPATH=$virtual_root_path/problem_util_yr # non exist in local machine
export PYTHONPATH=/custom_tool # tool /package needed to be called when train

problem_file=problem_symptom_relation
this_problem_path=$virtual_root_path/$problem_file
echo container
echo $this_problem_path

export CUDA_VISIBLE_DEVICES=1

docker rm relation

t2tuser=$this_problem_path/src
t2tdata=$this_problem_path/data
echo $t2tuser
echo $t2tdata
script_path=$this_problem_path/script_train
 
 

docker run --name relation --runtime=nvidia -i -v $local_root_path:$virtual_root_path -v $local_root_path:$PYTHONPATH -e PYTHONPATH=$virtual_root_path -w $script_path rxthinking:tensor2tensor_1.6.2 ./train_t2t162.sh


