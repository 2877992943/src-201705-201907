#!/bin/bash
### 先重启nvidia-docker_plugin
# ps -ef | grep nvidia-docker_plugin
# kill pid
# sudo nvidia-docker_plugin


# -v 宿主机的目录映射到容器内
# -w working dir


source configuration.sh
#echo $ROOT_PATH
local_root_path='/home/yangrui/t2t_problems'
virtual_root_path=/home_yangrui
#export PYTHONPATH=$virtual_root_path/problem_util_yr # non exist in local machine
export PYTHONPATH=/custom_tool # tool /package needed to be called when train

problem_file=problem_drug
this_problem_path=$virtual_root_path/$problem_file


sudo nvidia-docker run --name drug -i -v $local_root_path:$virtual_root_path -v $local_root_path:$PYTHONPATH -e PYTHONPATH=/home_yangrui -w $virtual_root_path dockerdist.bdmd.com/yangrui:tensor2tensor_1.4.4 t2t-trainer \
  --t2t_usr_dir $this_problem_path/src \
  --data_dir=$this_problem_path/data \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$PARAMSET \
  --output_dir=$this_problem_path/model \
  --worker_gpu_memory_fraction 0.3 \
  --hparams="batch_size=16384"