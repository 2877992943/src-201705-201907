#!/bin/bash




local_root_path='/home/yangrui'
virtual_root_path=/vroot
#export PYTHONPATH=$virtual_root_path/problem_util_yr # non exist in local machine
#export PYTHONPATH=/custom_tool # tool /package needed to be called when train

problem_file=train_pursueAsk_graphnet_impl_Direction_v280_graphnet_only




this_problem_path=$virtual_root_path/$problem_file
tool_path=$this_problem_path

echo container
echo $this_problem_path


docker rm sym1



script_path=$this_problem_path/train_script
echo $script_path
 
 
##. office0
#docker run --name mul --runtime=nvidia -i -v $local_root_path:$virtual_root_path -v $local_root_path:$PYTHONPATH -e PYTHONPATH=$virtual_root_path -w $script_path rxthinking:tensor2tensor_1.8.0 ./train_t2t162.sh

#docker run --name mul --runtime=nvidia -i -v $local_root_path:$virtual_root_path -v $local_root_path:$PYTHONPATH -e PYTHONPATH=$virtual_root_path -w $script_path rxthinking:tf190_t2t166  ./train_t2t162.sh


## nfyy
#sudo nvidia-docker run --name xbs -i -v $local_root_path:$virtual_root_path -v $local_root_path:$PYTHONPATH -e PYTHONPATH=$virtual_root_path -w $script_path dockerdist.bdmd.com/yangrui:tensor2tensor_1.6.2 ./train_t2t162.sh

### chanyi
#image_=rxthinking:tf1.11_req   # bert
#image_=image_=dockerdist.bdmd.com/rxthinking:tf1.12_t2t1.10  # bayes t2t
#image_=rxthinking:tf1.12_t2t1.11_req
#image_=rxthinking:tf1.12_t2t1.11_tfp_req
image_=dockerdist.bdmd.com/rxthinking:tfp_graphnet

#docker run --name sym --runtime=nvidia -i -v $local_root_path:$virtual_root_path -e PYTHONPATH=$tool_path -e PYTHONPATH=$virtual_root_path -w $script_path $image_ ./train.sh

docker run --name sym1 --runtime=nvidia -i -v $local_root_path:$virtual_root_path -e PYTHONPATH=$tool_path -e PYTHONPATH=$virtual_root_path -w $script_path $image_  bash


#docker run --name graph --runtime=nvidia -i -v rxthinking:tfp_graphnet bash -c "python"


