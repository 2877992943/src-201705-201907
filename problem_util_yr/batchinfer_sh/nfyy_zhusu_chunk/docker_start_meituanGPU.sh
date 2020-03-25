#!/bin/bash


export PYTHONPATH=/custom_tool

problem_file_name=problem_xbsNew_relation_cutSentence


docker rm xbsPredict


docker run --runtime=nvidia \
--name=xbsPredict \
-v /data/public/yangrui/:/vroot \
-v /home/yangrui/:$PYTHONPATH \
-e PYTHONPATH=/vroot \
-w /vroot/$problem_file_name \
-i -t \
rxthinking:tensor2tensor_1.6.2 /vroot/$problem_file_name/batch_infer22w/docker_batch_decode_chunk.sh



