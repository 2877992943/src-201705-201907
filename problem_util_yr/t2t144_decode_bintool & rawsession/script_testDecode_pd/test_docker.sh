#!/bin/bash

export ROOT_PATH=/training
#docker run --runtime=nvidia -i -v /Users/johnzhang/Desktop/test/problem:/training -e ROOT_PATH  -w /training/script_testDecode rxthinking:tensor2tensor_1.4.4 ./cp_t2t_py.sh
docker run -i -v /Users/johnzhang/Desktop/test/problem:/training -e ROOT_PATH  -w /training/script_testDecode rxthinking:tensor2tensor_1.4.4 ./cp_t2t_py.sh



