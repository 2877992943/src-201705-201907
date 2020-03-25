#!/bin/bash


#find / -name transformer.py
#/usr/lib/python2.7/compiler/transformer.py
#/training/trainers/sympton/script_testDecode/transformer.py

#docker run --runtime=nvidia -i -v /home/yangrui/:/training -e ROOT_PATH -e PYTHONPATH=/training -w /training dockerdist.bdmd.com/yangrui:tensor2tensor_1.3.2 ls usr/

#cp /training/trainers/sympton/script_testDecode/transformer.py /usr/local/lib/python2.7/dist-packages/tensor2tensor/models/transformer.py
#cat /usr/local/lib/python2.7/dist-packages/tensor2tensor/models/transformer.py

python ./ProblemDecoder_t2t144.py







