#!/bin/bash

#cp ./yr_models.py /usr/local/lib/python3.5/dist-packages/graph_nets/demos/models.py

export CUDA_VISIBLE_DEVICES=2
#export BERT_BASE_DIR=../../../download_model_uncase/uncased_L-12_H-768_A-12
#export GLUE_DIR=../download_data/glue_data



python3 test_ask_symptom_graphnet.py
   