 
# support model: univeral transformer,transformer
# support custom_problem_type :seq2seq ,classification
# for universal transformer, need to replace t2t_model.py  session.py and below requirement
#t2t166.  has univerdal transformer
#tf 1.9.0
#numpy 1.14.4. can work
# 
#### #################some replacement 
#if need specify output length
#if use unversal_transformer and want to fetch attention_mat 
# under this circumstance , see ./replacement/
#lib/python2.7/site-packages/tensor2tensor/utils/t2t_model.py  | add feature['decode length'], if need specify output length
#lib/python2.7/site-packages/tensorflow/python/client/session.py   | remove assert unfetch, if use unversal_transformer and want to fetchattention_mat 