# coding:utf-8

import os,sys
import logging,json

import json,copy




import collections
import itertools
import time

import tensorflow as tf
from graph_nets import utils_np
from graph_nets import utils_tf
import numpy as np

DATA_TYPE_HERE=tf.float32



def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


def get_graph_tuple_from_g(gll):
    return utils_np.networkxs_to_graphs_tuple(gll)




def get_optimizer1():
    # Optimizer.
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer
def get_optimizer2():
    learning_rate = tf.get_variable(
        "lr", initializer=tf.constant(FLAGS.lr_start, shape=(), dtype=tf.float32))
    learning_rate_update = learning_rate.assign(learning_rate * FLAGS.lr_decay)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    optimizer = GlobalNormClippingOptimizer(optimizer, clip_norm=5.0)
    return optimizer




def get_old_optimz1():
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    return optimizer



def get_optimz():
    # Optimize as usual.
    global_step = tf.get_variable(
        "num_weight_updates",
        initializer=tf.constant(0, dtype=tf.int32, shape=()),
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    learning_rate = tf.get_variable(
        "lr", initializer=tf.constant(FLAGS.lr_start, shape=(), dtype=tf.float32))
    learning_rate_update = learning_rate.assign(learning_rate * FLAGS.lr_decay)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    optimizer = GlobalNormClippingOptimizer(optimizer, clip_norm=5.0)



# global
def create_loss_ops_sigmoidXtarget(target_op, output_ops):

    """
    target[0 0 0 0 1 1 0 0 0]
    predict_prob [0.4 0.2 0.5 f...] f=sigmoid(x)
    loss=target * log(predict_prob) + (1-target) * log(1-predict_prob) # max this
    loss=-loss
    vocabsz 1000 , there are 1000  individual 2-category classification
    """

    eps = 0.000001

    #
    # ###
    # def onehotY(tensor,vocabsz): #
    #     y =tf.one_hot(tensor,vocabsz)
    #     #y=tf.concat([y1,y2],axis=1)#[? 1494][? 1494]
    #     #y=tf.squeeze(y,axis=1)
    #     y=tf.cast(y, tf.float64)
    #     return y

    loss=[]
    target=target_op.globals #[? vocabsz]
    # target_multihot = tf.reduce_sum(onehotY(target_op.globals[:, :], VOCABSZ_DIAGNOSE),
    #                                 axis=1)[:,1:]  # [? ,num_disease] -> [? numdisease 325] ->[? 325]->[? 324] remove padID
    # target_multihot = tf.concat([target_multihot, target_n], axis=-1)

    # #### split output -> multihot [325,]  n_disease [1]
    for output_op in output_ops:
        pred_globals = output_op.globals[:, :]  # [? vocabsz]

        pred_globals_prob= tf.sigmoid(pred_globals)

        target=tf.cast(target,dtype=DATA_TYPE_HERE)
        pred_globals_prob=tf.cast(pred_globals_prob,dtype=DATA_TYPE_HERE)

        loss.append(-1.*(target*tf.log(pred_globals_prob+eps) + (1.-target)*tf.log(1.-pred_globals_prob+eps))) # [? vocabsz]->[?,]

    return tf.reduce_mean(loss) # when every thing is correct , log1=0,loss=0



def create_loss_global_softmax(target_op, output_ops):


    eps = 0.000001

    #
    # ###
    # def onehotY(tensor,vocabsz): #
    #     y =tf.one_hot(tensor,vocabsz)
    #     #y=tf.concat([y1,y2],axis=1)#[? 1494][? 1494]
    #     #y=tf.squeeze(y,axis=1)
    #     y=tf.cast(y, tf.float64)
    #     return y

    loss=[]
    target=target_op.globals #[? vocabsz]
    # target_multihot = tf.reduce_sum(onehotY(target_op.globals[:, :], VOCABSZ_DIAGNOSE),
    #                                 axis=1)[:,1:]  # [? ,num_disease] -> [? numdisease 325] ->[? 325]->[? 324] remove padID
    # target_multihot = tf.concat([target_multihot, target_n], axis=-1)

    # #### split output -> multihot [325,]  n_disease [1]
    for output_op in output_ops:
        pred_globals = output_op.globals[:, :]  # [? vocabsz]

        pred_globals_prob= tf.nn.softmax(pred_globals)

        target=tf.cast(target,dtype=DATA_TYPE_HERE)
        pred_globals_prob=tf.cast(pred_globals_prob,dtype=DATA_TYPE_HERE)

        loss.append(-1.*(target*tf.log(pred_globals_prob+eps))) # [? vocabsz]->[?,]

    return tf.reduce_mean(loss) # when every thing is correct , log1=0,loss=0





def rawcalc_loss_softmax(target,pred): # -(p1logp1 +p0logp0)
    #-tf.reduce_mean(tf.reduce_sum(tmp * tf.log(tf.nn.softmax(pred_globals)), axis=-1))

    target=tf.cast(target,dtype=DATA_TYPE_HERE)
    pred=tf.cast(pred,dtype=DATA_TYPE_HERE)
    loss=-tf.reduce_mean(tf.reduce_sum(target * tf.log(tf.nn.softmax(pred)), axis=-1))
    return loss
def rawcalc_loss_sigmoid(target,pred): # -(p1logp1 +p0logp0)
    #-tf.reduce_mean(tf.reduce_sum(tmp * tf.log(tf.nn.softmax(pred_globals)), axis=-1))
    eps=0.000001
    target=tf.cast(target,dtype=DATA_TYPE_HERE)
    pred=tf.cast(pred,dtype=DATA_TYPE_HERE)
    loss=-tf.reduce_mean(tf.reduce_sum(target * tf.log(eps+tf.nn.sigmoid(pred)), axis=-1)) # log(x) x must >0
    return loss





# global
def create_loss_ops_sqrt(target_op, output_ops): # multihot 325 | n_disease 10
    """
        target[0 0 0 0 1 1 0 0 0]
        predict [0.4 0.2 0.5 f...] f=sigmoid(x)
        loss=sqrt(target - predict)
    """


    #
    # ###
    # def onehotY(tensor,vocabsz): #
    #     y =tf.one_hot(tensor,vocabsz)
    #     #y=tf.concat([y1,y2],axis=1)#[? 1494][? 1494]
    #     #y=tf.squeeze(y,axis=1)
    #     y=tf.cast(y, tf.float64)
    #     return y

    loss=[]
    target=target_op.globals
    # target_multihot = tf.reduce_sum(onehotY(target_op.globals[:, :], VOCABSZ_DIAGNOSE),
    #                                 axis=1)[:,1:]  # [? ,num_disease] -> [? numdisease 325] ->[? 325]->[? 324] remove padID
    # target_multihot = tf.concat([target_multihot, target_n], axis=-1)

    # #### split output -> multihot [325,]  n_disease [1]
    for output_op in output_ops:
        pred_globals = output_op.globals[:, :]  # [? vocabsz]
        #loss.append(tf.losses.sigmoid_cross_entropy(target_multihot, pred_globals))  # ignore PADID when calcate loss
        loss.append(multihot_square(target,pred_globals)) # [? vocabsz]->[?,]

    return tf.reduce_mean(loss)

# node
def create_loss_ops_node(target_op, output_ops):

    def onehotY(tensor): # target [? 2]
        x_in_y=tensor[:,0]# target.node [432 padid]not  grow |  [432 45] 432 grow to 45
        y=tensor[:,1]
        y =tf.one_hot(y,VOCABSZ_SYMPTOM)

        y=tf.cast(y, tf.float64)
        return y


    loss_ops = [
        tf.losses.softmax_cross_entropy(onehotY(target_op.nodes), output_op.nodes)

        for output_op in output_ops
    ]
    return loss_ops[:]

# global
def calc_accuracy_multihot(output,target):#  pred [? vocabsz]

    output[np.where(output>0.5)]=1.
    output[np.where(output <= 0.5)] = 0.

    output=output.flatten()
    target=target.flatten()
    pred_is_1=np.sum(output)
    tt_n=output.__len__()
    correct_and_is_1_not_0=np.sum(output*target) #  both predict target is 1, common set between predict and target

    return correct_and_is_1_not_0/float(pred_is_1+0.000001)







# node
def compute_accuracy_top5_yr(target, output):  # target [bxn, 2] output [bxn,vocabsz]
    # print ''
    ## pred
    node_pred = np.argsort(-output.nodes, axis=1)[:,:5]  # [n_node, vocabsz_node] ->[bxn,5]
    ## target
    node_target = target.nodes[:, 1]

    n_correct = 0
    n_tt = 1
    for ii in range(node_pred.shape[0]):
        pred = node_pred[ii]
        target = node_target[ii]
        if target == 0: continue
        ###
        n_tt += 1
        if target in list(pred):
            n_correct += 1
    return n_correct / float(n_tt)



# node
def compute_accuracy_yr(target,output):# target [bxn, 2] output [bxn,vocabsz]
    #print ''
    ## pred
    node_pred=np.argmax(output.nodes,axis=1) #[n_node, vocabsz_node] # 5 nodes[0 0 0 0 0] [0 124 0 0 0]
    ## target
    node_target=target.nodes[:,1]

    n_correct=0
    n_tt=1
    for ii in range(node_pred.shape[0]):
        pred=node_pred[ii]
        target=node_target[ii]
        if pred==0:continue
        ###
        n_tt+=1
        if pred==target:
            n_correct+=1
    return n_correct/float(n_tt)