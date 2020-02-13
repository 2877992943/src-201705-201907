# coding:utf-8

import os,sys
import logging,json
#reload(sys)
#sys.setdefaultencoding("utf8")
import json,copy
import pandas as pdd


path_to_be_add='/Users/yangrui/Desktop/bdmd/notebook_learn/tf2.0_图模型migrate/graph_nets-master_tf2'
import sys
sys.path.insert(0, path_to_be_add)


import collections
import itertools
import time

#from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
#from graph_nets.demos import models
import models_yr_v1104 as models


import networkx as nx
import numpy as np
#from scipy import spatial
import tensorflow as tf
import collections

from graph_utils import numpy_graph_to_tensor,\
    create_loss_seq_softmax_global

SEED = 1
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
import random


from absl import flags

FLAGS = flags.FLAGS


# flags.DEFINE_string("node_vocabsz", None,
#                     "number of nodes,sym")
# FLAGS.node_vocabsz=3666 ###  change
#VOCABSZ_SYMPTOM=FLAGS.node_vocabsz
VOCABSZ_SYMPTOM=1666

NEPOCH=200000000


BATCHSZ=1*50
debug_flag=True ## change
TT_MEMORY= 1000#2000*6

from lstm_decoder_class_keras import lstmDecoder

def relabel_node2(G):
    num=G.number_of_nodes()
    nodell=G.nodes()
   #nodell=sorted(nodell) #否则每次都不一样 xgraph ygraph
    dic=dict(zip(nodell,range(num)))
    H = nx.relabel_nodes(G, dic)
    return H


def compute_accuracy(target, output, use_nodes=True, use_edges=False):
  """Calculate model accuracy.

  Returns the number of correctly predicted shortest path nodes and the number
  of completely solved graphs (100% correct predictions).

  Args:
    target: A `graphs.GraphsTuple` that contains the target graph.
    output: A `graphs.GraphsTuple` that contains the output graph.
    use_nodes: A `bool` indicator of whether to compute node accuracy or not.
    use_edges: A `bool` indicator of whether to compute edge accuracy or not.

  Returns:
    correct: A `float` fraction of correctly labeled nodes/edges.
    solved: A `float` fraction of graphs that are completely correctly labeled.

  Raises:
    ValueError: Nodes or edges (or both) must be used
  """
  if not use_nodes and not use_edges:
    raise ValueError("Nodes or edges (or both) must be used")
  tdds = utils_np.graphs_tuple_to_data_dicts(target)
  odds = utils_np.graphs_tuple_to_data_dicts(output)
  cs = []
  ss = []
  for td, od in zip(tdds, odds):
    xn = np.argmax(td["nodes"], axis=-1)
    yn = np.argmax(od["nodes"], axis=-1)
    xe = np.argmax(td["edges"], axis=-1)
    ye = np.argmax(od["edges"], axis=-1)
    c = []
    if use_nodes:
      c.append(xn == yn)
    if use_edges:
      c.append(xe == ye)
    c = np.concatenate(c, axis=0)
    s = np.all(c)
    cs.append(c)
    ss.append(s)
  correct = np.mean(np.concatenate(cs, axis=0))
  solved = np.mean(np.stack(ss))
  return correct, solved


def compute_accuracy_yr(target,output):# target [bxn, 2] output [bxn,vocabsz]
    #print ''
    ## pred
    node_pred=np.argmax(output.nodes,axis=1) #[n_node, vocabsz_node] # 5 nodes[0 0 0 0 0] [0 124 0 0 0]
    ## target
    node_target=target.nodes[:,0]*target.nodes[:,1] # feaid x mask

    n_correct=0
    n_tt=1
    for ii in range(node_pred.shape[0]):
        pred=node_pred[ii]
        target=node_target[ii]
        if target==0:continue
        if pred==0:continue
        ###
        n_tt+=1
        if pred==target:
            n_correct+=1
    return n_correct/float(n_tt+0.00001)




def compute_accuracy_seq_yr(target,output):# target [batch step] logit [batch step vocabsz]
    #print ''
    ## pred
    y_arr=np.array(target)
    pred=np.argmax(output,axis=2) #
    pred=pred.flatten()
    ## target
    target=y_arr.flatten()

    n_correct=0
    n_tt=0
    for ii in range(pred.shape[0]):
        pred_i=pred[ii]
        target_i=target[ii]
        if target_i==0:continue
        if pred_i==0:continue
        ###
        n_tt+=1
        if pred_i==target_i:
            n_correct+=1
    return n_correct/float(n_tt+0.00001)




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






# def create_loss_ops(target_op, output_ops):
#
#     def onehotY(tensor): # target [? 2]
#         x_in_y=tensor[:,0]# target.node [432 padid]not  grow |  [432 45] 432 grow to 45
#         y=tensor[:,1]
#         y =tf.one_hot(y,VOCABSZ_SYMPTOM)
#
#         y=tf.cast(y, tf.float64)
#         return y
#
#
#     loss_ops = [
#         tf.compat.v1.losses.softmax_cross_entropy(onehotY(target_op.nodes), output_op.nodes)
#
#         for output_op in output_ops
#     ]
#     return loss_ops[:]




# def create_loss_seq_softmax_global(target,logits): # logit[batch step vocabsz] target[batch step 1 1]
#     ##
#     probs=tf.nn.softmax(logits,axis=2)#[batch step vocabsz]
#     target=tf.squeeze(target,axis=[2,3])
#
#     ##
#
#     def onehotY(tensor):
#         y=tensor #[batch step]
#         y =tf.one_hot(y,VOCABSZ_SYMPTOM)
#         #y=tf.cast(y, tf.float64)
#         return y
#
#     ###
#     target_onehot=onehotY(target)#[batch step vocabsz]
#     eps=0.000001
#     rst=tf.reduce_sum(input_tensor=target_onehot*tf.math.log(probs+eps),axis=2)
#     return -tf.reduce_mean(input_tensor=rst)






# def create_loss_softmax_seq(target_op,output_ops):
#     def onehotY(tensor): # target [? 2] first fea,second mask
#         y=tensor[:,0]# target.node [432 mask ]    | y=[n_node,]
#         mask=tensor[:,1]    # mask =[n_node,]
#         y =tf.one_hot(y,VOCABSZ_SYMPTOM) #[n,]->[n,vocabsz]
#         mask=tf.tile(tf.reshape(mask,[-1,1]),[1,VOCABSZ_SYMPTOM]) #[n,]->[n,vocabsz]
#
#         y=tf.cast(y, tf.float64)*tf.cast(mask, tf.float64)
#
#         return y
#
#
#     loss_ops = [
#         tf.losses.softmax_cross_entropy(onehotY(target_op.nodes), output_op.nodes)
#
#         for output_op in output_ops
#     ]
#     return loss_ops[:]





def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]








def get_graph_tuple_from_g(gll):
    return utils_np.networkxs_to_graphs_tuple(gll)



def data_generator(datapath):
    #####
    for eopch in range(NEPOCH):
        print ('epoch',eopch)
        #batchsz=int(TT_MEMORY/num_edge)
        #####
        for fname in os.listdir(datapath):

            if '.pkl' not in fname:continue
            if debug_flag==True:
                if 'graph_xy_9189.pkl' not in fname:continue
            fname=os.path.join(datapath,fname)
            print (fname)
            allObs=pdd.read_pickle(fname)
            # shuffle
            allObs_ind=list(range(len(allObs))) #python3 range not list
            random.shuffle(allObs_ind)
            #
            tt_num=len(allObs)

            batchsz= 10000

            nn=int(tt_num/batchsz)
            nn=1 if nn==0 else nn

            for ii in range(nn):
                thisbatch_ind=allObs_ind[ii*batchsz:ii*batchsz+batchsz]
                thisbatch=[allObs[ind] for ind in thisbatch_ind]
                num_edge=np.mean([obs['numedge'] for obs in thisbatch])
                #
                batchsz_real=int(TT_MEMORY/num_edge)
                n_batch_real=int(TT_MEMORY/batchsz_real)
                for iii in range(n_batch_real):
                    thisbatch_=thisbatch[iii*batchsz_real:(iii+1)*batchsz_real]
                    if len(thisbatch_)==0:continue

                    yield thisbatch_ # xll,yll




def prepare_x(gllx):# networkx list -> np graphtuple ->tensor
    gtx_np = get_graph_tuple_from_g(gllx)
    gtx=numpy_graph_to_tensor(gtx_np)
    return gtx

def prepare_y(yll):

    thisbatch_y_4d = np.expand_dims(np.expand_dims(yll, axis=2), axis=2)
    return tf.constant(thisbatch_y_4d)


def body_fn(enc,dec,y,gtx):
    # encode part
    output_ops = enc(gtx, num_processing_steps)  # this output is global vec [n_graph hidsz]
    output_global = output_ops[-1].globals
    output_global = tf.cast(output_global, dtype=tf.float32)

    ## decode part

    inputs = tf.expand_dims(tf.expand_dims(output_global, axis=1), axis=1)

    logits = dec([y, inputs])  # logits[batch step 1 vocabsz]
    logits = tf.squeeze(logits, axis=2)  # [batch step vocabsz]
    return logits


if __name__=='__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    if debug_flag==True:
        datapath = u'../../tmp_sym/pkls'
    else:
        datapath=u'../../mac_yr_local/v280/pursueAsk_graph_v280_190723/tmp_sym/pkls'

    gene=data_generator(datapath)
    gllx,glly=[],[]
    for x_y in gene:
        gllx=[p['x'] for p in x_y]
        #glly=[p['y'] for p in x_y]
        glly=[p['yid'] for p in x_y]
        break







    #tf.compat.v1.reset_default_graph()

    ######
    # gtx_np = get_graph_tuple_from_g(gllx)
    # #gty_np = utils_np.networkxs_to_graphs_tuple(glly)
    # gtx=numpy_graph_to_tensor(gtx_np)
    #gty=numpy_graph_to_tensor(gty_np)
    gtx=prepare_x(gllx)
    y=prepare_y(glly)







    #### placeholder
    # input_ph = utils_tf.placeholders_from_networkxs(
    #       gllx, force_dynamic_num_graphs=True)
    # # target_ph = utils_tf.placeholders_from_networkxs(
    # #       glly, force_dynamic_num_graphs=True)
    # target_ph=tf.compat.v1.placeholder(dtype=tf.int32,shape=[None,None,1,1])# [batch step 1 1]
    #
    #
    # ##
    # print ('')










    seed = 1
    rand = np.random.RandomState(seed=seed)

    # Model parameters.
    # Number of processing (message-passing) steps.
    #num_processing_steps_tr =  1 if debug_flag==True else 12
    #num_processing_steps_ge = 16

    # Data / training parameters.
    num_training_iterations = 10000000
    #theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
    #batch_size_tr = 5
    #batch_size_ge = 5
    # Number of nodes per graph sampled uniformly from this range.
    # num_nodes_min_max_tr = (8, 17)
    # num_nodes_min_max_ge = (9, 18)

    # Data.
    # Input and target placeholders.
    # input_ph, target_ph = create_placeholders(rand, batch_size_tr,
    #                                           num_nodes_min_max_tr, theta)

    # Connect the data to the model.
    # Instantiate the model.


    ### encode process decode
    #import yr_models as models
    model = models.EncodeProcessDecode(edge_output_size=None,
                                       node_output_size=None,
                                        VOCABSZ_SYMPTOM=VOCABSZ_SYMPTOM)
    # A list of outputs, one per processing step.
    num_processing_steps=1 if debug_flag==True else 10

    # output_ops = model(gtx,num_processing_steps) # this output is global vec [n_graph hidsz]
    # output_global=output_ops[-1].globals
    # output_global = tf.cast(output_global, dtype=tf.float32)





    #### encode graph -> vec [n_graph,hid]
    latent_dim=16
    instan_lstm = lstmDecoder(vocab_size=VOCABSZ_SYMPTOM,
                         latent_dim=latent_dim,
                         train_flag=True)
    # inputs=tf.expand_dims(tf.expand_dims(output_global,axis=1),axis=1)
    #
    # logits = instan_lstm([y,inputs]) #logits[batch step 1 vocabsz]
    # logits=tf.squeeze(logits,axis=2) #[batch step vocabsz]

    #w1,w2=model.trainable_variables ,instan_lstm.trainable_weights


    logits=body_fn(model, instan_lstm, y, gtx)


    ###
    #w1, w2 = model.trainable_variables, instan_lstm.trainable_weights








    # Training loss.
    loss_ops = create_loss_seq_softmax_global(y,logits,VOCABSZ_SYMPTOM)


    #tf.compat.v1.summary.scalar('loss_graphnets', loss_ops)


    # Optimizer.
    learning_rate = 1e-3
    #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ckpt = tf.train.Checkpoint(net=model,
                                dec_net=instan_lstm,
                                opt=optimizer,
                                step=tf.Variable(1))

    # c = tf.train.latest_checkpoint(checkpoint_path_encode)
    # status = ckpt.restore(c);
    #print('encoder params restore:', c)
    # print (status.assert_existing_objects_matched())
    # print (status.assert_consumed())
    output_dir='../../model'
    manager = tf.train.CheckpointManager(
        ckpt, directory=output_dir, max_to_keep=5)

    #step_op = optimizer.minimize(loss_ops)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    #input_ph, _= make_all_runnable_in_session(input_ph, input_ph)


    #@title Reset session  { form-width: "30%" }

    # This cell resets the Tensorflow session, but keeps the same computational
    # graph.

    # try:
    #   sess.close()
    # except NameError:
    #   pass
    # sess = tf.compat.v1.Session()
    #
    # saver = tf.compat.v1.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)


    ##### tensorboard
    # merged = tf.compat.v1.summary.merge_all()
    # train_writer = tf.compat.v1.summary.FileWriter('../model/')
    ###### see whether checkpoint exist
    ckpt.restore(manager.latest_checkpoint)  # name space not 'distribute' cannot use old model
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    last_iteration = 0
    logged_iterations = []
    losses_tr = []
    corrects_tr = []
    solveds_tr = []
    losses_ge = []
    corrects_ge = []
    solveds_ge = []


    #@title Run training  { form-width: "30%" }

    # You can interrupt this cell's training loop at any time, and visualize the
    # intermediate results by running the next cell (below). You can then resume
    # training by simply executing this cell again.

    # How much time between logging and printing the current results.
    log_every_seconds = 1

    print("# (iteration number), T (elapsed seconds), "
          "Ltr (training loss), Lge (test/generalization loss), "
          "Ctr (training fraction nodes/edges labeled correctly), "
          "Str (training fraction examples solved correctly), "
          "Cge (test/generalization fraction nodes/edges labeled correctly), "
          "Sge (test/generalization fraction examples solved correctly)")









    start_time = time.time()
    last_log_time = start_time
    save_freq = 1000


    tt_step=0


    for x_y in gene:
        tt_step+=1
        tt_step+=1
        thisbatch_x = [relabel_node2(p['x']) for p in x_y]
        thisbatch_y=[p['yid'] for p in x_y]
        gtx=prepare_x(thisbatch_x)
        y=prepare_y(thisbatch_y)

        with tf.GradientTape() as tape:

            logits=body_fn(model,instan_lstm,y,gtx)
            loss_ops = create_loss_seq_softmax_global(y, logits,VOCABSZ_SYMPTOM)

        weights = list(model.trainable_variables)+list(instan_lstm.trainable_weights)

        grads = tape.gradient(loss_ops, weights)
        optimizer.apply_gradients(zip(grads, weights))

        ####

        if tt_step % save_freq == 0:
            manager.save()





        correct_tr  = compute_accuracy_seq_yr(thisbatch_y, logits.numpy())
        print ('acc',correct_tr,'loss',loss_ops.numpy())





