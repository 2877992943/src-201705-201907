#encoding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from graph_nets.utils_np import data_dicts_to_graphs_tuple
"""https://www.tensorflow.org/tutorials/load_data/tf_records"""
import copy
from graph_nets import graphs


#tf.enable_eager_execution()
#tf.executing_eagerly()

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_yr(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

#####
v=_int64_feature_yr([56,34,901])
print (v)




def serialize_example_yr(feature0, feature1, feature2,feature3,feature4):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'edge': _int64_feature_yr(feature0),
        'node': _int64_feature_yr(feature1),
        'global': _int64_feature_yr(feature2),
        'receive':_int64_feature_yr(feature3),
        'send':_int64_feature_yr(feature4),

    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# def serialize_example_yr_xy(feature0, feature1, feature2,feature3,feature4,
#                             feature00, feature01, feature02, feature03, feature04):
#     """
#     Creates a tf.Example message ready to be written to a file.
#     """
#
#     # Create a dictionary mapping the feature name to the tf.Example-compatible
#     # data type.
#
#     feature = {
#         'edge0': _int64_feature_yr(feature0),
#         'node0': _int64_feature_yr(feature1),
#         'global0': _int64_feature_yr(feature2),
#         'receive0':_int64_feature_yr(feature3),
#         'send0':_int64_feature_yr(feature4),
#         'edge1': _int64_feature_yr(feature00),
#         'node1': _int64_feature_yr(feature01),
#         'global1': _int64_feature_yr(feature02),
#         'receive1': _int64_feature_yr(feature03),
#         'send1': _int64_feature_yr(feature04),
#
#     }
#
#     # Create a Features message using tf.train.Example.
#
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example_proto.SerializeToString()





# def serialize_example_yr_x(datadict):
#     feature={}
#     for k,v in datadict.items():
#         feature[k]=_int64_feature_yr(v)
#     ##
#     # Create a Features message using tf.train.Example.
#
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example_proto.SerializeToString()


def serialize_example_yr_gtx_gty(gtx,gty):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.


    feature={}
    ## gtx
    feature['x_edges']=_int64_feature_yr(gtx.edges)
    feature['x_nodes'] = _int64_feature_yr(gtx.nodes)
    feature['x_globals'] = _int64_feature_yr(gtx.globals)
    feature['x_receivers'] = _int64_feature_yr(gtx.receivers)
    feature['x_senders'] = _int64_feature_yr(gtx.senders)
    feature['x_n_edge'] = _int64_feature_yr(gtx.n_edge)
    feature['x_n_node'] = _int64_feature_yr(gtx.n_node)
    ##
    # gty
    feature['y_edges'] = _int64_feature_yr(gty.edges)
    feature['y_nodes'] = _int64_feature_yr(gty.nodes) ###
    feature['y_globals'] = _int64_feature_yr(gty.globals.flatten().astype(int))
    feature['y_receivers'] = _int64_feature_yr(gty.receivers)
    feature['y_senders'] = _int64_feature_yr(gty.senders)
    feature['y_n_edge'] = _int64_feature_yr(gty.n_edge)
    feature['y_n_node'] = _int64_feature_yr(gty.n_node)

    ####


    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _parse_function(example_proto):  ### serialstr-> structure
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, name_to_features)


def get_serial_writer(filename):
    writer = tf.python_io.TFRecordWriter(filename)
    return writer

# name_to_features = {
#         "edge0": tf.VarLenFeature(tf.int64),
#         "node0": tf.VarLenFeature(tf.int64),
#         "global0": tf.VarLenFeature(tf.int64),
#         "receive0": tf.VarLenFeature(tf.int64),
#         "send0": tf.VarLenFeature(tf.int64),
#         "edge1": tf.VarLenFeature(tf.int64),
#         "node1": tf.VarLenFeature(tf.int64),
#         "global1": tf.VarLenFeature(tf.int64),
#         "receive1": tf.VarLenFeature(tf.int64),
#         "send1": tf.VarLenFeature(tf.int64),
#       }
keyList=['x_edges','x_nodes','x_globals','x_receivers','x_senders','x_n_edge','x_n_node',
         'y_edges', 'y_nodes', 'y_globals', 'y_receivers', 'y_senders', 'y_n_edge', 'y_n_node']
name_to_features = {
    "x_edges": tf.VarLenFeature(tf.int64),
    "x_nodes": tf.VarLenFeature(tf.int64),
    "x_globals": tf.VarLenFeature(tf.int64),
    "x_receivers": tf.VarLenFeature(tf.int64),
    "x_senders": tf.VarLenFeature(tf.int64),
    "x_n_edge": tf.VarLenFeature(tf.int64),
    "x_n_node": tf.VarLenFeature(tf.int64),
    "y_edges": tf.VarLenFeature(tf.int64),
    "y_nodes": tf.VarLenFeature(tf.int64),
    "y_globals": tf.VarLenFeature(tf.int64),
    "y_receivers": tf.VarLenFeature(tf.int64),
    "y_senders": tf.VarLenFeature(tf.int64),
    "y_n_edge": tf.VarLenFeature(tf.int64),
    "y_n_node": tf.VarLenFeature(tf.int64),
  }

# def read_fileList_generate_data(filenamell,sess1=None,buffer=10):
#
#     raw_dataset = tf.data.TFRecordDataset(filenamell)
#
#     parsed_dataset = raw_dataset.map(_parse_function)
#     print (parsed_dataset.output_shapes)
#     print (parsed_dataset.output_types)
#     print (parsed_dataset.output_classes)
#
#     #### 迭代 dataset
#
#     ####
#     parsed_dataset=parsed_dataset.repeat()
#     dataset = parsed_dataset.batch(1)
#     iterator = dataset.make_initializable_iterator()
#     next_element = iterator.get_next()
#
#
#     ##
#     v=next_element
#     node0 =  v['node0']
#     edge0 =  v['edge0']
#     gro0 =  v['global0']
#     receive0 =  v['receive0']
#     send0 =  v['send0']
#     node1 = v['node1']
#     edge1 = v['edge1']
#     gro1 = v['global1']
#     receive1 = v['receive1']
#     send1 = v['send1']
#     ### tensor  sparse -> dense
#     node0_t = tf.sparse_tensor_to_dense(node0)
#     edge0_t=tf.sparse_tensor_to_dense(edge0)
#     glo0_t=tf.sparse_tensor_to_dense(gro0)
#     receive0_t=tf.sparse_tensor_to_dense(receive0)
#     send0_t=tf.sparse_tensor_to_dense(send0)
#     node1_t = tf.sparse_tensor_to_dense(node1)
#     edge1_t = tf.sparse_tensor_to_dense(edge1)
#     glo1_t = tf.sparse_tensor_to_dense(gro1)
#     receive1_t = tf.sparse_tensor_to_dense(receive1)
#     send1_t = tf.sparse_tensor_to_dense(send1)
#     #######
#     #### session
#     sess1 = tf.Session()
#     sess1.run(iterator.initializer)
#
#     ###
#     ll=[]
#     while True:
#
#
#
#         node0, edge0, glo0, receive0, send0,node1, edge1, glo1, receive1, send1= \
#             sess1.run([node0_t,edge0_t,glo0_t,receive0_t,send0_t,
#                     node1_t, edge1_t, glo1_t, receive1_t, send1_t])
#
#         ###### get datadict
#         datadict0={}
#         datadict0['nodes']=node0.flatten()
#         datadict0['edges']=edge0.flatten()
#         datadict0['receivers']=receive0.flatten()
#         datadict0['senders']=send0.flatten()
#         datadict0['globals']=glo0.flatten()
#         datadict0['n_node']=node0.shape[1]
#         datadict0['n_edge']= edge0.shape[1]
#         ##
#         datadict1 = {}
#         datadict1['nodes'] = node1.flatten()
#         datadict1['edges'] = edge1.flatten()
#         datadict1['receivers'] = receive1.flatten()
#         datadict1['senders'] = send1.flatten()
#         datadict1['globals'] = glo1.flatten()
#         datadict1['n_node'] = node1.shape[1]
#         datadict1['n_edge'] = edge1.shape[1]
#         ### datadict -> graph
#         ll.append([copy.copy(datadict0),copy.copy(datadict1)])
#         ####
#         if len(ll)>buffer:
#             ll_ret=copy.copy(ll)
#             ll=[]
#             yield ll_ret
#
#
# def get_next_from_fileList(filenamell,batchsz=1):
#     raw_dataset = tf.data.TFRecordDataset(filenamell)
#
#     parsed_dataset = raw_dataset.map(_parse_function)
#     print(parsed_dataset.output_shapes)
#     print(parsed_dataset.output_types)
#     print(parsed_dataset.output_classes)
#
#     #### 迭代 dataset
#
#     ####
#     parsed_dataset = parsed_dataset.repeat()
#     dataset = parsed_dataset.batch(batchsz)
#     iterator = dataset.make_initializable_iterator()
#     next_element = iterator.get_next()
#     ##
#
#     def change_fn(v):
#         node0 = v['node0']
#         edge0 = v['edge0']
#         gro0 = v['global0']
#         receive0 = v['receive0']
#         send0 = v['send0']
#         node1 = v['node1']
#         edge1 = v['edge1']
#         gro1 = v['global1']
#         receive1 = v['receive1']
#         send1 = v['send1']
#         ### tensor  sparse -> dense
#         node0_t = tf.sparse_tensor_to_dense(node0)
#         edge0_t = tf.sparse_tensor_to_dense(edge0)
#         glo0_t = tf.sparse_tensor_to_dense(gro0)
#         receive0_t = tf.sparse_tensor_to_dense(receive0)
#         send0_t = tf.sparse_tensor_to_dense(send0)
#         node1_t = tf.sparse_tensor_to_dense(node1)
#         edge1_t = tf.sparse_tensor_to_dense(edge1)
#         glo1_t = tf.sparse_tensor_to_dense(gro1)
#         receive1_t = tf.sparse_tensor_to_dense(receive1)
#         send1_t = tf.sparse_tensor_to_dense(send1)
#         ###
#         return {'node0':node0_t,
#                 'edge0':edge0_t,
#                 'global0':glo0_t,
#                 'receive0':receive0_t,
#                 'send0':send0_t,
#                 'node1':node1_t,
#                 'edge1':edge1_t,
#                 'global1':glo1_t,
#                 'receive1':receive1_t,
#                 'send1':send1_t}
#
#
#
#
#     return next_element,iterator,change_fn
#
#     ##
#
#
#
#
# class read_generate_data():
#     def __init__(self,filenamell,buffersz):
#         self.buffersz=buffersz
#         ### build graph
#         raw_dataset = tf.data.TFRecordDataset(filenamell)
#
#         parsed_dataset = raw_dataset.map(_parse_function)
#         print (parsed_dataset.output_shapes)
#         print (parsed_dataset.output_types)
#         print (parsed_dataset.output_classes)
#
#
#
#         ####
#         parsed_dataset = parsed_dataset.repeat()
#         dataset = parsed_dataset.batch(1)
#         self.iterator = dataset.make_initializable_iterator()
#         next_element = iterator.get_next()
#         ####
#         #sess1 = tf.Session()
#         #sess1.run(iterator.initializer)
#
#         ##
#         v = next_element
#         node0 = v['node0']
#         edge0 = v['edge0']
#         gro0 = v['global0']
#         receive0 = v['receive0']
#         send0 = v['send0']
#         node1 = v['node1']
#         edge1 = v['edge1']
#         gro1 = v['global1']
#         receive1 = v['receive1']
#         send1 = v['send1']
#         ### tensor  sparse -> dense
#         self.node0_t = tf.sparse_tensor_to_dense(node0)
#         self.edge0_t = tf.sparse_tensor_to_dense(edge0)
#         self.glo0_t = tf.sparse_tensor_to_dense(gro0)
#         self.receive0_t = tf.sparse_tensor_to_dense(receive0)
#         self.send0_t = tf.sparse_tensor_to_dense(send0)
#         self.node1_t = tf.sparse_tensor_to_dense(node1)
#         self.edge1_t = tf.sparse_tensor_to_dense(edge1)
#         self.glo1_t = tf.sparse_tensor_to_dense(gro1)
#         self.receive1_t = tf.sparse_tensor_to_dense(receive1)
#         self.send1_t = tf.sparse_tensor_to_dense(send1)
#         ###
#     def start(self,sess1):
#         self.sess1=sess1
#         self.sess1.run(self.iterator.initializer)
#     def generate(self):
#         ll=[]
#         #for _ in range(num):
#         while True:
#             node0, edge0, glo0, receive0, send0, node1, edge1, glo1, receive1, send1 = \
#                 self.sess1.run([self.node0_t, self.edge0_t, self.glo0_t, self.receive0_t, self.send0_t,
#                            self.node1_t, self.edge1_t, self.glo1_t, self.receive1_t, self.send1_t])
#
#             ###### get datadict
#             datadict0 = {}
#             datadict0['nodes'] = node0.flatten()
#             datadict0['edges'] = edge0.flatten()
#             datadict0['receivers'] = receive0.flatten()
#             datadict0['senders'] = send0.flatten()
#             datadict0['globals'] = glo0.flatten()
#             datadict0['n_node'] = node0.shape[1]
#             datadict0['n_edge'] = edge0.shape[1]
#             ##
#             datadict1 = {}
#             datadict1['nodes'] = node1.flatten()
#             datadict1['edges'] = edge1.flatten()
#             datadict1['receivers'] = receive1.flatten()
#             datadict1['senders'] = send1.flatten()
#             datadict1['globals'] = glo1.flatten()
#             datadict1['n_node'] = node1.shape[1]
#             datadict1['n_edge'] = edge1.shape[1]
#
#             ll.append([copy.copy(datadict0), copy.copy(datadict1)])
#             ####
#             if len(ll)>self.buffersz:
#                 ll_ret=copy.copy(ll)
#                 ll=[]
#                 yield ll_ret
#




def filenameList_to_iterator(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)

    parsed_dataset = raw_dataset.map(_parse_function)
    print (parsed_dataset.output_shapes)
    print (parsed_dataset.output_types)
    print (parsed_dataset.output_classes)

    parsed_dataset = parsed_dataset.repeat()
    dataset = parsed_dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    return next_element,iterator

def tensor_sparse_to_dense(next_element):
    fea = {}
    for key in keyList:
        v = next_element[key]
        v = tf.sparse_tensor_to_dense(v)
        v=tf.cast(v,dtype=tf.int32)
        ## reshape
        if key == 'y_globals':
            v = tf.reshape(v, [-1, 360])
        elif key == 'x_globals':
            v = tf.reshape(v, [-1, 1])
        elif key in ['x_edges', 'y_edges', 'x_nodes', 'y_nodes']:
            v = tf.reshape(v, [-1, 1]) # node [n_node,node_dim]
        elif key in ['x_n_edge', 'x_n_node', 'y_n_edge', 'y_n_node', 'x_senders', 'y_senders', 'x_receivers',
                     'y_receivers']:
            #v = tf.squeeze(v)   # if squeeze , shape not (?,)
            v=tf.reshape(v,[-1,])
            #v=v
        fea[key] = v
    ### split x y
    xfea,yfea={},{}
    for k,v in fea.items():
        if 'x' in k:
            k=k.strip('x_')
            xfea[k]=v
        elif 'y' in k:
            k = k.strip('y_')
            yfea[k] = v

    return xfea,yfea

def get_gt(xdic,ydic):
    #### datadict -> gt

    xfea = graphs.GraphsTuple(**xdic)
    yfea = graphs.GraphsTuple(**ydic)
    return xfea,yfea


if __name__=='__main__':
    #### 序列化

    serialized_example = serialize_example_yr([56,76],[1001,1002,1003,1004],[100],[0,2],[1,3]) # edge node global receive send
    serialized_example1 = serialize_example_yr([58,78,88],[1001,1005,1006,1007,1008],[100],[0,2,4],[1,3,0])

    ### writer
    filename='test.data'
    writer = tf.python_io.TFRecordWriter(filename)
    writer.write(serialized_example)
    writer.write(serialized_example1)
    writer.close()



    ################################################################
    ###########   已经写完  读取 和 还原
    #######################################################

    name_to_features = {
        "edge": tf.VarLenFeature(tf.int64),
        "node": tf.VarLenFeature(tf.int64),
        "global": tf.VarLenFeature(tf.int64),
        "receive": tf.VarLenFeature(tf.int64),
        "send": tf.VarLenFeature(tf.int64),
      }


    raw_dataset = tf.data.TFRecordDataset(filename)





    parsed_dataset = raw_dataset.map(_parse_function)
    print (parsed_dataset.output_shapes)
    print (parsed_dataset.output_types)
    print (parsed_dataset.output_classes)


    """ 
    ###
    ### 展示 有限条
    for parsed_record in parsed_dataset.take(10):
      print repr(parsed_record)
    """

    #### 迭代 dataset
    sess=tf.Session()





    parsed_dataset=parsed_dataset.repeat()
    dataset = parsed_dataset.batch(1)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess.run(iterator.initializer)
    ll=[]
    ##
    v=next_element
    node =  v['node']
    edge =  v['edge']
    gro =  v['global']
    receive =  v['receive']
    send =  v['send']
    ### tensor  sparse -> dense
    node_t = tf.sparse_tensor_to_dense(node)
    edge_t=tf.sparse_tensor_to_dense(edge)
    glo_t=tf.sparse_tensor_to_dense(gro)
    receive_t=tf.sparse_tensor_to_dense(receive)
    send_t=tf.sparse_tensor_to_dense(send)

    for _ in range(4):

        node, edge, glo, receive, send= sess.run([node_t,edge_t,glo_t,receive_t,send_t])

        ###### get datadict
        datadict={}
        datadict['nodes']=node.flatten()
        datadict['edges']=edge.flatten()
        datadict['receivers']=receive.flatten()
        datadict['senders']=send.flatten()
        datadict['globals']=glo.flatten()
        datadict['n_node']=node.shape[1]
        datadict['n_edge']= edge.shape[1]
        ### datadict -> graph
        ll.append(copy.copy(datadict))
    ####
    gt=data_dicts_to_graphs_tuple(ll)
    print ('')








