# coding:utf-8

import os,sys
import logging,json
#reload(sys)
#sys.setdefaultencoding("utf8")
import json,copy
import pandas as pdd


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import collections
import itertools
import time
import numpy as np




from graph_nets import utils_np
from graph_nets import utils_tf



def get_networks_from_graph_tuple(graphTuple):
    #return utils_np.graphs_tuple_to_networkxs(graphTuple)
    return utils_np.graphs_tuple_to_data_dicts(graphTuple)



def get_nodes_edge_from_gt(gt,node_id2str):
    graph = get_networks_from_graph_tuple(gt)
    graph = graph[0]
    nodes = graph['nodes']# [8,256]
    edges=graph['edges'] #[n,256]
    ## nodes str vec
    nodes_str2vec={}
    for id,stri in nodes_id2str.items():
        nodes_str2vec[stri]=nodes[id]
    ## edge  str vec
    receive, send = graph['receivers'], graph['senders']
    edges_str2vec = {}
    for ii in range(receive.shape[0]):
        s = send[ii]
        r = receive[ii]
        edge_i = nodes_id2str[s], nodes_id2str[r]
        edges_str2vec[' '.join(edge_i)] = edges[ii]
    return nodes_str2vec,edges_str2vec




def vec_dist(v1,v2): # small is similar
    v=v1 - v2
    return np.sqrt(np.dot(v,v))


def vec_dist1(v1,v2):# |x| |y| cos=x*y   # small is similar
    v=np.dot(v1,v2)
    v1=np.sqrt(np.dot(v1,v1))
    v2 = np.sqrt(np.dot(v2, v2))
    return -v/(v1*v2)


def pic(featureMap):
    ###   collectedGlobal 加总的节点 边  graphnetwork output: [updateEdgeSum, updateNodeSum,oldglobal]
    x = featureMap['collectedGlobal_7'].flatten()  # 加总的 updateEdge updateNode


    fig, axs = plt.subplots(4, 1)
    t = np.arange(256)
    axs[0].set_xlim(0, 256)
    axs[0].set_ylim(-5, 20)
    axs[0].plot(t, x)

    t1 = np.arange(64)
    axs[1].set_xlim(0, 256)
    axs[1].set_ylim(-5, 20)
    axs[1].plot(t1, x[:64])

    t2 = np.arange(64, 128)
    axs[2].set_xlim(0, 256)
    axs[2].set_ylim(-5, 20)
    axs[2].plot(t2, x[64:128])

    t3 = np.arange(128, 256)
    axs[3].set_xlim(0, 256)
    axs[3].set_ylim(-5, 20)
    axs[3].plot(t3, x[128:])
    plt.show()

if __name__=='__main__':

    rst = pdd.read_pickle('./tmp/x_infer.pkl')
    # print ('')
    batch_x, gtx, infer_values = rst  # collected global [64edge,64node,128global]
    xgraph = batch_x[0]

    ########input  id2str
    ##### 单独节点， 哪个单独节点的模式 最近似 collecteGlobal 中的 nodepart  sum_node的模式
    node_id_str = {}
    for node_index, node_feature in xgraph.nodes(data=True):
        # print ('')
        node_id_str[node_index] = node_feature['name']
    ##### 单独的边
    edge_id_str = {}
    ii = 0
    for u, v, fea in xgraph.edges(data=True):
        # print ('')
        node_u = node_id_str[u]
        node_v = node_id_str[v]
        edge_id_str[ii] = ' '.join([node_u, node_v])
        ii += 1
    ####### 节点组合
    node_2_id_str = {}
    for node_index, node_feature in xgraph.nodes(data=True):
        for node_index1, node_feature1 in xgraph.nodes(data=True):
            if node_index==node_index1:continue
            node_2_id_str[tuple([node_index,node_index1])] = node_feature['name']+'_2node_'+node_feature1['name']






    ### collected global, edgesum,nodesum
    feaMap = infer_values['featureMap']
    collectedGlobal = feaMap['collectedGlobal_7'].flatten()  # [1,256] [64edge, 64node, 128 global]
    edge_sum = collectedGlobal[0:64] # updated edge sum part in collectedGlobal
    node_sum = collectedGlobal[64:128]
    #pic(feaMap)




    ### 每个updated节点  边  node id2vec,edge id2vec
    gt_graphNetwork_7 = feaMap['core_7']
    graph = get_networks_from_graph_tuple(gt_graphNetwork_7)[0]

    core_node = graph['nodes']  # [n,d]
    core_edge = graph['edges']  # [n,d]
    ####### 节点组合的 node_2id_2vecsum
    node_2id_2vecsum={}
    for k1,k2 in node_2_id_str:
        node_2id_2vecsum[tuple([k1,k2])]=core_node[k1]+core_node[k2]


    ##############
    #  单独节点  边  距离 sumnode sumedge的距离
    node_id2dist = {}  # dist(node_i,sum_node)
    edge_id2dist = {}
    for nid in range(core_node.shape[0]):
        vec = core_node[nid]
        node_id2dist[nid] = vec_dist(vec,node_sum)
    for eid in range(core_edge.shape[0]):
        vec = core_edge[eid]
        edge_id2dist[eid]=vec_dist(vec,edge_sum)

    ### combine id2dist id2str
    node_str2dist={}
    edge_str2dist={}
    for nid in node_id2dist:
        node_str2dist[node_id_str[nid]]=float(node_id2dist[nid])
    for eid in edge_id2dist:
        edge_str2dist[edge_id_str[eid]]=float(edge_id2dist[eid])

    ################
    # 节点组合  距离 sumnode 的距离
    node_2_str_dist={}
    dist_appear=[]
    for k1k2,each_vec in node_2id_2vecsum.items():
        dist=vec_dist(each_vec,node_sum)
        if dist not in dist_appear:
            dist_appear.append(dist)
            name=node_2_id_str[k1k2]
            node_2_str_dist[name]=float(dist)



    print ('单独节点和nodesum距离..........')
    ll=sorted(node_str2dist.items(),key=lambda s:s[1],reverse=False)
    print (json.dumps(ll,ensure_ascii=False,indent=6))
    print ('单独边和edgesum距离.........')
    ll = sorted(edge_str2dist.items(), key=lambda s: s[1], reverse=False)
    print(json.dumps(ll, ensure_ascii=False, indent=6))
    print('节点组合和nodesum距离.........')
    ll = sorted(node_2_str_dist.items(), key=lambda s: s[1], reverse=False)
    print(json.dumps(ll, ensure_ascii=False, indent=6))







































