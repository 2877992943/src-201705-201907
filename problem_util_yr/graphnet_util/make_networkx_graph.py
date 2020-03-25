# coding:utf-8

import os,sys
import logging,json

import platform
version_python=platform.python_version()
if version_python.startswith('2'):

    reload(sys)
    sys.setdefaultencoding("utf8")
import json,copy
import networkx as nx

import random
import pandas as pdd
import numpy as np

from itertools import groupby
from operator import itemgetter

###### used for networkx 2.4





#
# def add_edge_feature(graph,fea,force_replace_fea=False):# 根据已有的 NODE  EDGE  提供[1 0] feature False : global symptom
#     def check():
#         for receiver, sender, features in graph.edges(data=True):
#             print ('')  # features={xx:,xx:,features:[0.,1.]}
#
#     # ### add edge
#     # for receiver, sender, features in graph.edges(data=True):
#     #     graph.add_edge(
#     #         receiver,sender,features=feature)
#
#     ### set edge
#     for receiver, sender, features in graph.edges(data=True):
#         if force_replace_fea==False: #不强行替换特征
#             if len(features)!=0:#已经有 特征 不替换
#                 continue
#
#         nx.set_edge_attributes(graph, ,{(receiver, sender): {'features': fea}})
#         # global_graphnet.add_edge(receiver,sender,features=features['features'])
#
#     ### check
#     #check()
#
#     return graph





# def add_singleNode_feature(graph,fea,node_index,k='features'):
#     nx.set_node_attributes(graph, {node_index: fea}, k)
#     return graph


def add_singleNode_feature(graph,fea,node_index,k='features'):
    nx.set_node_attributes(graph, k, {node_index:fea})
    return graph


def relabel_node(G):
    num=G.number_of_nodes()
    nodell=G.nodes()
   #nodell=sorted(nodell) #否则每次都不一样 xgraph ygraph
    dic=dict(zip(nodell,range(num)))
    H = nx.relabel_nodes(G, dic)
    return H


if __name__=='__main__':
    g=nx.DiGraph()
    ## add node
    g.add_nodes_from([1,2,3,4,5])
    ## add edge
    g.add_edges_from([(1,2),(1,3),(1,5)])
    adj=g.adj[1]
    ## add node fea
    g.nodes[1]['features']=[34]
    d=g.nodes.data()
    d=list(d)

    ## add edge fea
    g.edges[1,2]['features']=[0]
    d=g.edges.data()
    d=list(d)
    ## add global fea
    g.graph['features'] = []
    print ('')

