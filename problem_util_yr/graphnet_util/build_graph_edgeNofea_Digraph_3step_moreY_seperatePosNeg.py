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


#from group_input_position_wangyc import convertTrainable2Symptom
from problem_util_yr.loadDict.read_json_tool import read_json
from problem_util_yr.graphnet_util.encode_str2id import tokenization

from graph_nets import utils_np

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


random.seed(1)



age_reflected={'age_(70,)':'age_(70,)',
               'age_[40,70]':'age_(40,70]',
                'age_(0,15]':'age_(0,15]',
                'age_[0,15]':'age_(0,15]',
                'age_(15,40]':'age_(15,40]',
                'age_[15,40]':'age_(15,40]',
                'age_(40,70]':'age_(40,70]'}




def ingraph_str2id(d):
    if d=={}:
        return d
    else:
        d_={}
        for wstr,v_dic in d.items():
            wstr_id=token.str2id(wstr)
            d_[wstr_id]=ingraph_str2id(v_dic)
        return d_






# def get_node(cell):
#     root_type =[typ for typ in cell.keys() if typ!=u'属性'][0]
#     root_value=cell[root_type]
#     node=root_type+'_'+root_value
#     return node




def get_adj_dictOfList_0(d_id):
    def init_dic(k):
        if k not in node_subnodell_dict:
            node_subnodell_dict[k]=set()



    node_subnodell_dict={}
    for k,d in d_id.items():
        # init
        init_dic(k) # dict must has key :k
        ###
        for k1,d1 in d.items():
            node_subnodell_dict[k].add(k1)
            init_dic(k1)
            for k2,d2 in d1.items():
                node_subnodell_dict[k1].add(k2)
                init_dic(k2)
                for k3,d3 in d2.items():
                    node_subnodell_dict[k2].add(k3)


    ### change set -> list
    empty=[]
    for k,v in node_subnodell_dict.items():
        node_subnodell_dict[k]=list(v)
        if len(v)==0:
            empty.append(k)
    ## del empty key
    # for k in empty:
    #     del node_subnodell_dict[k]
    return node_subnodell_dict




def get_adj_dictOfList(d_id_ll):
    def init_dic(k):
        if k not in node_subnodell_dict:
            node_subnodell_dict[k]=set()



    node_subnodell_dict={}
    for d_id in d_id_ll:
        for k,d in d_id.items():
            # init
            init_dic(k) # dict must has key :k
            ###
            for k1,d1 in d.items():
                node_subnodell_dict[k].add(k1)
                init_dic(k1)
                for k2,d2 in d1.items():
                    node_subnodell_dict[k1].add(k2)
                    init_dic(k2)
                    for k3,d3 in d2.items():
                        node_subnodell_dict[k2].add(k3)


    ### change set -> list
    empty=[]
    for k,v in node_subnodell_dict.items():
        node_subnodell_dict[k]=list(v)
        if len(v)==0:
            empty.append(k)
    ## del empty key
    # for k in empty:
    #     del node_subnodell_dict[k]
    return node_subnodell_dict


def generate_multiGraph_through_path(dll):
    def get_str_path(dll):
        pathll=[]
        for d in dll:
            for k,v in d.items():
                if v=={}: # k is leaf,the end
                    thispath=[k]
                    pathll.append(copy.copy(thispath))
                else:
                    for k1,v1 in v.items():
                        if v1=={}: #k1 leaf , the end
                            thispath=[k,k1]
                            pathll.append(copy.copy(thispath))
                        else:
                            for k2,v2 in v1.items():
                                if v2=={}: # k2 leaf , the end
                                    thispath=[k,k1,k2]
                                    pathll.append(copy.copy(thispath))
                                else:
                                    for k3,v3 in v2.items():
                                        if v3=={}: # k3 leaf, the end
                                            thispath=[k,k1,k2,k3]
                                            pathll.append(thispath)
                                        else:
                                            for k4,v4 in v3.items():
                                                if v4=={}:# k4 leaf the end
                                                    thispath=[k,k1,k2,k3,k4]
                                                    pathll.append(thispath)
        return pathll



    #####
    def path_str2id(pathll):
        pathsid=[]
        for path in pathll:
            pathid=[token1.str2id(w) for w in path]
            pathsid.append(copy.copy(pathid))
        return pathsid

    ####
    # def get_edge_fea(nodePath_):# directGraph [34 56]->nouse   [23 56 34]->[nouse 23]  [34 56 23 14]->[nouse 34 56]
    #     nodePath=copy.copy(nodePath_)
    #     nodePath.insert(0,NOUSEID)
    #     return nodePath[:-2]
    def make_edge_fea(path_idll,G_):
        G=G_.copy()
        for path in path_idll:
            if len(path) == 1:
                G.add_node(path[0])  # node先不加FEATURE 后面 做INPUT TARGET时候在加
            elif len(path) == 2:
                G.add_nodes_from(path)
                fea = dict(features=NOUSEID)
                G = add_edges_multiGraph(path[0], path[1], fea, G)
            elif len(path) > 2:  # 3 4 5
                G.add_nodes_from(path)
                ##第一二个节点的边特征
                fea = dict(features=NOUSEID)
                G = add_edges_multiGraph(path[0], path[1], fea, G)
                ### 之后的节点的边
                for ii in range(len(path))[2:]:
                    u, v = path[ii - 1], path[ii]
                    fea = dict(features=path[ii - 2])
                    G = add_edges_multiGraph(u, v, fea, G)
        return G



    ### init graph
    G = nx.MultiDiGraph()
    ### get str path
    pathll=get_str_path(dll)
    ### str path -> id
    #path_idll=path_str2id(pathll)
    #path_idll=pathll  # 用STR 看  debug
    ### path fea |node fea[21 34 56] |edge fea[nouse 21]
    #G_str=make_edge_fea(pathll,G)
    #G_id=make_edge_fea(path_idll,G)

    #return G_str,pathll
    return G,pathll






def add_symptom_edge(symll,g,fea=None): #### between symptom root has 2 direction edge

    g_=g.copy()
    ll=[]
    for w in symll:
        for w1 in symll:
            if w==w1:continue
            ll.append([w,w1])
            if fea==None:
                g_.add_edge(w,w1)
            if fea!=None:
                g_.add_edge(w,w1,features=fea)

            #g_.add_edge(w,w1)
    return g_,ll

def mulitDiGraph_add_symptom_body_edge_bidirection(g): #症状  部位 双向边
    g_=g.copy()
    useful=[]
    for u, v, keys, fea in g_.edges(data='features', keys=True):
        #if fea!=None:
        useful.append([u,v,fea])
    # for sym,body in symptom_body_pair:
    #     if g_.has_edge(sym,body)==False:
    #         g_.add_edge(sym, body)
    #     if g_.has_edge(body, sym)==False:
    #         g_.add_edge(body, sym)
    ### 单项 -> 双向
    for u,v,_ in useful:
        if u'临床表现' in u and u'部位' in v:
            g_.add_edge(v,u)

    return g_

def DiGraph_add_symptom_body_edge_bidirection(g): #症状  部位 双向边
    g_=g.copy()
    useful=[]
    for u, v,  fea in g_.edges(data='features'):
        #if fea!=None:
        useful.append([u,v,fea])
    # for sym,body in symptom_body_pair:
    #     if g_.has_edge(sym,body)==False:
    #         g_.add_edge(sym, body)
    #     if g_.has_edge(body, sym)==False:
    #         g_.add_edge(body, sym)
    ### 单项 -> 双向
    for u,v,_ in useful:
        if u'临床表现' in u and u'部位' in v:
            g_.add_edge(v,u)
        elif u'部位' in u and u'临床表现' in v:
            g_.add_edge(v, u)

    return g_





def make_input_target_leafRemove(g,fixnum=1):

    target = g.copy()# dont change g
    input = g.copy()

    #### get all leaf ,nodes with only 1 edge
    leafll=[]
    for node in input.nodes:
        #num_edge_thisNode=input.degree[node]
        num_edge_thisNode=len(input.adj[node])
        if num_edge_thisNode==1:
            leafll.append(node)
    ####

    sampled=[leafll[0]]






    input.remove_nodes_from(sampled)
    return input,target


# def add_nodename(g):
#     g_=g.copy()
#     #g_=nx.Graph(g) #shallow copy
#     nodes=g.nodes
#     featurename='symptom_name'
#     for node in nodes:
#         nx.set_node_attributes(g_,name=featurename,values={node:token.id2str(node)})
#     return g_



def add_edge_feature(graph,fea,force_replace_fea=False):# 根据已有的 NODE  EDGE  提供[1 0] feature False : global symptom
    def check():
        for receiver, sender, features in graph.edges(data=True):
            print ('')  # features={xx:,xx:,features:[0.,1.]}

    # ### add edge
    # for receiver, sender, features in graph.edges(data=True):
    #     graph.add_edge(
    #         receiver,sender,features=feature)

    ### set edge
    for receiver, sender, features in graph.edges(data=True):
        if force_replace_fea==False: #不强行替换特征
            if len(features)!=0:#已经有 特征 不替换
                continue

        nx.set_edge_attributes(graph, {(receiver, sender): {'features': fea}})
        # global_graphnet.add_edge(receiver,sender,features=features['features'])

    ### check
    #check()

    return graph

def add_node_feature(graph,fea):# 根据已有的 NODE  EDGE  提供[1 0] feature False : global symptom
    ### add node
    #feature=np.array([1.,0.]) if feature==True else np.array([0.,1.])
    # for node_index, node_feature in graph.nodes(data=True):
    #     #nx.set_node_attributes( )
    #     graph.add_node(node_index,features=feature)


    #### set node
    for node_index, node_feature in graph.nodes(data=True):# {} or {'features':xxx}
        if len(node_feature)!=0: #已经有 特征 不替换
            continue
        nx.set_node_attributes(graph,{node_index:fea},'features')

    return graph


def add_singleNode_feature(graph,fea,node_index,k='features'):
    nx.set_node_attributes(graph, {node_index: fea}, k)
    return graph



def padding_node_to_fixlen(graph_):
    graph=graph_.copy()
    tt_node=set(range(MAXPAD))
    this_g_node=set(graph.nodes)

    diff = tt_node-this_g_node
    for ii in diff:
        #graph.add_node(ii + num_node_x, features=np.zeros((VOCABSZ)))
        graph.add_node(ii)
    return graph


def add_edge_for_all_node(graph,node_line_to_self=False):
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            if node_line_to_self==False and node1==node2:
                continue

            graph.add_edge(node1,node2)
    return graph



def remove_isolated_node(graph_): # node without neigbours = isolated node
    graph=graph_.copy()
    isolated=list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    return graph



def add_edges_multiGraph(node1,node2,fea,G):
    #G=G_.copy()
    # if G.has_edge(node1,node2):
    #     G[node1][node2][0]
    # else:
    G.add_edges_from([(node1, node2, fea)])
    return G


def multi_remove_edge_without_att0(G):
    useful=[]
    G_=nx.MultiDiGraph()
    for u, v, keys, fea in G.edges(data='features', keys=True):
        if fea!=None:
            useful.append([u,v,fea])
    ####
    for u,v,f in useful:
        G_.add_edges_from([(u,v, dict(features=f))])
    ###
    for node_index, node_feature in G.nodes(data=True):# {} or {'features':xxx}
        nx.set_node_attributes(G_,{node_index:node_feature['features']},'features')

    return G_


def multi_remove_edge_without_att(G):
    notuseful=[]

    for u, v, keys, fea in G.edges(data='features', keys=True):
        if fea==None:
            notuseful.append([u,v,keys])
    ####
    for u,v,k in notuseful:
        G.remove_edge(u,v,k)


    return G


# def relabel_unique_node(dict_id):
#     uniqueId_rawId_map={}
#     dict_id_unique={}
#
#     for d,v in dict_id.items():
#         uniqueId=len(uniqueId_rawId_map)
#         uniqueId_rawId_map[uniqueId]=d
#         dict_id_unique[uniqueId]={}
#         for d1,v1 in v.items():
#             uniqueId1 = len(uniqueId_rawId_map)
#             uniqueId_rawId_map[uniqueId1] = d1
#             dict_id_unique[uniqueId][uniqueId1] = {}
#             for d2, v2 in v1.items():
#                 uniqueId2 = len(uniqueId_rawId_map)
#                 uniqueId_rawId_map[uniqueId2] = d2
#                 dict_id_unique[uniqueId][uniqueId1][uniqueId2] = {}
#                 for d3, v3 in v2.items():
#                     uniqueId3 = len(uniqueId_rawId_map)
#                     uniqueId_rawId_map[uniqueId3] = d3
#                     dict_id_unique[uniqueId][uniqueId1][uniqueId2][uniqueId3] = {}
#     return dict_id_unique,uniqueId_rawId_map

# def relabel_unique_node_withstr(dict_id):
#
#     dict_id_unique={}
#
#     iid=-1
#
#     for d,v in dict_id.items():
#         iid+=1
#         d+='#'+str(iid)
#
#
#         dict_id_unique[d ]={}
#         for d1,v1 in v.items():
#             iid += 1
#             d1+='#'+str(iid)
#             #uniqueId1 = len(uniqueId_rawId_map)
#             #uniqueId_rawId_map[uniqueId1] = d1
#             dict_id_unique[d][d1] = {}
#             for d2, v2 in v1.items():
#                 iid += 1
#                 d2+='#'+str(iid)
#                 #uniqueId2 = len(uniqueId_rawId_map)
#                 #uniqueId_rawId_map[uniqueId2] = d2
#                 dict_id_unique[d][d1][d2] = {}
#                 for d3, v3 in v2.items():
#                     iid += 1
#                     d3+='#'+str(iid)
#                     #uniqueId3 = len(uniqueId_rawId_map)
#                     #uniqueId_rawId_map[uniqueId3] = d3
#                     dict_id_unique[d][d1][d2][d3] = {}
#     return dict_id_unique




# def remove_leaf(dic_,del_node):
#     dicx=copy.deepcopy(dic_)
#     dicy=copy.deepcopy(dic_)
#     #notyet_del_flag=False
#     node_num=-1
#
#
#     for k,v in dicx.items():
#         node_num+=1
#
#         if node_num==del_node:
#             del dicx[k]
#             dicy[k]={}
#             break
#         else:
#             for k1,v1 in v.items():
#                 node_num += 1
#                 if node_num==del_node:
#                     del dicx[k][k1]
#                     dicy[k][k1]={}
#                     break
#                 else:
#                     for k2, v2 in v1.items():
#                         node_num += 1
#                         if node_num == del_node:
#                             del dicx[k][k1][k2]
#                             dicy[k][k1][k2] = {}
#                             break
#                         else:
#                             for k3, v3 in v2.items():
#                                 node_num += 1
#                                 if node_num == del_node:
#                                     del dicx[k][k1][k2][k3]
#                                     dicy[k][k1][k2][k3] = {}
#                                     break
#
#
#
#
#
#     return dicx,dicy







def how_many_node(d_):
    global tt_num
    tt_num=0
    def iter_node(d):
        global tt_num
        for k,v in d.items():
            tt_num+=1
            if len(v)>0:
                iter_node(v)
    ###
    iter_node(d_)
    #print tt_num
    return tt_num


def see_all_node(d_):
    global nodeset
    nodeset=set()

    def iter_node(d):
        global nodeset
        for k,v in d.items():
            nodeset.add(k)
            if len(v)>0:
                iter_node(v)
    ###
    iter_node(d_)

    return nodeset




def remove_node(g_id,sampled):
    g_id1=g_id.copy()
    g_id1.remove_nodes_from(sampled)
    return g_id1



def relabel_node(G):
    num=G.number_of_nodes()
    nodell=G.nodes()
    nodell=sorted(nodell) #否则每次都不一样 xgraph ygraph
    dic=dict(zip(nodell,range(num)))
    H = nx.relabel_nodes(G, dic)
    return H

def relabel_node2(G):
    num=G.number_of_nodes()
    nodell=G.nodes()
   #nodell=sorted(nodell) #否则每次都不一样 xgraph ygraph
    dic=dict(zip(nodell,range(num)))
    H = nx.relabel_nodes(G, dic)
    return H

def relabel_node3(g,n_node_accumu):
    node_id_ll = list(g.nodes())
    to_node_id = [nid + n_node_accumu for nid in node_id_ll]
    dic_relabel = dict(zip(node_id_ll, to_node_id))
    H = nx.relabel_nodes(g, dic_relabel)
    return H

def update_nodedict(nodeDict,nodes):
    for node in nodes:
        ## init
        if node not in nodeDict:
            nodeDict[node]=0
        ####
        nodeDict[node]+=1
    return nodeDict




def get_predecessor(G,node):
    gene = G.predecessors(node)
    return [p for p in gene]

def get_successor(G,node):
    gene = G.successors(node)
    return [p for p in gene]


 
def update_age_gender(age,ageSet,genderSet):
    for p in age:
        p=p.lower()
        if 'age' in p:
            ageSet.add(p)
        elif 'gender' in p:
            genderSet.add(p)
    return ageSet,genderSet



def del_empty_node(g_id_):
    nodes=g_id_.nodes
    nodes=[n for n in nodes]
    for node in nodes:
        if len(node.split('_')[1])==0:
            g_id_.remove_node(node)
    return g_id_


def get_isolated_set(H):
    ll = list(nx.connected_components(H.to_undirected()))
    return ll


def del_invalid_component(G_):
    G=G_.copy()
    ## 取 分散的子集
    ll=get_isolated_set(G)
    ## 取 invalid小图的节点
    invalid_node=set()

    for ss in ll:
        ### 没有 临床表现  为无效
        prefixs=[p.split('_')[0] for p in ss]
        if u'临床表现' not in prefixs:
            invalid_node=invalid_node.union(ss)
        ### 子集 里节点只1个  无效
        if len(ss)==1:
            invalid_node = invalid_node.union(ss)



    ### del node
    for node in invalid_node:
        G.remove_node(node)
    return G




def get_symptom_node(g_id):
    nodes=g_id.nodes
    ret=set()
    for node in nodes:
        if u'临床表现' in node:
            ret.add(node)
    ###
    return ret



def get_age_gender_seperate(ageGender):
    age,gender=None,None
    for s in ageGender:

        if 'age' in s:
            age=s.lower()
        elif 'gender' in s:
            gender=s.lower()
    return age,gender



def multiGraph_edge_feature_str2id(G):
    G_=G.copy()
    for u, v, keys, fea in G.edges(data='features', keys=True):
        #print u, v, keys, fea
        if type(fea) !=int:
            G_[u][v][keys]['features'] = token1.str2id(fea)
    return G_


def multiGraph_add_edgeFeature_from_edge(G):
    G_=G.copy()
    for u, v, keys, fea in G.edges(data='features', keys=True):
        #print u, v, keys, fea
        #if fea==None:
        G_[u][v][keys]['features'] = token1.str2id(fea)
    return G_



def multiGraph_add_edge_fea(pathll,G):
    nodes=set(G.nodes)

    #print ''
    for path in pathll:

        if len(path) == 1: continue
        if set(path)&nodes!=set(path):
            continue
        ###
        if len(path) == 2:
            fea = dict(features=NOUSE_EDGE_ID)
            G = add_edges_multiGraph(path[0], path[1], fea, G)
        elif len(path) > 2:  # 3 4 5
            ##第一二个节点的边特征
            fea = dict(features=NOUSE_EDGE_ID)
            G = add_edges_multiGraph(path[0], path[1], fea, G)
            ### 之后的节点的边
            for ii in range(len(path))[2:]:
                u, v = path[ii - 1], path[ii]
                fea = dict(features=path[ii - 2])
                G = add_edges_multiGraph(u, v, fea, G)

    ####
    G=multi_remove_edge_without_att(G)
    return G


def get_graph_tuple_from_g(gll):
    return utils_np.networkxs_to_graphs_tuple(gll)


def procedure_add_group_sym_bottom2top(allpaths,rootnode,symrootll):

    g_str = nx.DiGraph()  # 有边  节点  没有 特征
    for path in allpaths:
        g_str.add_path(path)
    num_node = g_str.number_of_nodes()
    ## [ 症状 症状 ]之间 不 互相连
    ##  [ 症状 部位 ]互相长出

    ## 找到第一层症状根节点
    symll=symrootll


    ##
    ##  [ 症状 部位 ]
    g_str1 = DiGraph_add_symptom_body_edge_bidirection(g_str)



    ####   是否有临床表现节点

    node_layer1=symll
    has_sym=False
    for node in node_layer1:
        if u'临床表现' in node:
            has_sym=True
            break
    if has_sym==False:
        return None


    #  连接 [总体 根节点_肯定症状 , 症状]
    g_str1.add_node(rootnode)
    for sym in symll:
        g_str1.add_path([sym, rootnode])
    return g_str1

def procedure_add_group_sym(dll,rootnode):
    dict_of_list = get_adj_dictOfList(dll)  # 3 layer deep dict -> adj dict of list  2 layer 用ADJ只有2个节点信息 丢失3个节点信息
    g_str = nx.DiGraph(dict_of_list)  # 有边  节点  没有 特征
    num_node = g_str.number_of_nodes()
    ## [ 症状 症状 ]之间 不 互相连
    ##  [ 症状 部位 ]互相长出

    ## 找到第一层症状根节点
    symll=[]
    for d in dll:
        for k in d:
            symll.append(k)


    ##
    ##  [ 症状 部位 ]
    g_str1 = DiGraph_add_symptom_body_edge_bidirection(g_str)



    ####   是否有临床表现节点

    node_layer1=symll
    has_sym=False
    for node in node_layer1:
        if u'临床表现' in node:
            has_sym=True
            break
    if has_sym==False:
        return None


    #  连接 [总体 根节点_肯定症状 , 症状]
    g_str1.add_node(rootnode)
    for sym in symll:
        g_str1.add_path([rootnode, sym])
    return g_str1

def procedure_add_group_sym0(dll,rootnode):
    ################
    ####### 区别 部位和根节点连
    ####################
    dict_of_list = get_adj_dictOfList(dll)  # 3 layer deep dict -> adj dict of list  2 layer 用ADJ只有2个节点信息 丢失3个节点信息
    g_str = nx.DiGraph(dict_of_list)  # 有边  节点  没有 特征
    num_node = g_str.number_of_nodes()
    ## [ 症状 症状 ]之间 不 互相连
    ##  [ 症状 部位 ]互相长出

    ## 找到第一层症状根节点
    symll=[]
    for d in dll:
        for k in d:
            symll.append(k)
    #### 找到 部位
    bodyll=[]
    for node_index, node_feature in g_str.nodes(data=True):
        if u'部位' in node_index:
            bodyll.append(node_index)



    ##
    ##  [ 症状 部位 ]
    g_str1 = DiGraph_add_symptom_body_edge_bidirection(g_str)



    ####   是否有临床表现节点

    node_layer1=symll
    has_sym=False
    for node in node_layer1:
        if u'临床表现' in node:
            has_sym=True
            break
    if has_sym==False:
        return None


    #  连接 [总体 根节点_肯定症状 , 症状]
    g_str1.add_node(rootnode)
    for sym in symll:
        g_str1.add_path([rootnode, sym])
    #  连接 [总体 根节点_肯定症状 , 部位]
    for body in bodyll:
        g_str1.add_path([rootnode, body])
    return g_str1


def procedure_add_group_sym1(dll,rootnode):
    ##################
    ###### 区别 连接[根节点_肯定症状, 症状]   双向
    ##################
    dict_of_list = get_adj_dictOfList(dll)  # 3 layer deep dict -> adj dict of list  2 layer 用ADJ只有2个节点信息 丢失3个节点信息
    g_str = nx.DiGraph(dict_of_list)  # 有边  节点  没有 特征
    num_node = g_str.number_of_nodes()
    ## 症状  根节点  双向边
    ##  [ 症状 部位 ]
    g_str = DiGraph_add_symptom_body_edge_bidirection(g_str)

    ## 找到第一层症状根节点
    symll=[]
    for d in dll:
        for k in d:
            symll.append(k)

    ####



    ####   是否有临床表现节点
    #nodes=g_str1.nodes
    node_layer1=symll
    has_sym=False
    for node in node_layer1:
        if u'临床表现' in node:
            has_sym=True
            break
    if has_sym==False:
        return None


    #  连接 [  根节点_肯定症状 , 症状] 双向
    g_str.add_node(rootnode)
    for sym in symll:
        g_str.add_path([rootnode, sym])
        g_str.add_path([sym, rootnode])
    return g_str


def procedure_add_group_sym11(dll,rootnode):
    ##################
    ###### 区别 所有边 双向
    ##################
    dict_of_list = get_adj_dictOfList(dll)  # 3 layer deep dict -> adj dict of list  2 layer 用ADJ只有2个节点信息 丢失3个节点信息
    g_str = nx.Graph(dict_of_list)  # 有边  节点  没有 特征
    num_node = g_str.number_of_nodes()


    ## 找到第一层症状根节点
    symll=[]
    for d in dll:
        for k in d:
            symll.append(k)

    ####   是否有临床表现节点
    #nodes=g_str1.nodes
    node_layer1=symll
    has_sym=False
    for node in node_layer1:
        if u'临床表现' in node:
            has_sym=True
            break
    if has_sym==False:
        return None


    #  连接 [  根节点_肯定症状 , 症状] 双向
    g_str.add_node(rootnode)
    for sym in symll:
        g_str.add_path([rootnode, sym])
        g_str.add_path([sym, rootnode])
    return g_str



def get_rootnode_for_subgraph(notNoneGraphll):
    rootnode_list = []
    for gstr in notNoneGraphll:
        nodes = gstr.nodes
        for n in nodes:
            if u'根节点' in n:
                rootnode_list.append(n)
    return rootnode_list


def get_grow_to_node(g_str,low_freq_node):
    if g_str==None:return None
    x_y_node_dict = {}
    for send, receive, features in g_str.edges(data=True): # sendNode -> receiveNode
        # init
        if send not in x_y_node_dict:
            x_y_node_dict[send] = []
        ###
        if u'根节点' in receive:continue # 不要 [否定根节点 -> 肯定根节点]  [肯定根节点 -> 否定根节点]
        x_y_node_dict[send].append(receive)
    ### x:[y y y ] -> x:[y]
    x_y_pair = {}


    for send, receives in x_y_node_dict.items():
        #
        x_y_pair[send]=[]
        #
        if len(receives) > 1: # 1 node ,grow to several nodes
            ## 找到低频 词
            for w in receives:
                if w in low_freq_node:  # 低频 节点
                    x_y_pair[send].append(w)

            #####没有低频词
            if x_y_pair[send].__len__()==0:
                y = random.sample(receives, 1)[0]
                x_y_pair[send] = [y]
            ####  大于1个低频词,(1个低频次不用处理)
            elif x_y_pair[send].__len__() >1:
                y = random.sample(x_y_pair[send], 1)[0]
                x_y_pair[send] = [y]



        elif len(receives) == 1 :
            x_y_pair[send]=receives

    return x_y_pair




def sort_node_input_graph(graph3): # py3  dict won't sort by key 0 1 2 3...
    ## py3生成的图  node dict没有按照 node_index 排序，导致边是错的
    #### 重新排序 节点
    g_1 = nx.DiGraph()
    # get node fea
    ndic = {}
    for node_index, node_feature in graph3.nodes(data=True):
        ndic[node_index] = node_feature['features']

    ## sort by node id
    ll = sorted(ndic.items(), key=lambda s: s[0])
    for node, fea in ll:
        g_1.add_node(node)
        nx.set_node_attributes(g_1, {node: fea}, 'features')
    ##
    g_1.add_edges_from(g3.edges)

    for receiver, sender, features in g3.edges(data=True):
        nx.set_edge_attributes(g_1, {(receiver, sender): features})

    print('')
    g_1.graph['features'] = [0]




def procedure_make_group_basesign(basesign,rootnode):
    gstr=nx.DiGraph()
    ## 生命体征  互相连
    # for bs in basesign:
    #     for bs1 in basesign:
    #         if bs!=bs1:
    #             gstr.add_path([bs,bs1])

    #### 根节点 和各个生命体征连
    for bs in basesign:
        gstr.add_path([rootnode, bs])
    return gstr

def procedure_make_group_basesign1(basesign,rootnode):
    gstr=nx.DiGraph()
    ####  各个生命体征  -> 根节点
    for bs in basesign:
        gstr.add_path([bs,rootnode])
    return gstr




def procedure_add_group_signal(dll,rootnode):
    dict_of_list = get_adj_dictOfList(dll)  # 3 layer deep dict -> adj dict of list  2 layer 用ADJ只有2个节点信息 丢失3个节点信息
    g_str = nx.DiGraph(dict_of_list)  # 有边  节点  没有 特征
    num_node = g_str.number_of_nodes()

    ### get root node
    symll = []
    for d in dll:
        for k in d:
            symll.append(k)


    #  [根节点_肯定体格检查  体格检查]
    g_str.add_node(rootnode)
    for root in symll:
        g_str.add_path([rootnode, root])
    if g_str.number_of_nodes()*g_str.number_of_edges()==0:
        return None
    return g_str





def add_prefix_for_node(g_str,word):
    # 根节点 跳过
    mapn={}
    for node in g_str:
        if u'根节点' in node: continue
        # mapn[node] = u'症状_' + node
        mapn[node] = word + node
    ###
    g_str=nx.relabel_nodes(g_str,mapn)
    return g_str















