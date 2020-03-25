# coding:utf-8

import os,sys
import logging,json
reload(sys)
sys.setdefaultencoding("utf8")
import json,copy














def get_str_path(dll):
    pathlls = []
    for d in dll:
        pathll=[]
        ####
        for k, v in d.items():
            if v == {}:  # k is leaf,the end
                thispath = [k]
                pathll.append(copy.copy(thispath))
            else:
                for k1, v1 in v.items():
                    if v1 == {}:  # k1 leaf , the end
                        thispath = [k, k1]
                        pathll.append(copy.copy(thispath))
                    else:
                        for k2, v2 in v1.items():
                            if v2 == {}:  # k2 leaf , the end
                                thispath = [k, k1, k2]
                                pathll.append(copy.copy(thispath))
                            else:
                                for k3, v3 in v2.items():
                                    if v3 == {}:  # k3 leaf, the end
                                        thispath = [k, k1, k2, k3]
                                        pathll.append(thispath)
                                    else:
                                        for k4, v4 in v3.items():
                                            if v4 == {}:  # k4 leaf the end
                                                thispath = [k, k1, k2, k3, k4]
                                                pathll.append(thispath)

        ####

        pathlls.append(copy.copy(pathll))
    return pathlls



def restore_dic(pathll):
    dic={}
    for path in pathll:
        for ii in range(len(path)):
            w=path[ii]
            #
            if ii==0:
                if w not in dic:
                    dic[w]={}
            ##
            if ii==1:
                if w not in dic[path[0]]:
                    dic[path[0]][path[1]]={}
            ##
            if ii==2:
                if w not in dic[path[0]][path[1]]:
                    dic[path[0]][path[1]][path[2]]={}
            ##
            if ii==3:
                if w not in dic[path[0]][path[1]][path[2]]:
                    dic[path[0]][path[1]][path[2]][path[3]]={}
    return dic




#####
def combine_sym(dll):
    paths=get_str_path(dll)
    ### extend path
    path_line=[]
    for path in paths:
        for p in path:
            path_line.append(p)
    dic=restore_dic(path_line)
    #####
    retll=[]
    for k,v in dic.items():
        retll.append({k:v})
    return retll


