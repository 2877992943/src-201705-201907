# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import sys
sys.path.append('/home/yangrui/')
print sys.path

import os,json
import pandas as pdd

def get_lab_word_att(root):
    return root.get('labelId'), root.get('word'), root.get('att')


def jsonDict_filter(rootll,text,en2cn_dict):
    def whether_continue(root='not filled',word='not filled',labelId='not filled'):
        if root!='not filled': # fill root
            if root==None:return True
        else: # fill word labelId
            if None in [word, labelId]:return True
            if word.strip()=='':return True
            if word not in text:return True #预测生成的词  不在原文中
        return False




    all_ll=[]


    for root in rootll:
        #if root==None:continue
        flag=whether_continue(root)
        if flag:continue
        labelId,word,att=get_lab_word_att(root)

        # if None in [word,labelId]:continue
        # if word.strip()=='':continue
        flag=whether_continue(word=word,labelId=labelId)
        if flag:continue
        if att==[]:
            all_ll.append({'word':word,'labelId':en2cn_dict[labelId],'att':[]})
        elif att!=[]:
            att_=[]
            for root1 in att:
                #if root1 == None: continue
                flag=whether_continue(root1)
                if flag:continue
                labelId1,word1,att1=get_lab_word_att(root1)
                # if None in [word1, labelId1]: continue
                # if word1.strip() == '': continue
                flag=whether_continue(word=word1,labelId=labelId1)
                if flag:continue
                if att1==[]:
                    att_.append({'word':word1,'labelId':en2cn_dict[labelId1],'att':[]})
                elif att1!=[]:
                    att1_=[]
                    for root2 in att1:
                        #if root2 == None: continue
                        flag=whether_continue(root2)
                        if flag:continue
                        labelId2, word2, att2 = get_lab_word_att(root2)
                        # if None in [word2, labelId2]: continue
                        # if word2.strip() == '': continue
                        flag=whether_continue(word=word2,labelId=labelId2)
                        if flag:continue
                        if att2==[]:
                            att1_.append({'word': word2, 'labelId': en2cn_dict[labelId2], 'att': []})
                        elif att2!=[]:
                            att2_=[]
                            for root3 in att2:
                                #if root3 == None: continue
                                flag=whether_continue(root3)
                                if flag:continue
                                labelId3, word3, att3 = get_lab_word_att(root3)
                                # if None in [word3, labelId3]: continue
                                # if word3.strip() == '': continue
                                flag=whether_continue(word=word3,labelId=labelId3)
                                if flag:continue
                                if att3==[]:
                                    att2_.append({'word': word3, 'labelId': en2cn_dict[labelId3], 'att': []})

                            att1_.append({'word': word2, 'labelId': en2cn_dict[labelId2], 'att': att2_})


                    att_.append({'word': word1, 'labelId': en2cn_dict[labelId1], 'att': att1_})



            all_ll.append({'word': word, 'labelId': en2cn_dict[labelId], 'att': att_})


    return all_ll






def jsonDict_to_treeline(rootll):
    def label_add(lab): # xxx-> xxx_lab
        #return lab+'_lab'
        return lab





    all_ll=[]


    for root in rootll:
        #if root==None:continue


        labelId,word,att=get_lab_word_att(root)
        labelId=label_add(labelId)




        if att==[]:
            all_ll+=[labelId,word,'//'+labelId]
        elif att!=[]:
            att_=[]
            all_ll += [labelId, word]
            for root1 in att:

                labelId1,word1,att1=get_lab_word_att(root1)
                labelId1 = label_add(labelId1)


                if att1==[]:
                    all_ll += [labelId1, word1, '//' + labelId1]
                elif att1!=[]:
                    all_ll += [labelId1, word1]
                    att1_=[]
                    for root2 in att1:

                        labelId2, word2, att2 = get_lab_word_att(root2)
                        labelId2 = label_add(labelId2)


                        if att2==[]:
                            all_ll += [labelId2, word2, '//' + labelId2]
                        elif att2!=[]:
                            all_ll += [labelId2, word2]
                            att2_=[]
                            for root3 in att2:

                                labelId3, word3, att3 = get_lab_word_att(root3)
                                labelId3 = label_add(labelId3)

                                if att3==[]:
                                    all_ll += [labelId3, word3, '//' + labelId3]

                            all_ll +=['//' + labelId2]

                    all_ll += ['//' + labelId1]

            all_ll += ['//' + labelId]


    return all_ll






def get_reason_from_treeline(treelinell):
    reasonll=[]
    otherll=[]
    ii=0
    while ii <len(treelinell): #len 5 01234

        elem=treelinell[ii]
        if elem==u'前置条件':
            reasonll.append(treelinell[ii+1])
            ii+=3
        else:
            otherll.append(elem)
            ii+=1
    return reasonll,otherll







if __name__=='__main__':


    ###### load dict
    from problem_util_yr.loadDict.label_and_CN import get_LABEL_NEEDED_this_problem
    en2cn_dict=get_LABEL_NEEDED_this_problem(u'翻译.txt')


    ##
    rootll=[{"att": [{"att": [{"att": [], "word": u"左侧", "labelId": "feature8"}], "word": u"胸腔", "labelId": "feature5"}], "word": u"膨胀", "labelId": "feature1"}]
    text=u'左侧胸腔积液并左肺下叶局限性膨胀不全'

    # 过滤
    jsonDict_filter(rootll,text,en2cn_dict)
    # {} 变成  lab xxx //lab
    jsonDict_to_treeline(rootll)















