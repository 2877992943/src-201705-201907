#coding:utf-8
#将ann格式和ptb格式互换，并能找到ann中的错误标注
import sys, os
import scanannpath
import logging
import tensorflow as tf
import xlrd
import json
import copy
reload(sys)
sys.setdefaultencoding('utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/ann2ptb.log',
                    filemode='w')
#from predict2annotation import devide_prediction_by_documentsize

def remove_empty(stri):
    for n in  [' ',u'\xa0']:
        stri=stri.replace(n,'')
    return stri

class TreeNode(object):

    def __init__(self, line):
        line = line.strip()
        self.sons = []
        if line.startswith(u'T'):
            pos1 = line.find(u'\t')
            pos2 = line.find(u'\t', pos1+1)
            info = line[pos1+1: pos2]
            if u' ' not in info:
                #没有空格的annotation格式模式
                self.name = line[:pos1]
                self.pos = line[pos1+1: pos2]
                pos1 = line.find(u'\t', pos2+1)
                self.start = int(line[pos2+1: pos1])
                pos2 = line.find(u'\t', pos1 + 1)
                self.end = int(line[pos1+1: pos2])
                self.set_text(line[pos2+1:])
            else:
                self.name = line[:pos1]
                self.set_text(line[pos2+1:])
                items = info.split(u' ')
                self.pos = items[0]
                self.start = int(items[1])
                self.end = int(items[2])
        else:
            pos1 = line.find(u'\t')
            self.name = line[:pos1]
            info = line[pos1+1:]
            pos1 = info.find(u' ')
            pos2 = info.find(u' ', pos1+1)
            self.pos = info[:pos1]
            items1 = info[pos1+1: pos2].split(u':')
            self.relation_son = items1[1]
            items2 = info[pos2+1:].split(u':')
            self.relation_father = items2[1]

    def set_text(self, text):
        if u'\\' in text:
            text = text.replace(u'\\', u'\\\\')
        if u'(' in text:
            text = text.replace(u'(', u'\(')
        if u')' in text:
            text = text.replace(u')', u'\)')
        self.text = text

    def get_text(self):
        return self.text

    def set_start(self, start):
        self.start = start

    def get_start(self):
        return self.start

    def set_end(self, end):
        self.end = end

    def get_end(self):
        return self.end

    def get_pos(self):
        return self.pos

    def add_son(self, son):
        self.sons.append(son)

    @staticmethod
    def sort_nodes(nodes):
        try:
            dct = {x.get_start():x for x in nodes}
        except Exception:
            print(u'debug')
        return [dct[x] for x in sorted(dct.keys())]

    def get_sons(self):
        return self.sons

    def get_relation_father(self):
        return self.relation_father

    def get_relation_son(self):
        return self.relation_son

    def to_ptb(self):
        try:
            t=self.get_text()
            node_line = remove_empty(t) + u' ' + self.pos
        except:
            print ''
        if len(self.sons) == 0:
            return node_line
        else:
            self.sons = TreeNode.sort_nodes(self.sons)
            for son in self.sons:
                node_line += u' ' + son.to_ptb()
            return u'(' + node_line + u')'

    def has_son_pos(self, pos):
        for son in self.get_sons():
            if son.get_pos() == pos:
                return True
        return False


def get_chunk_unlabel(text,annotations):
    num=len(text.decode('utf-8'))
    chunk=['unlabel' for i in range(num)]
    for line in annotations:
        try:
            if line.startswith('T'):
                line=line.split('\t')
                _,lab,s,e,content=line
                s=int(s)
                e=int(e)
                chunk[s:e]=[lab]*(e-s)
            else:
                continue
        except:
            continue
            #print ''

    return chunk


class SentenceTree(object):
    #处理一个句子，可以读入annotation或者PTB，也可以输出annotation或者PTB格式，从而完成转换

    def read_ann(self, text, annotations):
        self.root = []
        self.text = text
        self.node_dict = {}
        self.chunk=get_chunk_unlabel(text,annotations)
        for item in annotations:
            node = TreeNode(item)
            self.node_dict[node.name] = node
        self.cover_relation()

    def cover_relation(self):
        son_nodes = set()
        for key in self.node_dict:
            if key.startswith(u'R'):
                relation = self.node_dict[key]
                relation_father_name = relation.get_relation_father()
                relation_son_name = relation.get_relation_son()
                father = self.node_dict[relation_father_name]
                son = self.node_dict[relation_son_name]
                father.add_son(son)
                son_nodes.add(relation_son_name)
        for item in son_nodes:
            self.node_dict.pop(item)

    def to_ptb(self, root_pos):
        out_line = u''
        sort_father_node = []
        error_root = False
        error_words = []
        for key in self.node_dict:
            if key.startswith(u'T'):
                if self.node_dict[key].get_pos() != u'other':
                    sort_father_node.append(self.node_dict[key])
        sort_father_node = TreeNode.sort_nodes(sort_father_node)
        for item in sort_father_node:
            item_pos = item.get_pos()
            if item_pos not in root_pos:
                error_root = True
                error_words.append(remove_empty(item.get_text()))
            item_to_ptb=item.to_ptb()
            out_line += u'[' + item_to_ptb + u']\n'

        return out_line, error_root, u','.join(error_words)

    def to_ptb_signal(self, root_pos, white_list):
        out_line = u''
        sort_father_node = []
        error_root = False
        error_words = []
        for key in self.node_dict:
            if key.startswith(u'T'):
                if self.node_dict[key].get_pos() != u'other':
                    sort_father_node.append(self.node_dict[key])
        sort_father_node = TreeNode.sort_nodes(sort_father_node)
        for item in sort_father_node:
            item_pos = item.get_pos()
            if item_pos not in root_pos:
                error_root = True
                error_words.append(item.get_text())
            elif item_pos == u'exam_result':
                if item.get_text() not in white_list:
                    # 检查结果结果必须有检查项目做必须有检查项目做为子节点
                    if not item.has_son_pos(u'exam_project'):
                        error_root = True
                        error_words.append(item.get_text())
            out_line += u'[' + item.to_ptb() + u']\n'

        return out_line, error_root, u','.join(error_words)


def ann2ptb_signal(input_path, out_file, root_pos, white_list):
    #将ann文件转换成ptb格式
    text_list, annotation_list, source_list, no_annotation_text, no_annotation_source = scanannpath.read_path(input_path)
    writer = open(out_file, u'w')
    for i in xrange(len(text_list)):
        print(i)
        if i == 1919:
            print(u'debug')
        text = text_list[i]
        annotation = annotation_list[i]
        source = source_list[i]
        st = SentenceTree()
        try:
            st.read_ann(text, annotation)
        except Exception:
            print(text)
        # print (st.text)
        ptb_line, error, error_words = st.to_ptb_signal(root_pos, white_list)
        if error:
            logging.info(source + u' error_words: ' + error_words)
        else:
            writer.write(u'sentence:' + text)
            writer.write(ptb_line + u'\n')


def ann2ptb(input_path, out_file, root_pos,out_file_chunk=None):
    text_list_ret=[]
    #将ann文件转换成ptb格式
    text_list, annotation_list, source_list, no_annotation_text, no_annotation_source = scanannpath.read_path(input_path)
    writer = open(out_file, u'w')
    if out_file_chunk:
        writer_chunk=open(out_file_chunk,'w')
    for i in xrange(len(text_list)):
        print(i)
        if i == 1919:
            print(u'debug')
        text = text_list[i];
        text=remove_empty(text)
        annotation = annotation_list[i]
        source = source_list[i]
        st = SentenceTree()
        try:
            st.read_ann(text, annotation)
            #print ''
        except Exception:
            print(text)
        # print (st.text)
        ptb_line, error, error_words = st.to_ptb(root_pos)
        if error:
            logging.info(source + u' error_words: ' + error_words)
        else:
            writer.write(u'sentence:' + text)
            text_list_ret.append(text)### for return use
            writer.write(ptb_line + u'\n')
            if out_file_chunk:
                writer_chunk.write(json.dumps({'t':text,'chunk':st.chunk},ensure_ascii=False)+'\n')

    return text_list_ret,no_annotation_text


def find_no_annotation(input_path):
    #将没有标的文本提出来，scanannopath会自动忽略完全没有标注的文件
    _, _, _, no_annotation_text, no_annotation_source = scanannpath.read_path(input_path)
    return [x.strip() for x in no_annotation_text]


def remove_multi_tag(path):
    files = os.listdir(path)
    for s in files:
        son = os.path.join(path, s)
        if os.path.isdir(son):
            remove_multi_tag(son)
        elif son.endswith('.ann'):
            remove_multi_tag_in_file(son)

def remove_multi_tag_in_file(file):
    reader = open(file, 'r')
    content = []
    for line in reader:
        line = unicode(line.strip())
        if len(line) == 0:
            continue
        content.append(line)
    reader.close()
    state = False
    writer = open(file, 'w')
    for line in content:
        if not line.startswith(u'T'):
            continue
        items = line.split(u'\t')
        if len(items) < 3:
            continue
        if u';' in items[1]:
            state = True
            continue
        writer.write(line)
        writer.write(u'\n')
    writer.close()
    if state:
        print(os.path.abspath(file))

def main():
    '''
    text = u'患者于20天前无明显诱因出现面部少量红斑,随后出现胸部红斑、鳞屑,无瘙痒,无发热、上呼吸道感染等症状'
    annotation = [u'T2	time 3 7	20天前',
                    u'T3	negation 7 8	无',
                    u'T4	reason 8 12	明显诱因',
                    u'T5	body 14 16	面部',
                    u'T6	degree 16 18	少量',
                    u'T7	symptom 18 20	红斑',
                    u'T8	body 25 27	胸部',
                    u'T9	symptom 27 29	红斑',
                    u'T10	symptom 30 32	鳞屑',
                    u'T11	symptom 34 36	瘙痒',
                    u'T12	symptom 38 40	发热',
                    u'T13	other 0 3	患者于',
                    u'T14	other 12 14	出现',
                    u'T15	other 21 25	随后出现',
                    u'T16	negation 33 34	无',
                    u'T17	negation 37 38	无',
                    u'T18	other 41 50	上呼吸道感染等症状',
                    u'R1	relation Arg1:T16 Arg2:T11',
                    u'R2	relation Arg1:T17 Arg2:T12',
                    u'R3	relation Arg1:T8 Arg2:T9',
                    u'R4	relation Arg1:T8 Arg2:T10',
                    u'R5	relation Arg1:T6 Arg2:T7',
                    u'R6	relation Arg1:T5 Arg2:T7',
                    u'R7	relation Arg1:T3 Arg2:T4',
                    u'R8	relation Arg1:T4 Arg2:T7',
                    u'R9	relation Arg1:T2 Arg2:T7']
    st = SentenceTree()
    st.read_ann(text, annotation)
    st.to_ptb()
    '''
    #symptom
    ann2ptb(u'data/xbs_new/1', u'data/relation_train_corpus.txt', [u'symptom', u'signal', u'metabolite', u'menstruation', u'pregnant', u'diagnosis', u'other'])
    #ann2ptb(u'data/test_treatment_xbs', u'data/test_treatment_xbs.txt', [u'medicine', u'program', u'program_date', u'principle', u'operation', u'handle', u'adverse_reaction'])
    #ann2ptb(u'data/test_test_xbs', u'data/test_test_xbs.txt', [u'test_name'])
    #no_annotation_text = find_no_annotation(u'data/test_inspect_xbs/')
    #devide_prediction_by_documentsize(no_annotation_text, u'data/test_inspect_xbs/imporvement/', 1)

    # signal
    workbook = xlrd.open_workbook(u'data/无部位体征.xlsx')
    sheet = workbook.sheet_by_index(0)
    rows = sheet.nrows
    whilt_list = set()
    for i in xrange(rows):
        word = sheet.cell(i, 0).value
        word.replace(u'\(', u'(')
        word.replace(u'\)', u')')
        whilt_list.add(word)
    ann2ptb_signal(u'data/test_signal_xbs', u'data/relation_train_signal_corpus.txt',
                   [u'exam_body', u'exam_project', u'exam_result', u'vital_signs'], whilt_list)

    # 抽取出inspect_relation未审核数据，给刘医生做第二批审核
    #remove_multi_tag(u'data/inspect_relation/')
    # no_annotation_text = find_no_annotation(u'data/inspect_relation/')
    # devide_prediction_by_documentsize(no_annotation_text, u'data/inspect_relation/improvement/', 1)


if __name__ == '__main__':
    main()