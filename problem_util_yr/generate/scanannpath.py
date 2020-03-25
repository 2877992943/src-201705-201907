#coding:utf-8
#扫描annotation的path，将所有text和annotation变成list条目
import sys, os
import json
reload(sys)
sys.setdefaultencoding('utf-8')

def get_start_pos(line, nodes):
    if line.startswith(u'T'):
        items = line.split(u'\t')
        name = items[0]
        i = items[1].split(u' ')
        start_pos = int(i[1])
        nodes[name] = start_pos
        return start_pos, nodes
    elif line.startswith(u'R'):
        items = line.split(u'\t')
        begin = items[1].find(u':')
        end = items[1].find(u' ', begin+1)
        son_node = items[1][begin+1:end]
        return nodes[son_node], nodes
    else:
        return -1, nodes


def whose_node(start_pos, segments):
    for i in xrange(len(segments)+1):
        if start_pos < segments[i]:
            return i
    return len(segments)


def change_pos(line, sentence_index, segments):
    if sentence_index == 0 or not line.startswith(u'T'):
        return line
    start_pos = segments[sentence_index-1] + 1
    items = line.split(u'\t')
    i = items[1].split(u' ')
    new_i = u'\t'.join([i[0], unicode(int(i[1])-start_pos), unicode(int(i[2])-start_pos)])
    new_line = u'\t'.join([items[0],
                           new_i,
                           items[2]])
    return new_line

def read_file(text_file, annotation_file):
    text_list = []
    annotation_list = []
    source_list = []

    sreader = open(text_file, u'r')
    source = []
    segments = []  # \n position
    index = 0
    for line in sreader:
        line = unicode(line)
        source.append(line)
        segments.append(index + len(line) - 1)
        index += len(line)

    areader = open(annotation_file, u'r')
    nodes = {}
    annotation = {}
    for line in areader:
        line = unicode(line.strip())
        start_pos, nodes = get_start_pos(line, nodes)
        if -1 == start_pos:
            # 没有找到开始位置
            continue
        sentence_index = whose_node(start_pos, segments)
        if sentence_index >= len(source):
            continue
        belong_sentence = source[sentence_index]
        if belong_sentence not in annotation:
            annotation[belong_sentence] = []
        annotation[belong_sentence].append(change_pos(line, sentence_index, segments))

    for key, value in annotation.items():
        if len(value) == 0:
            continue
        text_list.append(key)
        annotation_list.append(value)
        source_list.append(text_file + u':' + key)
    return text_list, annotation_list, source_list

def read_path(path):
    files = os.listdir(path)
    text = []
    annotation = []
    source = []
    for s in files:
        son = os.path.join(path, s)
        if os.path.isdir(son):
            temp_text, temp_annotation, temp_source = read_path(son)
            text.extend(temp_text)
            annotation.extend(temp_annotation)
            source.extend(temp_source)
        else:
            if son.endswith(u'.ann'):
                fsize = os.path.getsize(son)
                source_file = son[:-4] + u'.txt'
                #if fsize > 0 and os.path.exists(source_file):
                if os.path.exists(source_file):
                    temp_text, temp_annotation, temp_source = read_file(source_file, son)
                    text.extend(temp_text)
                    annotation.extend(temp_annotation)
                    source.extend(temp_source)
    return text, annotation, source