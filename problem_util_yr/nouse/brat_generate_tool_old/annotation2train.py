#coding:utf-8
#brat的annotation数据将多行放进一个文件，标注只针对文件，没有针对每个句子，这样不利于训练，这里分析一个文件夹下的所有annotation文件，解析结果放到一个文件中
import sys, os
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

def transform_file(source_file, annotation_file, target_file):
    sreader = open(source_file, u'r')
    source = []
    segments = [] #\n position
    index = 0
    for line in sreader:
        line = unicode(line)
        source.append(line)
        segments.append(index+len(line)-1)
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

    writer = open(target_file, u'a')
    for key, value in annotation.items():
        if len(value) == 0:
            continue
        writer.write(key)
        for item in value:
            writer.write(u'\t' + item + u'\n')
        writer.write(u'\n')
    writer.close()

def transform_all(path, target_file):
    files = os.listdir(path)
    for s in files:
        son = os.path.join(path, s)
        if os.path.isdir(son):
            transform_all(son, target_file)
        else:
            if son.endswith(u'.ann'):
                fsize = os.path.getsize(son)
                source_file = son[:-4] + u'.txt'
                if fsize > 0 and os.path.exists(source_file):
                    transform_file(source_file, son, target_file)


def main():
    #transform_all(u'data/menzhen_annotation/', u'data/annotation.txt')
    transform_file(u'data/1/text00001.txt', u'data/1/text00001.ann', u'data/annotation.txt')


if __name__ == '__main__':
    main()