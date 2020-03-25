#coding:utf-8
#将t2t预测的分块结果转化成annotation文件，包括.txt和.ann后缀文件
import sys, os
import random
from .scanannpath import read_path
import json
reload(sys)
sys.setdefaultencoding('utf-8')

sentence_num_per_file = 20
file_num_per_document = 50

user_list = [u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'10', u'11', u'12']

def read_text(path):
    #读一个path下的所有text
    files = os.listdir(path)
    text = []
    for s in files:
        son = os.path.join(path, s)
        if os.path.isdir(son):
            temp_text = read_text(son)
            text.extend(temp_text)
        else:
            if son.endswith(u'.txt'):
                reader = open(son, u'r')
                for textline in reader:
                    textline = unicode(textline.strip())
                    if len(textline) == 0:
                        continue
                    items = textline.split(u'。')
                    for item in items:
                        item = unicode(item.strip())
                        if len(item) > 4:
                            text.append(item)
    return text

def read_annotation(path, field):
    #从annotation中获取field字段内容
    files = os.listdir(path)
    text = set()
    for s in files:
        son = os.path.join(path, s)
        if os.path.isdir(son):
            temp_text = read_annotation(son, field)
            text = text | temp_text
        else:
            if son.endswith(u'.ann'):
                reader = open(son, u'r')
                for textline in reader:
                    textline = unicode(textline.strip())
                    if len(textline) == 0:
                        continue
                    pos1 = textline.find(u'\t')
                    if textline[pos1+1:].startswith(field):
                        pos2 = textline.find(u'\t', pos1 + 1)
                        sentences = textline[pos2+1:].split(u'。')
                        for s in sentences:
                            s = unicode(s.strip())
                            for ss in s.split(u'。'):
                                ss = ss.strip()
                                if len(ss) > 0:
                                    text.add(ss)
    return text


def read_prediction(pred_file, field):
    #pred_file：预测的json文件，从该文件中抽取对应字段
    reader = open(pred_file)
    value = []
    for line in reader:
        line = unicode(line.strip())
        if len(line) == 0:
            continue
        item = json.loads(line)
        if field in item:
            value.append(item[field])
    return value

def format_num(num):
    #将num转化成string，并在前面补0
    num_str = str(num)
    return u'0'*(5-len(num_str))+num_str

def devide_prediction_by_documentsize(text, path, start_documents=1):
    #将text平均放到文件夹，和divide_prediction不同的是，这里指定了每个文件夹最大包含的文件个数
    global sentence_num_per_file, file_num_per_document
    count = len(text)
    sentence_num_per_document = sentence_num_per_file * file_num_per_document
    document_num = count / sentence_num_per_document
    if count % sentence_num_per_document > 0:
        document_num += 1
    user_list = [str(x+start_documents) for x in xrange(document_num)]

    file_num = count / sentence_num_per_file
    if count % sentence_num_per_file > 0:
        file_num += 1
    text_file = [text[i * sentence_num_per_file: min((i + 1) * sentence_num_per_file, len(text))] for i in
                 xrange(file_num)]

    user_index = 0
    file_index = 1
    for t in text_file:
        write_annotation(t, [], user_list[user_index], format_num(file_index), path)
        file_index += 1
        if file_index > file_num_per_document:
            file_index = 1
            user_index += 1


def divide_prediction_sorted(text, path):
    #将text按照顺序，放入各个文件夹，比如前100个都放到1，101-200放到2...
    global sentence_num_per_file, user_list
    count = len(text)
    file_num = count / sentence_num_per_file
    if count % sentence_num_per_file > 0:
        file_num += 1
    text_file = [text[i*sentence_num_per_file: min((i+1)*sentence_num_per_file, len(text))] for i in xrange(file_num)]

    user_num = len(user_list)
    file_num_per_user = file_num / user_num
    if file_num % user_num > 0:
        file_num_per_user += 1

    user_index = 0
    file_index = 1
    for t in text_file:
        write_annotation(t, [], user_list[user_index], format_num(file_index), path)
        file_index += 1
        if file_index > file_num_per_user:
            file_index = 1
            user_index += 1

def divide_prediction(text, annotation, path):
    #将text平均放到各个文件夹，1放到文件夹1，2放到文件夹2，...
    global sentence_num_per_file, user_list
    file_num = 0
    while file_num*sentence_num_per_file < len(text):
        file_text = text[file_num*sentence_num_per_file: min((file_num+1)*sentence_num_per_file, len(text))]
        if len(annotation) == 0:
            file_annotation = []
        else:
            file_annotation = annotation[file_num*sentence_num_per_file: min((file_num+1)*sentence_num_per_file, len(text))]
        text_user = user_list[file_num%len(user_list)]
        text_num = file_num/len(user_list)+1
        text_num = format_num(text_num)
        write_annotation(file_text, file_annotation, text_user, text_num, path)
        file_num += 1

def extract_annotation_perline(text, annotation):
    begin = 0
    end = 0
    last_pos = u''
    none_singals = [u'pad']
    result = []
    if len(text) != len(annotation):
        print(u'bad input')
    for index in xrange(len(text)):
        if index >= len(text) and last_pos != u'':
            result.append([last_pos, begin, end, text[begin:end]])
            break
        else:
            if last_pos == u'':
                if annotation[index] in none_singals:
                    continue
                else:
                    last_pos = annotation[index]
                    begin = index
                    end = begin+1
            elif last_pos == annotation[index]:
                end += 1
            else:
                result.append([last_pos, begin, end, text[begin:end]])
                begin = index
                end = begin+1
                if annotation[index] in none_singals:
                    last_pos = u''
                else:
                    last_pos = annotation[index]
    return result

def add_annotation(set, items, num):
    for item in items:
        set.append([item[0], str(item[1]+num), str(item[2]+num), item[3]])
    return set

def write_annotation(text, annotation, user, num, path):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path+user):
        os.mkdir(path+user)
    file_name = path + user + u'/text' + num
    writer_txt = open(file_name + u'.txt', u'w')
    writer_ann = open(file_name + u'.ann', u'w')
    for line in text:
        writer_txt.write(line+u'\n')
    writer_txt.close()
    if len(annotation) == 0:
        writer_ann.close()
    else:
        annotation_num = 0
        annotation_set = []
        for line_num in xrange(len(annotation)):
            ann_items = extract_annotation_perline(text[line_num], annotation[line_num])
            annotation_set = add_annotation(annotation_set, ann_items, annotation_num)
            annotation_num += len(annotation[line_num])+1
        index = 0
        for item in annotation_set:
            writer_ann.write(u'T'+str(index) + u'\t' + item[0] + u' ' + item[1] + u' ' + item[2] + u'\t' + item[3] + u'\n')
            index += 1

def read_dict(path):
    dct = set()
    reader = open(path, u'r')
    for line in reader:
        line = unicode(line.strip())
        if len(line) == 0:
            continue
        dct.add(line+u'ct')
        dct.add(line+u'b超')
        dct.add(line+u'x线')
    return dct

def removen(text):
    result = []
    for line in text:
        line = line.strip()
        line = line.replace(u'\n', u'')
        if len(line) == 0:
            continue
        result.append(line)
    return result

def main():
    '''
    text = read_prediction(u'data/ret.json', u'text')
    annotation = read_prediction(u'data/ret.json', u'pred')
    divide_prediction(text, annotation, u'/Users/sj/PycharmProjects/medical_structuring/annotation_tool/data/zhusu_cut/')
    '''

    #text = read_prediction(u'data/abstract1117.txt', u'abs')
    #divide_prediction(text, [], u'/Users/sj/PycharmProjects/medical_structuring/annotation_tool/data/zhusu_zhengzhuang/')

    #text = read_prediction(u'data/xianbingshi.txt', u'xbs')
    #divide_prediction(text, [], u'/Users/sj/PycharmProjects/medical_structuring/annotation_tool/data/xianbingshi/')

    #抽出来一部分语料，需要医生把其中对体针块标识出来
    #text = read_prediction(u'data/vid_text_sj.txt', u'text')
    #divide_prediction(text, [], u'/Users/sj/PycharmProjects/medical_structuring/annotation_tool/data/more_signal/')

    #医生对现病史分块，根据分块结果，抽取特定的文本，比如症状，让医生从这里面看有哪些属性需要抽取
    #text = list(read_annotation(u'data/xianbingshi_annotated', u'test'))
    #random.shuffle(text)
    #devide_prediction_by_documentsize(text, u'data/test_test_xbs/', 1)
    #divide_prediction(text, [], u'/Users/sj/PycharmProjects/medical_structuring/annotation_tool/data/test_signal_xbs/')

    #text1 = list(read_annotation(u'data/test_signal_xbs_old', u'Person'))
    #text2 = list(read_annotation(u'data/more_signal', u'signal'))
    #text1.extend(text2)
    #devide_prediction_by_documentsize(text1, u'data/test_signal_xbs/', 1)

    #text = read_text(u'data/test_relation_xbs')
    #text.sort(lambda a, b: len(a)-len(b))
    #divide_prediction_sorted(text, u'data/test_relation_xbs_less/')

    #text = read_text(u'data/test_relation_xbs')
    #text = list(set(text)) #去重
    #devide_prediction_by_documentsize(text, u'data/test_relation_xbs_less/')

    #text = read_text(u'data/test_relation_xbs')
    #text = list(set(text))
    #random.shuffle(text)
    #devide_prediction_by_documentsize(text, u'data/shuffle_ralation_xbs/', 31) #医生要求从31开始

    #杨蕊给的120个科室，每个科室最大1万条化验数据，加上医生现病史分块中的化验数据，代替brat服务器上的test_test_xbs,医生标化验关系
    #text1 = read_prediction(u'data/xbsId_testHUAYANll.json', u'test')
    #text2 = read_annotation(u'data/xianbingshi_annotated', u'test')
    #for x in text1:
    #    text2.add(x)
    #text = list(text2)
    #random.shuffle(text)
    #devide_prediction_by_documentsize(text, u'data/test_test_xbs/', 1)

    # 杨蕊给的120个科室，每个科室最大1万条检查数据，加上医生现病史分块中的检查数据，代替brat服务器上的test_inspectll_xbs,医生标化验关系
    # text1 = read_prediction(u'data/xbsId_inspectll.json', u'inspect')
    # text2 = read_annotation(u'data/xianbingshi_annotated', u'inspect')
    # for x in text1:
    #     text2.add(x)
    # text = list(text2)
    # text = text[:50000]
    # random.shuffle(text)
    # devide_prediction_by_documentsize(text, u'data/test_inspect_xbs/', 1)

    # 医生现病史分块中的治疗数据，代替brat服务器上的test_inspectll_xbs,医生标化验关系
    # text2 = read_annotation(u'data/xianbingshi_annotated', u'treatment')
    # text = list(text2)
    # random.shuffle(text)
    # devide_prediction_by_documentsize(text, u'data/test_treatment_xbs/', 1)

    # 杨蕊给的120个科室，每个科室最大1万条诊断数据，分成4类放到brat上，分别是：初步诊断、鉴别诊断、诊断依据、诊疗计划
    # text = read_prediction(u'data/120treatment/初步诊断_1w', u'text')
    # random.shuffle(text)
    # devide_prediction_by_documentsize(text, u'data/120treatment/pre_diagnosis/', 1)
    #
    # text = read_prediction(u'data/120treatment/鉴别诊断_1w', u'text')
    # random.shuffle(text)
    # devide_prediction_by_documentsize(text, u'data/120treatment/identify_diagnosis/', 1)
    #
    # text = read_prediction(u'data/120treatment/诊断依据_1w', u'text')
    # random.shuffle(text)
    # devide_prediction_by_documentsize(text, u'data/120treatment/diagnosis_base/', 1)
    #
    # text = read_prediction(u'data/120treatment/诊疗计划_1w', u'text')
    # random.shuffle(text)
    # devide_prediction_by_documentsize(text, u'data/120treatment/treatment_plan/', 1)

    # 医生审核完检查的18个分块，将审核完的语料放到brat上供医生标关系
    # en2cn_dict_inspect = {'ultrasound': u'超声检查',
    #                       'pathology': u'病理检查',
    #                       'electrocardiogram': u'心电图检查',
    #                       'electromyogram': u'肌电图检查',
    #                       'electroencephalogram': u'脑电图检查',
    #                       'pulmonary': u'肺功能检查',
    #                       'endoscope': u'内窥镜检查',
    #                       'petct': u'PET-CT检查',
    #                       'nuclearbone': u'骨核医学检查',
    #                       'nuclearother': u'其他核医学检查',
    #                       'ct': u'CT检查',
    #                       'magnetic': u'磁共振检查',
    #                       'x': u'X线检查',
    #                       'hearing': u'听力检查',
    #                       'oculi': u'眼底检查',
    #                       'puncture': u'穿刺类检查',
    #                       'other': u'其他',
    #                       'test': u'化验'}
    # if not os.path.exists(u'data/inspect_relation/'):
    #     os.mkdir(u'data/inspect_relation/')
    # for field, value in en2cn_dict_inspect.items():
    #     text1 = read_annotation(u'data/test_inspect_xbs', field)
    #
    #     text2 = read_prediction(u'data/tmp/abstract_'+value+u'.json', value)
    #     text2 = set(text2)
    #     for txt in text1:
    #         if txt in text2:
    #             text2.remove(txt)
    #     text1 = list(text1)
    #     random.shuffle(text1)
    #     text2 = list(text2)
    #     random.shuffle(text2)
    #     text1.extend(text2)
    #     path = u'data/inspect_relation/'+field+u'/'
    #     if not os.path.exists(path):
    #         os.mkdir(path)
    #     devide_prediction_by_documentsize(text1, u'data/inspect_relation/'+field+u'/', 1)
    #     print(value+':'+str(len(text1)))

    # 给高医生的既往史
    # text = read_prediction(u'data/jiwangshi_out.txt', u'text')
    # text = list(set(text))
    # devide_prediction_by_documentsize(text, u'/Users/sj/PycharmProjects/annotation_tool/data/jiwangshi/')

    #给高医生新的现病史
    # text = read_prediction(u'data/new_xbs.json', u'xbs')
    # text = list(set(text))
    # devide_prediction_by_documentsize(text, u'/Users/sj/PycharmProjects/annotation_tool/data/xbs_new/')

    #给高医生的妇产科现病史
    # text = read_prediction(u'data/dept_xbs.txt', u'xbs')
    # text = list(set(text))
    # devide_prediction_by_documentsize(text, u'/Users/sj/PycharmProjects/annotation_tool/data/xbs_fc/')

    #从药品数据中抽取适应症
    text = read_prediction(u'data/drug.json', u'indication')
    text = list(set(text))
    text = removen(text)
    devide_prediction_by_documentsize(text, u'/Users/sj/PycharmProjects/annotation_tool/data/drug_indication/')




if __name__ == '__main__':
    main()