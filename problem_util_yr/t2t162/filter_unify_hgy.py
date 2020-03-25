# encoding=utf8
import sys
import json
import copy
import httplib

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')

url = "http://ai-platform.rxthinking.com"
conn = httplib.HTTPConnection("ai-platform.rxthinking.com")
conn.request('GET', url + '/platform/dict/symptom_relation_unify')
relation_unify_response = conn.getresponse().read().decode('utf8')

conn.request('GET', url + '/platform/dict/symptom_word_unify')
word_unify_response = conn.getresponse().read().decode('utf8')
conn.close()

if 'data' not in word_unify_response and 'code' not in word_unify_response:
    raise Exception(word_unify_response)

relation_unify_dict = {tuple(k.split('_')): [tuple(_v) for _v in v]
                       for k, v in json.loads(relation_unify_response).get('data').items()}
word_unify_dict = json.loads(word_unify_response).get('data')


def _node_info(node):
    return node['labelId'], node['word'], node.get('att', [])


def _to_node(word, label, attr=[]):
    return {"labelId": label, "word": word, "att": attr}


def _is_negative(node):
    label, word, attrs = _node_info(node)
    return label == u'否定' or word == u'无'


def _merge_same_words(nodes):
    merged_nodes_dict = {}
    for node in nodes:
        try:
            word = node['word']
            if len(word.strip()) == 0:
                pass
            if word not in merged_nodes_dict:
                merged_nodes_dict[word] = node
            else:
                merged_nodes_dict[word]['att'] = merged_nodes_dict[word]['att'] + node.get('att', [])
        except Exception as e:
            print('merge same words error: {} in node: {}'.format(e, json.dumps(node, ensure_ascii=False)))
    merged_nodes = list(merged_nodes_dict.values())
    for node in merged_nodes:
        node['att'] = _merge_same_words(node['att'])
    return merged_nodes


def unify_words(nodes):
    def unify_word(node):
        try:
            label, word, attrs = _node_info(node)
            if label == u'否定':
                return node
            if label not in word_unify_dict:
                return None
            unified_word = word_unify_dict[label][word]
            return _to_node(unified_word, label, unify_words(attrs))
        except Exception as e:
            # print('{} is filtered.'.format(e))
            return None

    return [n for n in [unify_word(node) for node in _merge_same_words(nodes)] if n is not None]


def _transform_attr(word, attr):
    att_label, att, sub_atts = _node_info(attr)
    diverse = (word, att_label, att)
    unifies = relation_unify_dict.get(diverse, [diverse])
    if len(unifies) > 1:
        return unifies[0][0], sub_atts + [_to_node(att, att_label) for word, att_label, att in unifies]
    uni_word, uni_att_label, uni_att = unifies[0]
    return (uni_word if uni_word != word else None), \
        [] if uni_att_label == '' else [_to_node(uni_att, uni_att_label, sub_atts)]


def _transform_relation(label, word, attrs):
    if label != u'临床表现':
        return [_to_node(word, label, attrs)]
    unified_attrs = []
    unified_words = []
    diverse = (word, "", "")
    u_word, u_label, u_att = relation_unify_dict.get(diverse, [diverse])[0]
    if u_word != word:
        word = u_word
    if u_label != '':
        unified_attrs.append(_to_node(u_att, u_label))
    for attr in attrs:
        uni_word, uni_att = _transform_attr(word, attr)
        if uni_word is None:
            unified_attrs.extend(uni_att)
        else:
            unified_words.append(_to_node(uni_word, u"临床表现", uni_att))
    if not unified_words:
        unified_words.append(_to_node(word, u"临床表现"))
    for unified_word in unified_words:
        unified_word['att'] = unified_attrs + unified_word['att']
        # unified_word['att'].extend(unified_attrs)
    return unified_words


def unify_relations(nodes):
    unified_nodes = []
    for node in nodes:
        label, word, attrs = _node_info(node)
        attrs = unify_relations(attrs)
        if any([_is_negative(att) for att in attrs]):
            continue
        unodes = _transform_relation(label, word, attrs)
        unified_nodes = unified_nodes + unodes
    return unified_nodes


def nice_print(w, infers, norms, norms_train):
    w.write()
    w.write('{}\n###原文：\n{}\n\n'.format('-' * 60, text))
    w.write('###归一后(非否定)：\n{}\n\n'.format(json.dumps(unifies, indent=4, ensure_ascii=False)))


def format_nodes_nice(nodes,indent=None):
    """是可以生成比较可视化的结果"""
    def format_node(node):
        label, word, attrs = _node_info(node)
        attrs = [format_node(attr) for attr in attrs]
        return {label: word} if attrs == [] else {label: word, u'属性': attrs}
    res = [format_node(node) for node in nodes]
    return json.dumps(res, indent=indent, ensure_ascii=False)


def format_nodes_train(nodes):
    """是可以生成玥煜要的那种格式"""
    def _format_node(node):
        label, word, attrs = _node_info(node)
        attrs = ' '.join([_format_node(attr) for attr in attrs])
        return 'S{} {} E{}'.format(label, word, label) if attrs == '' else \
            'S{} {} {} E{}'.format(label, word, attrs, label)
    return ' '.join([_format_node(n) for n in nodes])


def filter_unify(infers):
    filters = unify_words(copy.deepcopy(infers))
    x = unify_relations(filters)
    unifies = _merge_same_words(x)
    return unifies


if __name__ == '__main__':
    infers = [
        {
            "att": [
                {
                    "att": [],
                    "labelId": u"身体部位",
                    "word": u"腹部"
                },
                {
                    "att": [
                        {
                            "att": [],
                            "labelId": u"身体部位",
                            "word": u"肩部"
                        },
                        {
                            "att": [],
                            "labelId": u"身体部位",
                            "word": u"背部"
                        }
                    ],
                    "labelId": u"临床表现",
                    "word": u"放射"
                }
            ],
            "labelId": u"临床表现",
            "word": u"疼痛"
        }
    ]
    unifies = filter_unify(infers)

    print('###结构化：\n{}\n\n'.format(json.dumps(infers, indent=4, ensure_ascii=False)))
    print('###归一(nice)：\n{}\n\n'.format(format_nodes_nice(unifies)))
    print('###归一(train)：\n{}\n\n'.format(format_nodes_train(unifies)))

    s=format_nodes_nice(unifies)
    s1=format_nodes_train(unifies)
    print ''

    '''
    c = 0
    with open('out.text', 'w', 'utf8') as w:
        with open('test.json', 'r', 'utf8') as f:
            for line in f:
                data = json.loads(line)
                text, infers = data['text'], data['relation_infer_result_cnlabel']
                unifies = filter_unify(infers, text)
                nice_print(w, text, unifies, infers)
                c += 1
                if c > 100:
                    break
    '''
