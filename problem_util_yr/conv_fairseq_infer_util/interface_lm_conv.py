#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import sys
import time,json,copy

import torch

from fairseq import data, tasks, tokenizer, utils
import options_yr as options

from fairseq.sequence_generator import SequenceGenerator
from fairseq.utils import import_user_module
import numpy as np

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer




def buffered_read_yr(inputll, buffer_size):
    buffer=[]
    h=inputll
    for src_str in h:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer




def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(src_str, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'],
            src_lengths=batch['net_input']['src_lengths'],
        )







def cut_sentence(inputl):
    inputll = []
    for l in inputl:
        l = l.split(' ')
        num = len(l)
        for ii in range(5, num, 1):# [5 7 9 ...]
            candidate = l[:ii]
            if len(candidate) == 0: continue
            if len(candidate)>20:continue
            if len(candidate)<10:continue

            inputll.append(' '.join(candidate))
    return inputll





def readtxt(f):
    reader=open(f)
    ll=[]
    for line in reader.readlines():
        ll.append(line)
    return ll





class convolutionLM_predict(object):
    def __init__(self,nbest,modelpath,datapath,max_len_b,maxsent,no_repeat_ngram_size=5,local_flag=True):
        """

        :param nbest: choose top n best result
        :param modelpath: checkpoint path
        :param datapath: where can find dict.txt
        :param max_len_b: max length of generate story
        :param maxsent: num of sentents each batch
        :param no_repeat_ngram_size: num of repeat tokens in one sentence
        """
        inter_flag = False
        parser = options.get_generation_parser(interactive=inter_flag)

        args = options.parse_args_and_arch(parser)

        args.task =  'language_modeling'

        args.nbest =  nbest#1  # 15

        args.path =  modelpath#'./model/checkpoint_best.pt'
        #args.input = inputll[:1]


        args.data =  datapath#'./bin_data_medical/'
        args.output_dictionary_size = -1
        args.beam =  4  # 15
        args.max_len_b =  max_len_b#30
        args.diverse_beam_groups =  4  # 15
        args.no_repeat_ngram_size = no_repeat_ngram_size
        args.add_bos_token = False
        args.max_sentences = maxsent#1  # samples allowed in 1 batch
        args.buffer_size = maxsent

        if local_flag == True:
            args.cpu = True
        else:
            ### gpu
            args.fp16 = True
        ####
        self.args=args



        import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        print(args)


        use_cuda = torch.cuda.is_available() and not args.cpu
        self.use_cuda=use_cuda

        # Setup task, e.g., translation
        self.task = tasks.setup_task(args)

        # Load ensemble
        print('| loading model(s) from {}'.format(args.path))
        models, _model_args = utils.load_ensemble_for_inference(
            args.path.split(':'), self.task, model_arg_overrides=eval(args.model_overrides),
        )


        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()
        ####
        self.models = models

        # Initialize generator
        #t0=time.time()
        #self.generator = self.task.build_generator(self.args)
        #print (time.time()-t0)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)


        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in models]
        )
    #####
    def infer(self,inputs,beam=4,max_len_b=30,nbest=1,no_repeat_ngram_size=2,diverse_beam_groups=1): # inputs=[xxx,xxx]
        #####Initialize generator
        t0=time.time()
        self.args.beam=beam
        self.args.max_len_b=max_len_b
        self.args.diverse_beam_groups=diverse_beam_groups
        self.args.nbest=nbest
        self.args.no_repeat_ngram_size=no_repeat_ngram_size
        self.generator = self.task.build_generator(self.args)
        print (time.time()-t0)


        results = [] # carry all batch result
        results_ret=[]
        infer_time_eachbatch=[]

        for batch in make_batches(inputs, self.args, self.task, self.max_positions):  # 1 batch n sample
            # each obs,each batch
            t0 = time.time()
            inp_pred_d = {'time_infer': 0, 'time_id2str': 0, 'input': '', 'pred': []}
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            start_id=0
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))
            ####
            infer_time_eachbatch.append(time.time()-t0)

        #####
        print ('infer end,,,,')
        print (np.histogram(infer_time_eachbatch))
        print (np.mean(infer_time_eachbatch))

        time_id2str_eachObs=[]
        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            t0 = time.time() # each obs in all batch result
            ### record
            inp_pred_d = {}

            ###
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                #print('S-{}\t{}'.format(id, src_str))  # input
            inp_pred_d['input'] = src_str
            inp_pred_d['pred'] = []
            inp_pred_d['score']=[]

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                ###
                inp_pred_d['pred'].append(hypo_str)
                inp_pred_d['score'].append(hypo['score'])
            ###
            #src_tokens = src_tokens_i
            time_id2str_eachObs.append(time.time() - t0)


            results_ret.append(copy.copy(inp_pred_d))
            #writer.write(json.dumps(inp_pred_d,ensure_ascii=False,indent=4))
        #####
        print ('id2str...')
        print (np.histogram(time_id2str_eachObs))
        print (np.mean(time_id2str_eachObs))
        return results_ret






class convolution_seq2seq_predict(object):
    def __init__(self,nbest,modelpath,datapath,max_len_b,maxsent,no_repeat_ngram_size=5,local_flag=True,emb_path=None):
        """

        :param nbest: choose top n best result
        :param modelpath: checkpoint path
        :param datapath: where can find dict.txt
        :param max_len_b: max length of generate story
        :param maxsent: num of sentents each batch
        :param no_repeat_ngram_size: num of repeat tokens in one sentence default=2
        """
        inter_flag = False
        parser = options.get_generation_parser(interactive=inter_flag)

        args = options.parse_args_and_arch(parser)

        args.task =  'translation'

        args.nbest =  nbest#1  # 15

        args.path =  modelpath#'./model/checkpoint_best.pt'
        #args.input = inputll[:1]


        args.data =  datapath#'./bin_data_medical/'
        args.output_dictionary_size = -1
        args.beam =  4  # 15
        args.max_len_b =  max_len_b#30
        args.diverse_beam_groups =  4  # 15
        args.no_repeat_ngram_size = no_repeat_ngram_size
        args.add_bos_token = False
        args.max_sentences = maxsent#1  # samples allowed in 1 batch
        args.buffer_size = maxsent
        ### add
        args.left_pad_source=True
        args.left_pad_target=True
        args.source_lang = 'input'
        args.target_lang = 'label'
        args.dataset_impl = 'raw'
        args.max_target_positions=1024
        args.max_source_positions=1024
        ###### override previous model argments
        if emb_path:
            args.encoder_embed_path=emb_path
            override_dict={'encoder_embed_path':emb_path}
            args.model_overrides=json.dumps(override_dict)
        # else:
        #     args.encoder_embed_path = emb_path
        #     override_dict = {'encoder_embed_path': emb_path}
        #     args.model_overrides = json.dumps(override_dict)


        if local_flag == True:
            args.cpu = True
        else:
            ### gpu
            args.fp16 = True
        ####
        self.args=args



        import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        print(args)


        use_cuda = torch.cuda.is_available() and not args.cpu
        self.use_cuda=use_cuda

        # Setup task, e.g., translation
        self.task = tasks.setup_task(args)

        # Load ensemble
        print('| loading model(s) from {}'.format(args.path))
        models, _model_args = utils.load_ensemble_for_inference(
            args.path.split(':'), self.task, model_arg_overrides=eval(args.model_overrides),
        )


        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            if args.fp16:
                model.half()
            if use_cuda:
                model.cuda()
        ####
        self.models = models

        # Initialize generator
        #t0=time.time()
        #self.generator = self.task.build_generator(self.args)
        #print (time.time()-t0)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)


        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in models]
        )
    #####
    def infer(self,inputs,beam=4,max_len_b=30,nbest=1,no_repeat_ngram_size=2): # inputs=[xxx,xxx]
        #####Initialize generator
        t0=time.time()
        self.args.beam=beam
        self.args.max_len_b=max_len_b
        self.args.diverse_beam_groups=beam
        self.args.nbest=nbest
        self.args.no_repeat_ngram_size=no_repeat_ngram_size

        self.generator = self.task.build_generator(self.args)
        print (time.time()-t0)


        results = [] # carry all batch result
        results_ret=[]
        infer_time_eachbatch=[]

        for batch in make_batches(inputs, self.args, self.task, self.max_positions):  # 1 batch n sample
            # each obs,each batch
            t0 = time.time()
            inp_pred_d = {'time_infer': 0, 'time_id2str': 0, 'input': '', 'pred': []}
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = self.task.inference_step(self.generator, self.models, sample)
            start_id=0
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))
            ####
            infer_time_eachbatch.append(time.time()-t0)

        #####
        print ('infer end,,,,')
        print (np.histogram(infer_time_eachbatch))
        print (np.mean(infer_time_eachbatch))

        time_id2str_eachObs=[]
        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            t0 = time.time() # each obs in all batch result
            ### record
            inp_pred_d = {}

            ###
            if self.src_dict is not None:
                src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                #print('S-{}\t{}'.format(id, src_str))  # input
            inp_pred_d['input'] = src_str
            inp_pred_d['pred'] = []

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                ###
                inp_pred_d['pred'].append(hypo_str)
            ###
            #src_tokens = src_tokens_i
            time_id2str_eachObs.append(time.time() - t0)


            results_ret.append(copy.copy(inp_pred_d))
            #writer.write(json.dumps(inp_pred_d,ensure_ascii=False,indent=4))
        #####
        print ('id2str...')
        print (np.histogram(time_id2str_eachObs))
        print (np.mean(time_id2str_eachObs))
        return results_ret


if __name__ == '__main__':
    import sys
    import argparse
    local_flag=True


    """  
    ####
    #inputl=['辅 助 检 查 ： B 超 （ 广 东 省 中 西 医 结 合 医 院 ， 今 年 3 月 ） ： 双 乳 低 回 声 结 节 ， 定 期 复 查']
    inputl=readtxt('./tmp/t50.json')
    inputl = readtxt('./tmp/t1000.json')
    inputll=cut_sentence(inputl)
    print (len(inputll))
    import random
    #random.shuffle(inputll)
    input=inputll[:10000]

    """








    writer=open('../tmp/rst.json','w')
    writer1 = open('../tmp/rst1.json', 'w')

    #cli_main()


    input=['辅 助 检 查 ： B 超 （ 广 东 省','无 发 热 、 头 疼','左 下 肢 外 伤 内 外 固 定 术 后 一 年 余 , 流 ']
    input=['患 者 无 明 显 诱 因 出 现 胸 闷']

    predInstance=convolutionLM_predict(nbest=4,
                                       modelpath='../model/checkpoint_best.pt',
                                       datapath='../../data-bin/',
                                       max_len_b=30,
                                       maxsent=1)
    rst=predInstance.infer(input,beam=80,nbest=4,max_len_b=30)
    print ('')

