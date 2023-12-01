from PIL import Image
from transformers import EncoderDecoderModel, GPT2Tokenizer, GPT2Model, ViltProcessor, ViltModel
#from lic_eval import lic_eval_sample

import torch
import spacy
import re
import pickle
import random
import nltk
nltk.download('punkt')
import copy
import argparse
import os
import numpy as np
import pylab
from nltk.tokenize import word_tokenize
from io import open
import sys
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from operator import itemgetter
#from sklearn.metrics import average_precision_score
#from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.utils.data as data

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import GPT2_ManualDebiasLoader_for_test
from gpt2_model import Vilt_GPT2

import utils
from nltk.translate.bleu_score import corpus_bleu ##--
import sys

from torch.autograd import Variable


torch.backends.cudnn.benchmark = True

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, BIAS_IDX, NONBIAS_IDX = 0, 1, 2, 3, 4, 5

correction_list = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ed', 'ly', 'es', 'ing']

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_beams", default=5, type=int)
    parser.add_argument("--do_sample", default=False, type=bool)
    parser.add_argument("--no_repeat_ngram_size", default=4, type=int)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--temperature", default=1.0, type=float, help="temperature for beam search")
    parser.add_argument("--save_debiased_caps", default=False, type=bool)
    parser.add_argument("--gpt2_model", default="distilgpt2", type=str, help="distilgpt2 or gpt2")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--encoder_classify_bias", default=False, type=bool)
    parser.add_argument("--decoder_classify_bias", default=False, type=bool)
    parser.add_argument("--only_crossattention", default=False, type=bool)
    parser.add_argument("--freeze_gpt2", default=False, type=bool)
    parser.add_argument("--freeze_vilt", default=False, type=bool)
    parser.add_argument("--only_encoder_classify_bias", default=False, type=bool)
    parser.add_argument("--tf_bias_classify", default=False, type=bool)
    parser.add_argument("--use_vilt_cls_pred_for_test_ipt", default=False, type=bool)
    parser.add_argument("--use_vilt_cls_pred_for_use_masking", default=False, type=bool)
    parser.add_argument("--use_vilt_cls_pred_for_use_debiasing", default=False, type=bool)
    parser.add_argument("--decoder_max_len", default=33, type=int)
    parser.add_argument("--pred_cap_path", default=None, type=str, help="pred cap path")
    parser.add_argument("--model_path", default=None, type=str, help="model path")
    parser.add_argument("--rand_test_ipt_mask", default=False, type=bool)
    parser.add_argument("--rand_mask_rate", default=0.2, type=float)
    parser.add_argument("--image_dir", default=None, type=str)

    return parser


def show_settings(args):
    print('---------------------------------')
    print('[Settings]')
    print('pred_cap_path:', args.pred_cap_path)
    print('model_path:', args.model_path)
    print('image_dir:', args.image_dir)
    # For masking for test
    print('rand_test_ipt_mask:', args.rand_test_ipt_mask)
    if args.rand_test_ipt_mask:
        print('  rand_mask_rate:', args.rand_mask_rate)

    print('save_debiased_caps:', args.save_debiased_caps) ###
    print('---------------------------------')
    print()



def clean_sent(sent, correction_list):
    sent = sent.replace('  ', ' ')
    sent = sent.replace('   ', ' ')
    sent = sent.replace(' , ', ', ')

    for corr_word in correction_list:
        before = ' ' + corr_word + ' '
        after = corr_word + ' '
        sent = sent.replace(before, after)

    sent = sent.replace(' un ', ' un')
    return sent


def words_to_sentence(word_list):
    if len(word_list) == 0:
        sent = '.'
        return sent
    else:
        for i, word in enumerate(word_list):
            if i == 0:
                sent = word
            elif word in [',']:
                sent = sent + word
            elif word == '.':
                break
            else:
                sent = sent + ' ' + word

        sent = sent + '.'
        sent = clean_sent(sent, correction_list)
        return sent


def words_to_mask_sentence(word_list, mask_rate):
    # Remove the last period if exists
    if word_list[-1] == '.':
        word_list = word_list[:-1]

    round_int = lambda x: int((x * 2 + 1) // 2)
    num_replaced_words = round_int(len(word_list) * mask_rate)
    num_replaced_words = num_replaced_words if num_replaced_words != 0 else 1

    x = [j for j in range(len(word_list))]
    replaced_ids = random.sample(x, num_replaced_words)
    replaced_ids.sort()

    for i, word in enumerate(word_list):
        if i in replaced_ids:
            if i == 0:
                sent = '[MASK]'
            else:
                sent = sent + ' ' + '[MASK]'
        elif i == 0:
            sent = word
        elif word in [',']:
            sent = sent + word
        elif word == '.':
            break
        else:
            sent = sent + ' ' + word

    sent = sent + '.'
    sent = clean_sent(sent, correction_list)
    return sent


def prepare_data(args):
        
    # Load pred cap entries
    pred_cap_entries = pickle.load(open(args.pred_cap_path, 'rb'))
    print('Load pred cap entries from %s' %args.pred_cap_path)
    print('len(pred_cap_entries):', len(pred_cap_entries))

    # Load gender subset
    imid_2_gender = pickle.load(open('Data/val_imid_gender.pkl','rb'))

    # Further preprocess the data
    new_pred_cap_entries = []
    for entry in pred_cap_entries:
        if entry['image_id'] not in imid_2_gender:
            continue

        new_entry = {}
        #gender = id_2_original_gender_val[entry['image_id']]

        ind = str(entry['image_id'])
        zero = '0'
        for _ in range(11 - len(ind)):
            zero = zero + '0'
        ind = zero + ind
        path = os.path.join(args.image_dir, "COCO_val2014_%s.jpg" %ind)

        image_name = "COCO_val2014_%s.jpg" %ind

        # Create input caption (randomly mask 20% of the tokens)
        orig_cap = copy.deepcopy(entry['caption'])
        ipt_cap = entry['caption']
        tokens = word_tokenize(ipt_cap)
        ipt_cap = words_to_mask_sentence(tokens, args.rand_mask_rate)

        new_entry['image_id'] = entry['image_id']
        new_entry['image_name'] = image_name
        new_entry['image_path'] = path
        new_entry['orig_cap'] = orig_cap
        new_entry['ipt_cap'] = ipt_cap
        ##new_entry['bb_gender'] = gender
        new_pred_cap_entries.append(new_entry)

    pred_cap_entries = new_pred_cap_entries

    return pred_cap_entries


def generate_caps(args, vilt_gpt2, gpt2_tokenizer, processor, test_loader, device):
    print('---Start generating captions')
    vilt_gpt2.eval()

    m = nn.Softmax(dim=1)

    debiased_cap_entries = []
    gender_debiased_cap_entries = []
    orig_entries = []

    debiased_cap_dict = {}
    orig_cap_dict = {}
    data_cnt = 0
    with torch.no_grad():
        batch_i = 0
        for batch in tqdm(test_loader):
            if batch_i == 10:
                break
    
            ipt_cap_list = batch['ipt_cap_list']
            im_path_list = batch['im_path_list']

            ### For Vilt input ###
            image_list = []
            for im_path in im_path_list:
                image = Image.open(im_path)  #PIL version
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_list.append(image)

            inputs = processor(image_list, ipt_cap_list, return_tensors="pt", padding=True, truncation=True)

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            pixel_values = inputs['pixel_values'].to(device)
            pixel_mask = inputs['pixel_mask'].to(device)

            tokenized_captions = None
            labels = None
            tf_cls_ipt = None

            with torch.cuda.amp.autocast():

                ### Processing ###
                beam_outputs, preds, tf_pred = vilt_gpt2.generate_text(args, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, 
                                                                        tokenized_captions, labels, tf_cls_ipt)

                if args.tf_bias_classify:
                    tf_pred = torch.permute(tf_pred, (0, 2, 1)) #[batch_size, vocab_size(class), seq_len - 1]
                    tf_pred_tmp = np.argmax(m(tf_pred).cpu().detach(), axis=1)#[batch_size, seq_len - 1]
                    tf_pred_tmp = tf_pred_tmp[:, 0] #[batch_size]

                for i, beam_output in enumerate(beam_outputs):

                    new_entry = {}
                    new_orig_entry = {}

                    debiased_sent = gpt2_tokenizer.decode(beam_output, skip_special_tokens=True)

                    if batch['period_flag_list'][i]:
                        debiased_sent = debiased_sent.split('.')[0] + '.'
                    else:
                        debiased_sent = debiased_sent.split('.')[0]

                    new_entry['debiased_sent'] = debiased_sent

                    if i == 0 and batch_i % 100 == 0:
                        print("deb text:", debiased_sent)
                        print("usd text:", new_entry['debiased_sent'])
                        print("ipt text:", ipt_cap_list[i])
                        print('org text:', batch['orig_cap_list'][i])
                        print("image id:", batch['imid_list'][i])
                        print()

                    new_entry['image_id'] = batch['imid_list'][i]
                    new_entry['ipt_cap'] = ipt_cap_list[i]
                    new_entry['orig_cap'] = batch['orig_cap_list'][i]
                    new_entry['orig_cap'] = ipt_cap_list[i]

                    debiased_cap_dict[batch['im_name_list'][i]] = new_entry['debiased_sent'] 
                    orig_cap_dict[batch['im_name_list'][i]] = new_entry['orig_cap']

                    data_cnt += 1

                    new_orig_entry['image_id'] = batch['imid_list'][i]
                    new_orig_entry['debiased_sent'] = new_entry['orig_cap'] 
                    #new_orig_entry['bb_gender'] = batch['bb_gender_list'][i]

                    debiased_cap_entries.append(new_entry)

            batch_i += 1

    return debiased_cap_entries, debiased_cap_dict, orig_cap_dict



def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    ###### GPT2 tokenizer #####
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token # This is Ok?

    ###### Prepare trained debiasing model #####
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    #### Load trained vilt_gpt2 ###
    vilt_gpt2 = Vilt_GPT2(args, gpt2_tokenizer)
    vilt_gpt2.to(device)

    model_path = args.model_path
    print(model_path)
    if isinstance(vilt_gpt2, nn.DataParallel):
        vilt_gpt2.module.load_state_dict(torch.load(model_path))
    else:
        vilt_gpt2.load_state_dict(torch.load(model_path))

    print('---------------------------------------------------------------')
    pred_cap_entries = prepare_data(args)

    print('pred_cap_entries[0]:', pred_cap_entries[0])
    print('len(pred_cap_entries):', len(pred_cap_entries))

    cap_model_loader, _ = GPT2_ManualDebiasLoader_for_test(args, pred_cap_entries)

    ##### Generate debiased captions #####
    debiased_cap_entries, debiased_cap_dict, orig_cap_dict = generate_caps(args, vilt_gpt2, gpt2_tokenizer, processor, cap_model_loader, device) 

    if args.num_beams == 1:
        decode_method = 'greedy_n%s' %str(args.no_repeat_ngram_size)
    else:
        decode_method = 'beam_n%s' %str(args.no_repeat_ngram_size)

    if args.num_beams > 5:
        decode_method = 'beam%s_n%s' %(str(args.num_beams), str(args.no_repeat_ngram_size))

    if args.save_debiased_caps:
        file_name = 'libra_m_ratio_%s.pkl' %str(args.rand_mask_rate)
        file_path = os.path.join('Output', file_name)
        pickle.dump(debiased_cap_entries, open(file_path, 'wb'))

        print('Saved debiased_cap_entries to %s' %file_path)

    print()



if __name__=='__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    show_settings(args)
    print()
    main(args)

