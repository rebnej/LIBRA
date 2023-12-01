import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
import random
import math
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import torch.nn.utils.rnn as rnn
from tqdm import tqdm
import torchvision.transforms as transforms
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from transformers import ViltProcessor

masculine = ['man','men','male','father','gentleman','gentlemen','boy','boys','uncle','husband','actor',
            'prince','waiter','son','he','his','him','himself','brother','brothers', 'guy', 'guys',
            'emperor','emperors','dude','dudes','cowboy','boyfriend','chairman','policeman','policemen']
feminine = ['woman','women','female','lady','ladies','mother','girl', 'girls','aunt','wife','actress',
            'princess','waitress','daughter','she','her','hers','herself','sister','sisters', 'queen',
            'queens','pregnant','girlfriend','chairwoman','policewoman','policewomen']
gender_words = masculine + feminine

neutral = ['person','people','parent','parents','child','children','spouse','server','they','their','them',
           'theirs','baby','babies','partner','partners','friend','friends','spouse','spouses','sibling',
           'siblings', 'chairperson','officer', 'surfer', 'kid', 'kids']



UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, BIAS_IDX, NONBIAS_IDX = 0, 1, 2, 3, 4, 5
correction_list = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ed', 'ly', 'es', 'ing']

MAX_TEXT_LENGTH = 32

# function to decide gender
def decide_gender(sent_tokens, masculine, feminine, neutral):
    gender_list = []
    for token in sent_tokens:
        token = token.lower()
        if token in masculine:
            gender_list.append('Male')
        if token in feminine:
            gender_list.append('Female')
        if token in neutral:
            gender_list.append('Neut')

    if 'Male' in gender_list and 'Female' not in gender_list:
        gender = 'Male'
    elif 'Male' not in gender_list and 'Female' in gender_list:
        gender = 'Female'
    elif 'Male' in gender_list and 'Female' in gender_list:
        gender = 'Both'
    elif 'Neut' in gender_list:
        gender = 'Neut'
    else:
        gender = 'None'

    return gender


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable) -> List[str]:
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    for data_sample in data_iter:
        yield tokenizer(data_sample)


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


def tensor_transform(args, token_ids, bias_or_nonbias):
    
    if args.decoder_classify_bias:
        if bias_or_nonbias == 'bias':
            return torch.cat((torch.tensor([BOS_IDX]),
                              torch.tensor([BIAS_IDX]),
                              torch.tensor(token_ids),
                              torch.tensor([EOS_IDX])))
        else:
            return torch.cat((torch.tensor([BOS_IDX]),
                              torch.tensor([NONBIAS_IDX]),
                              torch.tensor(token_ids),
                              torch.tensor([EOS_IDX])))
    else:
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

def is_period(sent):
    tokens = word_tokenize(sent)
    if tokens[-1] == '.':
        flag = True
    else:
        flag = False

    return flag



def ManualDebiasLoader(args, entries, split, train_vocab=None):
    random.seed(args.seed)

    batch_size = args.batch_size
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', args.bias_token, args.nonbias_token]
    random.shuffle(entries)

    # Torchtext-related (for target text)
    # build vocab
    if split == 'train':
        all_caps = []
        for entry in entries:
            ipt_cap = entry['ipt_cap']
            tgt_cap = entry['tgt_cap']
            all_caps.append(ipt_cap)
            all_caps.append(tgt_cap)

        vocab = build_vocab_from_iterator(yield_tokens(all_caps), min_freq=1, 
                                      specials=special_symbols, special_first=True)
        vocab.set_default_index(UNK_IDX)
        print("Num of vocab:", len(vocab))
        print()
    else:
        vocab = train_vocab


    # Sample entry up to batch_size
    cnt = 0
    num_batch = math.ceil(len(entries)/batch_size) 
    print("Num of batch:", num_batch)
    print()
    batch_list = []
    current = 0
    if len(entries) % batch_size != 0:
        for _ in range(num_batch-1):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size
        batch_list.append(entries[current:])
    else:
        for _ in range(num_batch):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size

    #print("batch_list[0]:", batch_list[0])
    #print()

    # Make appropriate inputs
    transform = transforms.Compose([transforms.PILToTensor()]) #for torch tensor version

    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    final_batch_list = []
    for batch_entries in tqdm(batch_list):
        new_batch_entries = []
        new_batch = {}
        #image_list = []
        im_path_list = []
        ipt_cap_list = []
        tgt_cap_tensor_list = []
        tgt_cap_list = []
        bias_or_nonbias_list = []
        max_len_tgt = 0
        imid_list = []
        bias_tgt_list = []
        for entry in batch_entries:
            imid_list.append(entry['image_id'])
            bias_or_nonbias = entry['bias_or_nonbias']
            bias_or_nonbias_list.append(bias_or_nonbias)
            if bias_or_nonbias == 'bias':
                bias_tgt_list.append(0)
            else:
                bias_tgt_list.append(1)

            im_path_list.append(entry['image_path'])

            # Input text
            ipt_cap = words_to_sentence(tokenizer(entry['ipt_cap']))
            ipt_cap_list.append(ipt_cap)

            # Target text
            tgt_cap = words_to_sentence(tokenizer(entry['tgt_cap']))
            tgt_tensor = tensor_transform(args, vocab(tokenizer(tgt_cap)), bias_or_nonbias)
            if tgt_tensor.shape[0] > max_len_tgt:
                max_len_tgt = tgt_tensor.shape[0]
            tgt_cap_tensor_list.append(tgt_tensor)
            tgt_cap_list.append(tgt_cap)

        new_batch['imid_list'] = imid_list
        #new_batch['image_list'] = image_list
        new_batch['im_path_list'] = im_path_list
        new_batch['ipt_cap_list'] = ipt_cap_list
        tgt_cap_tensor = rnn.pad_sequence(tgt_cap_tensor_list, batch_first=True)
        new_batch['tgt_cap_tensor'] = tgt_cap_tensor
        new_batch['tgt_cap_list'] = tgt_cap_list
        new_batch['bias_or_nonbias_list'] = bias_or_nonbias_list 
        new_batch['bias_tgt_tensor'] = torch.tensor(bias_tgt_list)
        final_batch_list.append(new_batch)


    return final_batch_list, vocab


def GPT2_ManualDebiasLoader(args, entries, gpt2_tokenizer, split):
    random.seed(args.seed)

    batch_size = args.batch_size
    random.shuffle(entries)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

    if args.tf_bias_classify:
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', args.bias_token, args.nonbias_token]
        # Torchtext-related (for target text)
        # build vocab
        all_caps = [args.nonbias_token, args.bias_token]

        vocab = build_vocab_from_iterator(yield_tokens(all_caps), min_freq=1,
                                      specials=special_symbols, special_first=True)
        vocab.set_default_index(UNK_IDX)
        print("Num of vocab:", len(vocab))
    else:
        vocab = None


    # Sample entry up to batch_size
    cnt = 0
    num_batch = math.ceil(len(entries)/batch_size)
    print("Num of batch:", num_batch)
    print()
    batch_list = []
    current = 0
    if len(entries) % batch_size != 0:
        for _ in range(num_batch-1):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size
        batch_list.append(entries[current:])
    else:
        for _ in range(num_batch):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size

    # Make appropriate inputs
    transform = transforms.Compose([transforms.PILToTensor()]) #for torch tensor version
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    final_batch_list = []
    batch_i = 0
    for batch_entries in tqdm(batch_list):
        new_batch_entries = []
        new_batch = {}
        ipt_cap_list = []
        tgt_cap_list = []
        bias_or_nonbias_list = []
        imid_list = []
        bias_tgt_list = []
        im_path_list = []
        tf_cls_tgt_list = []
        tgt_gender_list = []
        for entry in batch_entries:
            imid_list.append(entry['image_id'])
            bias_or_nonbias = entry['bias_or_nonbias']
            if bias_or_nonbias == 'bias':
                bias_tgt_list.append(0)
                if args.tf_bias_classify:
                    tf_cls_tgt = tensor_transform(args, vocab([args.bias_token]), bias_or_nonbias)
            else:
                bias_tgt_list.append(1)
                if args.tf_bias_classify:
                    tf_cls_tgt = tensor_transform(args, vocab([args.nonbias_token]), bias_or_nonbias)

            if args.tf_bias_classify:
                tf_cls_tgt_list.append(tf_cls_tgt)
            
            bias_or_nonbias_list.append(bias_or_nonbias)

            im_path_list.append(entry['image_path'])

            # Input text (feed into Vilt)
            if args.mask_input:
                ipt_cap = words_to_mask_sentence(word_tokenize(entry['ipt_cap']), args.mask_rate)
            else:
                ipt_cap = words_to_sentence(word_tokenize(entry['ipt_cap']))
            ipt_cap_list.append(ipt_cap)

            # Target text (feed into GPT2)
            tgt_cap = words_to_sentence(word_tokenize(entry['tgt_cap']))
            if args.decoder_classify_bias:
                if bias_or_nonbias == 'bias':
                    tgt_cap = '<b> ' + tgt_cap
                else:
                    tgt_cap = '<nb> ' + tgt_cap

            tgt_gender = decide_gender(word_tokenize(tgt_cap), masculine, feminine, neutral) 
            tgt_gender_list.append(tgt_gender)
                    
            tgt_cap_list.append(tgt_cap)

        new_batch['imid_list'] = imid_list
        new_batch['ipt_cap_list'] = ipt_cap_list
        new_batch['tgt_cap_list'] = tgt_cap_list
        new_batch['bias_or_nonbias_list'] = bias_or_nonbias_list
        new_batch['bias_tgt_tensor'] = torch.tensor(bias_tgt_list)
        new_batch['im_path_list'] = im_path_list
        new_batch['tgt_gender_list'] = tgt_gender_list
        if args.tf_bias_classify:
            tf_cls_tgt_tensor = rnn.pad_sequence(tf_cls_tgt_list, batch_first=True)
            #tf_cls_tgt_tensor = torch.tensor(tf_cls_tgt_list)
            new_batch['tf_cls_tgt_tensor'] = tf_cls_tgt_tensor
        final_batch_list.append(new_batch)


    return final_batch_list, vocab



def ManualDebiasLoader_for_test(args, entries):
    random.seed(args.seed)

    batch_size = args.batch_size
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', args.bias_token, args.nonbias_token]
    random.shuffle(entries)

    # Torchtext-related (for target text)
    # build vocab
    all_caps = []
    for entry in entries:
        ipt_cap = entry['ipt_cap']
        ##tgt_cap = entry['tgt_cap']
        all_caps.append(ipt_cap)
        ##all_caps.append(tgt_cap)

    vocab = build_vocab_from_iterator(yield_tokens(all_caps), min_freq=1,
                                      specials=special_symbols, special_first=True)
    vocab.set_default_index(UNK_IDX)
    print("Num of vocab:", len(vocab))
    print()


    # Sample entry up to batch_size
    cnt = 0
    num_batch = math.ceil(len(entries)/batch_size)
    print("Num of batch:", num_batch)
    print()
    batch_list = []
    current = 0
    if len(entries) % batch_size != 0:
        for _ in range(num_batch-1):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size
        batch_list.append(entries[current:])
    else:
        for _ in range(num_batch):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size

    #print("batch_list[0]:", batch_list[0])
    #print()

    # Make appropriate inputs
    transform = transforms.Compose([transforms.PILToTensor()]) #for torch tensor version

    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    final_batch_list = []
    for batch_entries in tqdm(batch_list):
        new_batch_entries = []
        new_batch = {}
        im_path_list = []
        ipt_cap_list = []
        imid_list = []
        for entry in batch_entries:
            imid_list.append(entry['image_id'])

            im_path_list.append(entry['image_path'])

            # Input text
            ipt_cap = words_to_sentence(tokenizer(entry['ipt_cap']))
            ipt_cap_list.append(ipt_cap)

        new_batch['imid_list'] = imid_list
        new_batch['im_path_list'] = im_path_list
        new_batch['ipt_cap_list'] = ipt_cap_list
        final_batch_list.append(new_batch)


    return final_batch_list, vocab


def GPT2_ManualDebiasLoader_for_test(args, entries):
    random.seed(args.seed)

    batch_size = args.batch_size
    random.shuffle(entries)

    if args.tf_bias_classify:
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', args.bias_token, args.nonbias_token]
        # Torchtext-related (for target text)
        # build vocab
        all_caps = [args.nonbias_token, args.bias_token]

        vocab = build_vocab_from_iterator(yield_tokens(all_caps), min_freq=1,
                                        specials=special_symbols, special_first=True)
        vocab.set_default_index(UNK_IDX)
        print("Num of vocab:", len(vocab))
    else:
        vocab = None


    # Sample entry up to batch_size
    cnt = 0
    num_batch = math.ceil(len(entries)/batch_size)
    print("Num of batch:", num_batch)
    print()
    batch_list = []
    current = 0
    if len(entries) % batch_size != 0:
        for _ in range(num_batch-1):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size
        batch_list.append(entries[current:])
    else:
        for _ in range(num_batch):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size

    # Make appropriate inputs
    transform = transforms.Compose([transforms.PILToTensor()]) #for torch tensor version
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    final_batch_list = []
    for batch_entries in tqdm(batch_list):
        new_batch_entries = []
        new_batch = {}
        im_path_list = []
        ipt_cap_list = []
        imid_list = []
        im_name_list = []
        bb_gender_list = []
        tf_cls_ipt_list = []
        period_list = []
        vilt_cls_pred_list = []
        orig_cap_list = []
        for entry in batch_entries:
            imid_list.append(entry['image_id'])
            im_path_list.append(entry['image_path'])
            im_name_list.append(entry['image_name']) 
            if 'bb_gender' in entry.keys():
                bb_gender_list.append(entry['bb_gender'])
            else:
                bb_gender_list.append('dummy')

            if args.tf_bias_classify:
                tf_cls_ipt_list.append(torch.tensor([BOS_IDX]))
            
            ###ipt_cap = words_to_sentence(word_tokenize(entry['ipt_cap'])) #!used in the CVPR paper
            ipt_cap = entry['ipt_cap']
            ipt_cap_list.append(ipt_cap)
            period_flag = is_period(entry['ipt_cap'])
            ###ipt_cap_list.append(entry['ipt_cap'])
            period_list.append(period_flag)

            if args.use_vilt_cls_pred_for_test_ipt or args.rand_test_ipt_mask:
                orig_cap_list.append(words_to_sentence(word_tokenize(entry['orig_cap'])))

                if args.use_vilt_cls_pred_for_use_masking or args.use_vilt_cls_pred_for_use_debiasing:
                    vilt_cls_pred_list.append(entry['vilt_cls_pred'])

        new_batch['imid_list'] = imid_list
        new_batch['ipt_cap_list'] = ipt_cap_list
        new_batch['im_path_list'] = im_path_list
        new_batch['im_name_list'] = im_name_list 
        new_batch['bb_gender_list'] = bb_gender_list
        new_batch['period_flag_list'] = period_list

        if args.use_vilt_cls_pred_for_test_ipt or args.rand_test_ipt_mask:
            new_batch['orig_cap_list'] = orig_cap_list

            if args.use_vilt_cls_pred_for_use_masking or args.use_vilt_cls_pred_for_use_debiasing:
                new_batch['vilt_cls_pred_list'] = vilt_cls_pred_list

        if args.tf_bias_classify:
            tf_cls_ipt_tensor = rnn.pad_sequence(tf_cls_ipt_list, batch_first=True)
            #tf_cls_tgt_tensor = torch.tensor(tf_cls_tgt_list)
            new_batch['tf_cls_ipt_tensor'] = tf_cls_ipt_tensor

        final_batch_list.append(new_batch)


    return final_batch_list, vocab


def ManualDebiasLoader_for_vilt_cls(args, entries, is_cap_model=False):
    random.seed(args.seed)

    batch_size = args.batch_size
    random.shuffle(entries)

    # Sample entry up to batch_size
    cnt = 0
    num_batch = math.ceil(len(entries)/batch_size)
    print("Num of batch:", num_batch)
    print()
    batch_list = []
    current = 0
    if len(entries) % batch_size != 0:
        for _ in range(num_batch-1):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size
        batch_list.append(entries[current:])
    else:
        for _ in range(num_batch):
            batch_entries = entries[current:current+batch_size]
            batch_list.append(batch_entries)
            current += batch_size

    # Make appropriate inputs
    transform = transforms.Compose([transforms.PILToTensor()]) #for torch tensor version
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    final_batch_list = []
    for batch_entries in tqdm(batch_list):
        new_batch_entries = []
        new_batch = {}
        im_path_list = []
        ipt_cap_list = []
        tgt_cap_list = []
        imid_list = []
        bb_gender_list = []
        im_name_list = []
        #cap_id_list = []
        for entry in batch_entries:
            imid_list.append(entry['image_id'])
            im_path_list.append(entry['image_path'])

            ipt_cap = words_to_sentence(word_tokenize(entry['ipt_cap']))
            ipt_cap_list.append(ipt_cap)

            if not is_cap_model:
                tgt_cap_list.append(entry['tgt_cap'])

            if is_cap_model:
                bb_gender_list.append(entry['bb_gender'])

        new_batch['imid_list'] = imid_list
        new_batch['ipt_cap_list'] = ipt_cap_list
        new_batch['im_path_list'] = im_path_list

        if not is_cap_model:
            new_batch['tgt_cap_list'] = tgt_cap_list
        else:
            new_batch['bb_gender_list'] = bb_gender_list

        final_batch_list.append(new_batch)


    return final_batch_list

