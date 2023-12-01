import os
import random
import pickle
from nltk import word_tokenize

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


def create_im_path(imid, im_data_root):
    ind = str(imid)
    zero = '0'
    for _ in range(11 - len(ind)):
        zero = zero + '0'
    ind = zero + ind
    im_name = 'COCO_train2014_%s.jpg' %ind
    im_path = os.path.join(im_data_root, im_name)
    
    return im_path


def create_im_path_for_test(imid, im_data_root):
    ind = str(imid)
    zero = '0'
    for _ in range(11 - len(ind)):
        zero = zero + '0'
    ind = zero + ind
    im_name = 'COCO_val2014_%s.jpg' %ind
    im_path = os.path.join(im_data_root, im_name)

    return im_path

def create_im_name_for_test(imid):
    ind = str(imid)
    zero = '0'
    for _ in range(11 - len(ind)):
        zero = zero + '0'
    ind = zero + ind
    im_name = 'COCO_val2014_%s' %ind

    return im_name



def new_t5_prepare_dataset_only_gender(args, coco_train_postag_list, lic_filtered_bias_passed_entries, gender_unpassed_bias_passed_entries, vocab_2_gender):
    random.seed(args.seed)

    # Original gender/non-gender captions

    orig_dataset_gender_entries = []
    cnt = 0
    for entries in coco_train_postag_list:
        for cap_entry in entries: # 5 captions
            entry = {}
            imid = cap_entry['image_id']
            entry['ipt_cap'] = cap_entry['caption']
            entry['tgt_cap'] = cap_entry['caption']
            entry['image_id'] = imid
            entry['image_path'] = create_im_path(imid, args.im_data_root)
            entry['bias_or_nonbias'] = 'nonbias'
            try:
                if cap_entry['gender'] in ['male', 'female']:
                    orig_dataset_gender_entries.append(entry)
            except Exception as e:
                cnt += 1
    print("!!!", cnt)
    random.shuffle(orig_dataset_gender_entries)
    print("--- Num of orig gender samples:", len(orig_dataset_gender_entries))


    # Lic filtered bias passed entries

    t5_l_b_p_entries = []
    for entry in lic_filtered_bias_passed_entries:
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['syn_cap']
        bias_entry['tgt_cap'] = entry['orig_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        bias_entry['cap_id'] = entry['cap_id']
        #bias_entry['syn_gender'] = entry['syn_gender']
        bias_entry['replaced_words'] = entry['replaced_words']
        bias_entry['generated_words'] = entry['generated_words']
        t5_l_b_p_entries.append(bias_entry)
    random.shuffle(t5_l_b_p_entries)
    print("--- Num of lic filtered and bias passed samples:", len(t5_l_b_p_entries))


    # Gender unpassed bias passed entries

    t5_g_u_b_p_entries = []
    for entry in gender_unpassed_bias_passed_entries:
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['gs_syn_cap'] # Use swapped syn cap
        bias_entry['tgt_cap'] = entry['orig_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        bias_entry['cap_id'] = entry['cap_id']
        bias_entry['syn_gender'] = entry['syn_gender']
        bias_entry['replaced_words'] = entry['replaced_words']
        bias_entry['generated_words'] = entry['generated_words']
        bias_entry['swapped_syn_gender'] = entry['swapped_syn_gender']
        t5_g_u_b_p_entries.append(bias_entry)
    random.shuffle(t5_g_u_b_p_entries)
    print("--- Num of gender unpassed bias passed samples:", len(t5_g_u_b_p_entries))


    # Gender swapping (Augly)

    gs_entries = []
    for item in vocab_2_gender:
        bias_entry = {}
        imid = item['image_id']
        orig_cap = item['orig_sent']
        bias_cap = item['bias_sent']
        bias_entry['image_id'] = imid
        bias_entry['ipt_cap'] = bias_cap
        bias_entry['tgt_cap'] = orig_cap
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        gs_entries.append(bias_entry)
    random.shuffle(gs_entries)
    print("--- Num of GS samples:", len(gs_entries))


    ###### Construct dataset #####

    orig_gender_num = len(orig_dataset_gender_entries)

    t5_l_b_p_num = len(t5_l_b_p_entries)
    
    t5_g_u_b_p_num = len(t5_g_u_b_p_entries)

    gs_num = len(gs_entries)

    # Construct by following the decided ratio 
    
    if args.use_all_biased_data:
        biased_data = t5_l_b_p_entries + t5_g_u_b_p_entries + gs_entries
        used_t5_l_b_p_num = t5_l_b_p_num
        used_t5_g_u_b_p_num = t5_g_u_b_p_num
        used_gs_num = gs_num
    else:
        if args.t5_g_u_b_p_ratio == 0 and args.gs_ratio == 0:
            min_biased_num = min([t5_l_b_p_num])
        elif args.gs_ratio == 0:
            min_biased_num = min([t5_l_b_p_num, t5_g_u_b_p_num])
        else:
            min_biased_num = min([t5_l_b_p_num, t5_g_u_b_p_num, gs_num])

        used_t5_l_b_p_num = round(args.t5_l_b_p_ratio * min_biased_num)
        used_t5_g_u_b_p_num = round(args.t5_g_u_b_p_ratio * min_biased_num)
        used_gs_num = round(args.gs_ratio * min_biased_num)

        biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]

    random.shuffle(biased_data)
    print("--- Num of constructed biased data:", len(biased_data))
    print("--- *# l_b_p : g_u_b_p : gs = ", used_t5_l_b_p_num, used_t5_g_u_b_p_num, used_gs_num)

    # Final construction 
    
    if args.only_bias_data:
        all_dataset_entries = biased_data
    else:
        # num.bias : num.orig = 1 : 1
        if len(biased_data) > orig_gender_num:
            used_biased_data_num = orig_gender_num
            used_biased_data = biased_data[:used_biased_data_num]
            used_orig_data = orig_dataset_gender_entries
        else:
            used_orig_data_num = len(biased_data)
            used_orig_data = orig_dataset_gender_entries[:used_orig_data_num]
            used_biased_data = biased_data

    if args.only_bias_data:
        print('--- #bias samples =', len(all_dataset_entries))
    else:
        print("--- #bias samples : #orig samples = ", len(used_biased_data), len(used_orig_data))

    if args.only_bias_data:
        pass
    else:
        all_dataset_entries = used_biased_data + used_orig_data
        random.shuffle(all_dataset_entries)

    num_all_dataset = len(all_dataset_entries)
    num_val = round(num_all_dataset * args.val_ratio)
    num_train = num_all_dataset - num_val
    
    if not args.imid_constraint:
        all_dataset_entries = random.sample(all_dataset_entries, len(all_dataset_entries))
        train_dataset_entries = all_dataset_entries[:num_train]
        val_dataset_entries = all_dataset_entries[num_train:]
    else:
        # Do not use same images between train and val
        pass

    return train_dataset_entries, val_dataset_entries



def vilt_masked_prepare_dataset(args, coco_train_postag_list, m_orig_gender_entries, m_lic_filtered_bias_passed_entries, m_lic_filtered_bias_passed_entries_new, m_gender_unpassed_bias_passed_entries, m_gs_entries):
    random.seed(args.seed)

    # Original gender entries

    if args.use_orig:
        orig_gender_entries = []
        for entry in m_orig_gender_entries:
            bias_entry = {}
            imid = entry['image_id']
            bias_entry['ipt_cap'] = entry['masked_sent']
            bias_entry['tgt_cap'] = entry['tgt_cap']
            bias_entry['image_id'] = imid
            bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
            bias_entry['bias_or_nonbias'] = 'nonbias'
            orig_gender_entries.append(bias_entry)
        random.shuffle(orig_gender_entries)
        print("--- Num of orig gender entries (masked):", len(orig_gender_entries))


    # Lic filtered bias passed entries

    t5_l_b_p_entries = []
    for entry in m_lic_filtered_bias_passed_entries:
        bias_entry = {}
        imid = entry['image_id']
        if args.use_vilt_masked_data:
            bias_entry['ipt_cap'] = entry['masked_sent']
        else:
            bias_entry['ipt_cap'] = entry['ipt_cap']
        bias_entry['tgt_cap'] = entry['tgt_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        #bias_entry['cap_id'] = entry['cap_id']
        #bias_entry['syn_gender'] = entry['syn_gender']
        #bias_entry['replaced_words'] = entry['replaced_words']
        #bias_entry['generated_words'] = entry['generated_words']
        t5_l_b_p_entries.append(bias_entry)
    random.shuffle(t5_l_b_p_entries)
    print("--- Num of lic filtered and bias passed samples (masked):", len(t5_l_b_p_entries))

    new_t5_l_b_p_entries = []
    for entry in m_lic_filtered_bias_passed_entries_new:
        bias_entry = {}
        imid = entry['image_id']
        if args.use_vilt_masked_data:
            bias_entry['ipt_cap'] = entry['masked_sent']
        else:
            bias_entry['ipt_cap'] = entry['ipt_cap']
        bias_entry['tgt_cap'] = entry['tgt_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        #bias_entry['cap_id'] = entry['cap_id']
        #bias_entry['syn_gender'] = entry['syn_gender']
        #bias_entry['replaced_words'] = entry['replaced_words']
        #bias_entry['generated_words'] = entry['generated_words']
        new_t5_l_b_p_entries.append(bias_entry)
    random.shuffle(new_t5_l_b_p_entries)
    print("--- Num of NEW lic filtered and bias passed samples (masked):", len(new_t5_l_b_p_entries))


    # Gender unpassed bias passed entries

    t5_g_u_b_p_entries = []
    t5_g_u_b_p_gender_entries = []
    t5_g_u_b_p_neut_entries = []
    cnt = 0
    for entry in m_gender_unpassed_bias_passed_entries:
        bias_entry = {}
        gender = decide_gender(word_tokenize(entry['tgt_cap']), masculine, feminine, neutral)
        if gender not in ['Female', 'Male']:
            cnt += 1
            if args.use_gender_loss or args.use_new_g_u_b_p:
                continue
        imid = entry['image_id']
        if args.use_vilt_masked_data:
            bias_entry['ipt_cap'] = entry['masked_sent']
        else:
            bias_entry['ipt_cap'] = entry['ipt_cap']
        bias_entry['tgt_cap'] = entry['tgt_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        #bias_entry['cap_id'] = entry['cap_id']
        #bias_entry['syn_gender'] = entry['syn_gender']
        #bias_entry['replaced_words'] = entry['replaced_words']
        #bias_entry['generated_words'] = entry['generated_words']
        #bias_entry['swapped_syn_gender'] = entry['swapped_syn_gender']
        t5_g_u_b_p_entries.append(bias_entry)
        if gender not in ['Female', 'Male']:
            t5_g_u_b_p_neut_entries.append(bias_entry)
        else:
            t5_g_u_b_p_gender_entries.append(bias_entry)
    random.shuffle(t5_g_u_b_p_entries)
    random.shuffle(t5_g_u_b_p_neut_entries)
    random.shuffle(t5_g_u_b_p_gender_entries)
    print("--- Num of gender unpassed bias passed samples (masked):", len(t5_g_u_b_p_entries))

    print('!!CNT:', cnt)

    # Gender swapping (Augly)

    gs_entries = []
    for item in m_gs_entries:
        bias_entry = {}
        imid = item['image_id']
        bias_entry['image_id'] = imid
        if args.use_vilt_masked_data:
            bias_entry['ipt_cap'] = item['masked_sent']  
        else:
            bias_entry['ipt_cap'] = item['ipt_cap']
        bias_entry['tgt_cap'] = item['tgt_cap']
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        gs_entries.append(bias_entry)
    random.shuffle(gs_entries)
    print("--- Num of GS samples (masked):", len(gs_entries))

    ###### Construct dataset #####

    t5_l_b_p_num = len(t5_l_b_p_entries)
    new_t5_l_b_p_num = len(new_t5_l_b_p_entries)

    t5_g_u_b_p_num = len(t5_g_u_b_p_entries)

    gs_num = len(gs_entries)

    # Construct by following the decided ratio 

    if args.use_all_biased_data:
        biased_data = t5_l_b_p_entries + t5_g_u_b_p_entries + gs_entries
        used_t5_l_b_p_num = t5_l_b_p_num
        used_t5_g_u_b_p_num = t5_g_u_b_p_num
        used_gs_num = gs_num
    else:
        if args.t5_g_u_b_p_ratio == 0 and args.gs_ratio == 0:
            min_biased_num = min([t5_l_b_p_num])
        elif args.gs_ratio == 0:
            min_biased_num = min([t5_l_b_p_num, t5_g_u_b_p_num])
        else:
            min_biased_num = min([t5_l_b_p_num, t5_g_u_b_p_num, gs_num])

        if args.gs_ratio == 0 and args.use_fix_num:
            used_t5_l_b_p_num = int(114568 / 2)
            used_t5_g_u_b_p_num = int(114568 / 2)
            used_gs_num = int(0)
        else:
            used_t5_l_b_p_num = round(args.t5_l_b_p_ratio * min_biased_num)
            used_t5_g_u_b_p_num = round(args.t5_g_u_b_p_ratio * min_biased_num)
            used_gs_num = round(args.gs_ratio * min_biased_num)

        if args.add_neut:
            if t5_l_b_p_num < used_t5_l_b_p_num:
                diff = used_t5_l_b_p_num - t5_l_b_p_num
                if t5_g_u_b_p_num < used_t5_g_u_b_p_num:
                    diff_gubp = used_t5_g_u_b_p_num - t5_g_u_b_p_num
                    used_gubp_entries = t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + t5_g_u_b_p_entries[:diff_gubp]
                    used_gubp_entries_add_neut = used_gubp_entries + t5_g_u_b_p_neut_entries
                    random.shuffle(used_gubp_entries_add_neut)
                    new_used_gubp_entries = used_gubp_entries_add_neut[:used_t5_g_u_b_p_num]
                    biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + new_t5_l_b_p_entries[:diff] + new_used_gubp_entries + gs_entries[:used_gs_num]
                else:
                    used_gubp_entries_add_neut = t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + t5_g_u_b_p_neut_entries
                    random.shuffle(used_gubp_entries_add_neut)
                    new_used_gubp_entries = used_gubp_entries_add_neut[:used_t5_g_u_b_p_num]
                    biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + new_t5_l_b_p_entries[:diff] + new_used_gubp_entries + gs_entries[:used_gs_num]
            else:
                used_gubp_entries_add_neut = t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + t5_g_u_b_p_neut_entries
                random.shuffle(used_gubp_entries_add_neut)
                new_used_gubp_entries = used_gubp_entries_add_neut[:used_t5_g_u_b_p_num]
                biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]

        else:
            if t5_l_b_p_num < used_t5_l_b_p_num:
                diff = used_t5_l_b_p_num - t5_l_b_p_num
                if t5_g_u_b_p_num < used_t5_g_u_b_p_num:
                    diff_gubp = used_t5_g_u_b_p_num - t5_g_u_b_p_num
                    biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + new_t5_l_b_p_entries[:diff] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + t5_g_u_b_p_entries[:diff_gubp] + gs_entries[:used_gs_num]
                else:
                    biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + new_t5_l_b_p_entries[:diff] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]
            else:
                biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]

        #if args.new_l_b_p_option == 'only_new':
        #    biased_data = new_t5_l_b_p_entries[:used_t5_l_b_p_num] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]
        #elif args.new_l_b_p_option == 'half':
        #    half_num = round(used_t5_l_b_p_num / 2)
        #    biased_data = new_t5_l_b_p_entries[:half_num] + t5_l_b_p_entries[:half_num] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]
        #elif args.new_l_b_p_option == 'add':
        #    biased_data = new_t5_l_b_p_entries[:used_t5_l_b_p_num] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]
        #elif args.new_l_b_p_option == 'not_use':
        #    biased_data = t5_l_b_p_entries[:used_t5_l_b_p_num] + t5_g_u_b_p_entries[:used_t5_g_u_b_p_num] + gs_entries[:used_gs_num]


    random.shuffle(biased_data)
    print("--- Num of constructed biased data:", len(biased_data))
    print("--- *# l_b_p : g_u_b_p : gs = ", used_t5_l_b_p_num, used_t5_g_u_b_p_num, used_gs_num)

    if args.add_neut:
        cnt = 0
        for entry in new_used_gubp_entries:
            gender = decide_gender(word_tokenize(entry['tgt_cap']), masculine, feminine, neutral)
            if gender not in ['Female', 'Male']:
                cnt += 1
        print('--- Num of neut samples:', cnt)



    # Final construction 

    if args.use_orig:
        if len(orig_gender_entries) < len(biased_data):
            biased_data = biased_data[:len(orig_gender_entries)]
        else:
            orig_gender_entries = orig_gender_entries[:len(biased_data)]
        
        all_dataset_entries = biased_data + orig_gender_entries
        print('--- #bias : #orig = ', len(biased_data), len(orig_gender_entries))

    else:
        all_dataset_entries = biased_data
        print('--- #bias samples =', len(all_dataset_entries))

    random.shuffle(all_dataset_entries)

    if args.use_fix_num:
        fix_num = 114568 
        all_dataset_entries = all_dataset_entries[:fix_num]

    num_all_dataset = len(all_dataset_entries)
    num_val = round(num_all_dataset * args.val_ratio)
    num_train = num_all_dataset - num_val

    if not args.imid_constraint:
        all_dataset_entries = random.sample(all_dataset_entries, len(all_dataset_entries))
        train_dataset_entries = all_dataset_entries[:num_train]
        val_dataset_entries = all_dataset_entries[num_train:]
    else:
        # Do not use same images between train and val
        pass

    return train_dataset_entries, val_dataset_entries


def half_gs_prepare_dataset(args, m_all_gs_entries, m_orig_gender_entries):
    print('--- Using half gender swapped data ---')
    random.seed(args.seed)

    # All gender swapping (Augly)
    gs_entries = []
    for item in m_all_gs_entries:
        bias_entry = {}
        imid = item['image_id']
        bias_entry['image_id'] = imid
        if args.use_vilt_masked_data:
            bias_entry['ipt_cap'] = item['masked_sent']
        else:
            bias_entry['ipt_cap'] = item['ipt_cap']
        bias_entry['tgt_cap'] = item['tgt_cap']
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        gs_entries.append(bias_entry)
    random.shuffle(gs_entries)
    print("--- Num of All GS samples (masked):", len(gs_entries))

    # Original
    orig_gender_entries = []
    for entry in m_orig_gender_entries:
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['masked_sent']
        bias_entry['tgt_cap'] = entry['tgt_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'nonbias'
        orig_gender_entries.append(bias_entry)
    random.shuffle(orig_gender_entries)
    print("--- Num of orig gender entries (masked):", len(orig_gender_entries))

    fix_num = 114568
    half_num = 57284

    selected_gs_entries = gs_entries[:half_num]
    selected_orig_entries = orig_gender_entries[:half_num]
    all_dataset_entries = selected_gs_entries + selected_orig_entries
    random.shuffle(all_dataset_entries)

    num_all_dataset = len(all_dataset_entries)
    num_val = round(num_all_dataset * args.val_ratio)
    num_train = num_all_dataset - num_val

    all_dataset_entries = random.sample(all_dataset_entries, len(all_dataset_entries))
    train_dataset_entries = all_dataset_entries[:num_train]
    val_dataset_entries = all_dataset_entries[num_train:]

    return train_dataset_entries, val_dataset_entries



def rand_noise_prepare_dataset(args, masked_nobias_nolic_t5_entries, masked_nobias_nolic_t5_gs_entries, masked_nobias_t5_nolic_incgender_entries):
    print('--- Using random noise data ---')
    random.seed(args.seed)

    t5_entries = []
    t5_gs_entries = []

    if args.use_fix_num:
        fix_num = 114568
    else:
        print("!!!!! PLEASE FIX NUM !!!!!")

    
    for i, entry in enumerate(masked_nobias_nolic_t5_entries):
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['masked_sent']
        bias_entry['tgt_cap'] = entry['tgt_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        #bias_entry['cap_id'] = entry['cap_id']
        #bias_entry['syn_gender'] = entry['syn_gender']
        #bias_entry['replaced_words'] = entry['replaced_words']
        #bias_entry['generated_words'] = entry['generated_words']
        t5_entries.append(bias_entry)
    random.shuffle(t5_entries)
    print("--- Num of rand noise T5 entries:", len(t5_entries))


    for i, entry in enumerate(masked_nobias_nolic_t5_gs_entries):
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['masked_sent']
        bias_entry['tgt_cap'] = entry['tgt_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        #bias_entry['cap_id'] = entry['cap_id']
        #bias_entry['syn_gender'] = entry['syn_gender']
        #bias_entry['replaced_words'] = entry['replaced_words']
        #bias_entry['generated_words'] = entry['generated_words']
        t5_gs_entries.append(bias_entry)
    random.shuffle(t5_gs_entries)
    print("--- Num of rand noise T5 + GS entries:", len(t5_gs_entries))


    if args.inc_gender:
        incgender_entries = []
        for i, entry in enumerate(masked_nobias_t5_nolic_incgender_entries):
            bias_entry = {}
            imid = entry['image_id']
            bias_entry['ipt_cap'] = entry['masked_sent']
            bias_entry['tgt_cap'] = entry['tgt_cap']
            bias_entry['image_id'] = imid
            bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
            bias_entry['bias_or_nonbias'] = 'bias'
            incgender_entries.append(bias_entry)
        random.shuffle(incgender_entries)
        print("--- Num of incgender entries:", len(incgender_entries))

    if args.inc_gender:
        all_dataset_entries = incgender_entries
    else:
        all_dataset_entries = t5_entries + t5_gs_entries

    random.shuffle(all_dataset_entries)
    all_dataset_entries = all_dataset_entries[:fix_num]

    num_all_dataset = len(all_dataset_entries)
    num_val = round(num_all_dataset * args.val_ratio)
    num_train = num_all_dataset - num_val

    all_dataset_entries = random.sample(all_dataset_entries, len(all_dataset_entries))
    train_dataset_entries = all_dataset_entries[:num_train]
    val_dataset_entries = all_dataset_entries[num_train:]

    return train_dataset_entries, val_dataset_entries



def t5_prepare_dataset(args, coco_train_postag_list, gender_bias_passed_entries, gender_unpassed_bias_passed_entries, bias_passed_nongender_entries, vocab_2_gender):
    random.seed(args.seed)
    # Original gender/non-gender captions
    orig_dataset_gender_entries = []
    orig_dataset_nongender_entries = []
    cnt = 0
    for entries in coco_train_postag_list:
        for cap_entry in entries: # 5 captions
            entry = {}
            imid = cap_entry['image_id']
            entry['ipt_cap'] = cap_entry['caption']
            entry['tgt_cap'] = cap_entry['caption']
            entry['image_id'] = imid
            entry['image_path'] = create_im_path(imid, args.im_data_root)
            entry['bias_or_nonbias'] = 'nonbias'
            try:
                if cap_entry['gender'] in ['male', 'female']:
                    orig_dataset_gender_entries.append(entry)
                else:
                    orig_dataset_nongender_entries.append(entry)
            except Exception as e:
                cnt += 1
    print("!!!", cnt)
    random.shuffle(orig_dataset_gender_entries)
    random.shuffle(orig_dataset_nongender_entries)
    print("--- Num of orig gender samples:", len(orig_dataset_gender_entries))
    print("--- Num of orig non-gender samples:", len(orig_dataset_nongender_entries))

    # Gender and bias passed entries 
    t5_g_b_p_entries = []
    for entry in gender_bias_passed_entries:
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['syn_cap']
        bias_entry['tgt_cap'] = entry['orig_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        bias_entry['cap_id'] = entry['cap_id']
        bias_entry['syn_gender'] = entry['syn_gender']
        bias_entry['replaced_words'] = entry['replaced_words']
        bias_entry['generated_words'] = entry['generated_words']
        t5_g_b_p_entries.append(bias_entry)
    random.shuffle(t5_g_b_p_entries)
    print("--- Num of gender and bias passed samples:", len(t5_g_b_p_entries))

    # Gender unpassed bias passed entries
    t5_g_u_b_p_entries = []
    for entry in gender_unpassed_bias_passed_entries:
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['gs_syn_cap'] # Use swapped syn cap
        bias_entry['tgt_cap'] = entry['orig_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        bias_entry['cap_id'] = entry['cap_id']
        bias_entry['syn_gender'] = entry['syn_gender']
        bias_entry['replaced_words'] = entry['replaced_words']
        bias_entry['generated_words'] = entry['generated_words']
        bias_entry['swapped_syn_gender'] = entry['swapped_syn_gender']
        t5_g_u_b_p_entries.append(bias_entry)
    random.shuffle(t5_g_u_b_p_entries)
    print("--- Num of gender unpassed bias passed samples:", len(t5_g_u_b_p_entries))

    # Bias passed non-gender entries 
    t5_b_p_ng_entries = []
    for entry in bias_passed_nongender_entries:
        bias_entry = {}
        imid = entry['image_id']
        bias_entry['ipt_cap'] = entry['syn_cap']
        bias_entry['tgt_cap'] = entry['orig_cap']
        bias_entry['image_id'] = imid
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        bias_entry['cap_id'] = entry['cap_id']
        bias_entry['syn_gender'] = entry['syn_gender']
        bias_entry['replaced_words'] = entry['replaced_words']
        bias_entry['generated_words'] = entry['generated_words']
        t5_b_p_ng_entries.append(bias_entry)
    random.shuffle(t5_b_p_ng_entries)
    print("--- Num of bias passed non-gender samples:", len(t5_b_p_ng_entries))


    # Gender swapping (Augly)
    gs_entries = []
    for item in vocab_2_gender:
        bias_entry = {}
        imid = item['image_id']
        orig_cap = item['orig_sent']
        bias_cap = item['bias_sent']
        bias_entry['image_id'] = imid
        bias_entry['ipt_cap'] = bias_cap
        bias_entry['tgt_cap'] = orig_cap
        bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
        bias_entry['bias_or_nonbias'] = 'bias'
        gs_entries.append(bias_entry)
    random.shuffle(gs_entries)
    print("--- Num of GS samples:", len(gs_entries))

    # Construct dataset
    orig_gender_num = len(orig_dataset_gender_entries)
    orig_nongender_num = len(orig_dataset_nongender_entries)
    orig_num = orig_gender_num + orig_nongender_num

    t5_g_b_p_num = len(t5_g_b_p_entries)
    t5_g_u_b_p_num = len(t5_g_u_b_p_entries)
    t5_gender_num = t5_g_b_p_num + t5_g_u_b_p_num
    print("-- Num of T5 gender samples:", t5_gender_num)
    t5_b_p_ng_num = len(t5_b_p_ng_entries)
    t5_num = t5_g_b_p_num +  t5_g_u_b_p_num + t5_b_p_ng_num
    print("-- Num of all T5 samples:", t5_num)

    gs_num = len(gs_entries)

    all_bias_num = t5_num + gs_num
    all_gender_bias_num = t5_gender_num + gs_num 
    print("--- Num of all gender biased samples:", all_gender_bias_num)
    print("--- Num of all biased samples:", all_bias_num)

    all_orig_entries = orig_dataset_gender_entries + orig_dataset_nongender_entries
    t5_gender_entries = t5_g_b_p_entries + t5_g_u_b_p_entries
    bias_gender_entries = t5_gender_entries + gs_entries
    all_bias_entries = bias_gender_entries + t5_b_p_ng_entries
    random.shuffle(all_orig_entries)
    random.shuffle(t5_gender_entries)
    random.shuffle(bias_gender_entries)
    random.shuffle(all_bias_entries)

    if args.use_all_t5_data:
        if orig_num >= all_bias_num:
            used_orig_num = all_bias_num
            nonbias_entries = all_orig_entries[:used_orig_num]
            bias_entries = all_bias_entries
        else:
            used_bias_num = orig_num
            bias_entries = all_bias_entries[:used_bias_num]
            nonbias_entries = all_orig_entries

    elif args.only_use_gender_data: 
        if args.only_bias_data:
            all_dataset_entries = bias_gender_entries
        elif args.only_gender_swapping:
            used_orig_gender_num = gs_num
            bias_entries = gs_entries
            nonbias_entries = orig_dataset_gender_entries[:used_orig_gender_num]
        elif all_gender_bias_num < orig_gender_num:
            used_orig_gender_num = all_gender_bias_num
            used_gender_bias_num = all_gender_bias_num
            bias_entries = bias_gender_entries
            nonbias_entries = orig_dataset_gender_entries[:used_orig_gender_num] 
        else:
            used_orig_gender_num = orig_gender_num
            used_gender_bias_num = orig_gender_num
            bias_entries = bias_gender_entries[:used_gender_bias_num]
            nonbias_entries = orig_dataset_gender_entries

    else: # Use all bias gender samples w/ ratio compared to bias non-gender samples
        if all_gender_bias_num > orig_gender_num:
            num_used_nongender_orig_data = round(orig_gender_num * args.t5_nongender_ratio)
            nongender_orig_data = orig_dataset_nongender_entries[:num_used_nongender_orig_data]
            nonbias_entries = orig_dataset_gender_entries + nongender_orig_data
            random.shuffle(nonbias_entries)
            num_used_gender_bias_data = orig_gender_num
            num_used_nongender_bias_data = num_used_nongender_orig_data
            bias_entries = t5_b_p_ng_entries[:num_used_nongender_bias_data] + bias_gender_entries[:num_used_gender_bias_data]
        else:
            num_used_nongender_bias_data = round(all_gender_bias_num * args.t5_nongender_ratio)
            nongender_bias_data = t5_b_p_ng_entries[:num_used_nongender_bias_data]
            bias_entries = nongender_bias_data + bias_gender_entries
            random.shuffle(bias_entries)
            num_used_gender_orig_data = all_gender_bias_num
            num_used_nongender_orig_data = num_used_nongender_bias_data
            nonbias_entries = orig_dataset_gender_entries[:num_used_gender_orig_data] + orig_dataset_nongender_entries[:num_used_nongender_orig_data]
            random.shuffle(nonbias_entries)

    if args.only_bias_data:
        print('--- #bias samples =', len(all_dataset_entries))
    else:
        print("--- #bias samples : #nonbias samples = ", len(bias_entries), len(nonbias_entries))

    if args.only_bias_data:
        pass
    else:
        all_dataset_entries = bias_entries + nonbias_entries

    num_all_dataset = len(all_dataset_entries)
    num_val = round(num_all_dataset * args.val_ratio)
    num_train = num_all_dataset - num_val

    all_dataset_entries = random.sample(all_dataset_entries, len(all_dataset_entries))
    train_dataset_entries = all_dataset_entries[:num_train]
    val_dataset_entries = all_dataset_entries[num_train:]

    return train_dataset_entries, val_dataset_entries



def prepare_data_for_vilt_mask(args, entries, is_orig=False, is_b_p_g_u=False, is_gs=False, is_all_gs=False, is_cap_model=False):

    if is_orig:
        # Original gender/non-gender captions
        new_entries = []
        cnt = 0
        for cap_entries in entries:
            for cap_entry in cap_entries: # 5 captions
                entry = {}
                imid = cap_entry['image_id']
                entry['ipt_cap'] = cap_entry['caption']
                entry['tgt_cap'] = cap_entry['caption']
                entry['image_id'] = imid
                entry['image_path'] = create_im_path(imid, args.im_data_root)
                entry['bias_or_nonbias'] = 'nonbias'
                try:
                    if cap_entry['gender'] in ['male', 'female']:
                        new_entries.append(entry)
                except Exception as e:
                    cnt += 1
        print("!!!", cnt)


    elif is_gs:
        # Gender swapping (Augly)
        new_entries = []
        for item in entries:
            bias_entry = {}
            imid = item['image_id']
            orig_cap = item['orig_sent']
            bias_cap = item['bias_sent']
            bias_entry['image_id'] = imid
            bias_entry['ipt_cap'] = bias_cap
            bias_entry['tgt_cap'] = orig_cap
            bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
            bias_entry['bias_or_nonbias'] = 'bias'
            new_entries.append(bias_entry)

    elif is_all_gs:
        # Gender swapping for all gender entries (Augly)
        new_entries = []
        for item in entries:
            if item['gender'] not in ['female', 'male']:
                continue
            bias_entry = {}
            imid = item['image_id']
            orig_cap = item['caption']
            bias_cap = item['swapped_cap']
            bias_entry['image_id'] = imid
            bias_entry['ipt_cap'] = bias_cap
            bias_entry['tgt_cap'] = orig_cap
            bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
            bias_entry['bias_or_nonbias'] = 'bias'
            new_entries.append(bias_entry)

    elif is_cap_model:
        # Captioning model outputs
        coco_val_imid_gender = pickle.load(open('Data/val_imid_gender.pkl', 'rb'))
        new_entries = []
        for entry in entries:

            if 'image_id' in entry:
                imid = int(entry['image_id'])
            elif 'img_id' in entry:
                imid = int(entry['img_id'])
            elif 'imid' in entry:
                imid = int(entry['imid'])

            if imid not in coco_val_imid_gender:
                continue

            entry['image_id'] = imid

            if 'pred' in entry:
                entry['ipt_cap'] = entry['pred'].lower()
            elif 'caption' in entry:
                entry['ipt_cap'] = entry['caption'].lower()

            entry['image_path'] = create_im_path_for_test(imid, args.im_data_root_for_test) # Add image_path

            if imid in coco_val_imid_gender:
                entry['bb_gender'] = coco_val_imid_gender[imid]
            else:
                entry['bb_gender'] = None

            new_entries.append(entry)

    else:
        new_entries = []
        for entry in entries:
            bias_entry = {}
            imid = entry['image_id']
            if not is_b_p_g_u:
                bias_entry['ipt_cap'] = entry['syn_cap']
            else:
                bias_entry['ipt_cap'] = entry['gs_syn_cap']
            bias_entry['tgt_cap'] = entry['orig_cap']
            bias_entry['image_id'] = imid
            bias_entry['image_path'] = create_im_path(imid, args.im_data_root)
            bias_entry['bias_or_nonbias'] = 'bias'
            bias_entry['cap_id'] = entry['cap_id']
            #bias_entry['syn_gender'] = entry['syn_gender']
            bias_entry['replaced_words'] = entry['replaced_words']
            bias_entry['generated_words'] = entry['generated_words']

            new_entries.append(bias_entry)

    print("--- Num of this smples:", len(new_entries))

    return new_entries

def prepare_data_for_gcc_vilt_mask(args, train_entries, val_entries):
    new_entries = []
    for e in train_entries:
        bias_entry = {}
        imid = e['image_id']

        if 'soft_gender_label' not in e:
            continue
        if e['sat'] == '':
            continue 
        
        if e['soft_gender_label']['female'] > e['soft_gender_label']['male']:
            gender = 'female'
        else:
            gender = 'male'
        e['bb_gender'] = gender
        e['ipt_cap'] = e['sat']
        e['image_path'] = "/home2/y-hirota/Bias-Mitigate/Debiasing/GCC/Data/train/" + imid

        new_entries.append(e)

    for e in val_entries:
        bias_entry = {}
        imid = e['image_id']

        if 'soft_gender_label' not in e:
            continue

        if e['soft_gender_label']['female'] > e['soft_gender_label']['male']:
            gender = 'female'
        else:
            gender = 'male'
        e['bb_gender'] = gender
        e['ipt_cap'] = e['sat']
        e['image_path'] = "/home2/y-hirota/Bias-Mitigate/Debiasing/GCC/Data/val/" + imid

        new_entries.append(e)

    return new_entries

    
def prepare_for_generated_caps(args, generated_caps_entries, captioning_model=None):

    """
    generated_caps_entries: [
                             {'image_id': 574769,
                             'pred': 'a woman standing in a kitchen with a refrigerator',
                              'coco_cap_list': [list of 5 GT caps]
                             } ... 
                            ]
    """

    random.seed(args.seed)

    updated_generated_caps_entries = [] 
    gender_entries = []

    bias_cnt = 0
    orig_cnt = 0

    coco_val_imid_gender = pickle.load(open('Data/val_imid_gender.pkl', 'rb'))

    round_int = lambda x: int((x * 2 + 1) // 2)

    for pred_cap_entry in generated_caps_entries:

        if 'image_id' in pred_cap_entry:
            imid = int(pred_cap_entry['image_id'])
        elif 'img_id' in pred_cap_entry:
            imid = int(pred_cap_entry['img_id'])
        elif 'imid' in pred_cap_entry:
            imid = int(pred_cap_entry['imid'])

        pred_cap_entry['image_id'] = imid

        if args.use_vilt_cls_pred_for_test_ipt:
            # Use cls pred for slecting mask or not
            if args.use_vilt_cls_pred_for_use_masking or args.use_vilt_cls_pred_for_use_debiasing:
                if pred_cap_entry['vilt_cls_pred'] == 'bias':
                    pred_cap_entry['orig_cap'] = pred_cap_entry['ipt_cap']
                    pred_cap_entry['ipt_cap'] = pred_cap_entry['masked_sent']
                    bias_cnt += 1
                else:
                    pred_cap_entry['orig_cap'] = pred_cap_entry['ipt_cap']
                    orig_cnt += 1 #pred_cap_entry['ipt_cap'] = pred_cap_entry['ipt_cap']

            # Use all vilt masked captions
            else:
                pred_cap_entry['orig_cap'] = pred_cap_entry['ipt_cap']
                pred_cap_entry['ipt_cap'] = pred_cap_entry['masked_sent']

        # No mask or random mask
        else:
            if 'pred' in pred_cap_entry:
                pred_caption = pred_cap_entry['pred']
            else:
                pred_caption = pred_cap_entry['caption']

            if not args.rand_test_ipt_mask:
                pred_cap_entry['ipt_cap'] = pred_caption 
            else:
                pred_cap_entry['orig_cap'] = pred_caption

                c_tokens = word_tokenize(pred_caption)
                new_list = []
                cap_len = len(c_tokens)
                num_replaced_words = round_int(cap_len * args.rand_mask_rate)
                num_replaced_words = num_replaced_words if num_replaced_words != 0 else 1
                x = [j for j in range(cap_len)]
                replaced_ids = random.sample(x, num_replaced_words)
                replaced_ids.sort()

                for i, t in enumerate(c_tokens):
                    if i in replaced_ids:
                        new_list.append('[MASK]')
                    else:
                        new_list.append(t)
                new_sent = ' '.join([c for c in new_list])
                pred_cap_entry['ipt_cap'] = new_sent

            
        if captioning_model == 'sat_gcc':
            pred_cap_entry['image_name'] = str(imid)
            gender_entries.append(pred_cap_entry)
        else:
            pred_cap_entry['image_path'] = create_im_path_for_test(imid, args.im_data_root_for_test) # Add image_path
            pred_cap_entry['image_name'] = create_im_name_for_test(imid)

            if imid in coco_val_imid_gender:
                pred_cap_entry['bb_gender'] = coco_val_imid_gender[imid]
                gender_entries.append(pred_cap_entry)
            else:
                pred_cap_entry['bb_gender'] = None

        updated_generated_caps_entries.append(pred_cap_entry)
        
    if args.use_vilt_cls_pred_for_use_masking or args.use_vilt_cls_pred_for_use_debiasing:
        print("--- Vilt_cls preds (bias : orig) = ", bias_cnt, orig_cnt)

    return updated_generated_caps_entries, gender_entries

