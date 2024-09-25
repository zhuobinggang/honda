from taku_reader3 import read_ds_all_with_title
from chatgpt import cal_from_csv
from title_as_append import Sector_Title_Append
import torch
from printer import *
from t_test import load_first_model, load_model
from common import flatten
from functools import lru_cache
import numpy as np

model_path = '/usr01/taku/checkpoint/honda/SECTOR_TITLE_APPEND_RP5_DS0_step2700_dev0.409_test0.436.checkpoint'

@lru_cache(maxsize=None)
def get_first_ten_article(number = 10):
    count = 0
    ds_all = read_ds_all_with_title()
    title = ''
    starts = []
    for idx,item in enumerate(ds_all):
        current_title = item[-1]
        if current_title != title:
            title = current_title
            starts.append(idx)
            count += 1
            if count > number:
                break
    arts = []
    for i in range(number):
        arts.append(ds_all[starts[i]:starts[i+1]])
    return arts


def print_art(art):
    result_txt = ''
    for item in art:
        tokens = item[0]
        result_txt += ''.join(tokens)
    return result_txt

# For one-shot example
def print_with_emphasis_mark(art):
    result_txt = ''
    tokens = []
    labels = []
    for item in art:
        tokens += item[0]
        labels += item[1]
    for i in range(len(tokens)):
        prev_label = labels[i-1] if i > 0 else 0
        cur_label = labels[i]
        if cur_label != prev_label:
            if cur_label == 1:
                result_txt += f'**{tokens[i]}'
            elif cur_label == 0:
                result_txt += f'**{tokens[i]}'
        else:
            result_txt += tokens[i]
    return result_txt


# def load_model():
#     model = Sector_Title_Append()
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.requires_grad_(False) # Freeze
#     return model


def run_others():
    path = {
        '安部': '/home/taku/research/honda/achive/hitote/output.csv', 
        'chatgpt': '/home/taku/research/honda/achive/chatgpt/chatgpt_output.csv', 
        'gpt4': '/home/taku/research/honda/achive/gpt4/gpt4output.csv'
    }
    cal_from_csv(path = path['安部'])
    cal_from_csv(path = path['chatgpt'])
    cal_from_csv(path = path['gpt4'])


############################# scripts #####################################

def common_func(instance_func, checkpoint_name, need_flatten = True):
    if not need_flatten:
        print('Sorry I will keep it simple.')
        need_flatten = True
    cal = common_func_by_model
    ress = []
    for repeat_index in range(3):
        model = load_model(checkpoint_name, instance_func, fold_index = 0, repeat_index = repeat_index)
        ress.append(cal(model, need_flatten))
    return np.array(ress)

# old version of common_func, only calculate the first model's score
def common_func_first_one(instance_func, checkpoint_name, need_flatten = True):
    model = load_first_model(checkpoint_name, instance_func)
    return common_func_by_model(model, need_flatten)

def common_func_by_model(model, need_flatten = True):
    arts = get_first_ten_article()
    if need_flatten:
        score = model.test(flatten(arts))
        return score
    else:
        scores = [model.test(art) for art in arts]
        return np.array(scores)

def run_model_roberta():
    from roberta import Sector_Roberta
    return common_func(Sector_Roberta, 'ROBERTA', need_flatten = False)


def run_model_bilstm():
    from compare_lstm import BILSTM
    return common_func(BILSTM, 'BILSTM', need_flatten = False)

def run_model_bert():
    from taku_subword_expand import Sector
    return common_func(Sector, 'NORMAL', need_flatten = False)

def run_model_ours():
    from roberta import Sector_Roberta_Title_Append_Crf
    return common_func(Sector_Roberta_Title_Append_Crf, 'ROBERTA_TITLE_APPEND_CRF', need_flatten = False)

def run_model_without_crf():
    from roberta import Sector_Roberta_Title_Append
    return common_func(Sector_Roberta_Title_Append, 'ROBERTA_TITLE_APPEND', need_flatten = False)

def run_model_without_title():
    from roberta import Sector_Roberta_Crf
    return common_func(Sector_Roberta_Crf, 'ROBERTA_CRF', need_flatten = False)

def run_model_without_roberta():
    from title_as_append import Sector_Title_Append_CRF
    return common_func(Sector_Title_Append_CRF, 'SECTOR_TITLE_APPEND_CRF', need_flatten = False)


# t_test

def get_fs_dic():
    dic = {}
    res = run_model_roberta()
    dic['roberta'] = res[:, 2].tolist()
    res = run_model_bert()
    dic['bert'] = res[:, 2].tolist()
    res = run_model_bilstm()
    dic['bilstm'] = res[:, 2].tolist()
    res = run_model_ours()
    dic['ours'] = res[:, 2].tolist()
    res = run_model_without_crf()
    dic['ours_no_crf'] = res[:, 2].tolist()
    res = run_model_without_title()
    dic['ours_no_title'] = res[:, 2].tolist()
    res = run_model_without_roberta()
    dic['ours_no_roberta'] = res[:, 2].tolist()
    # manual & chatgpt
    from chatgpt import cal_from_csv
    res = cal_from_csv(path = '/home/taku/research/honda/achive/hitote/miki.csv', need_flatten = False)
    dic['miki'] = res[:, 2].tolist()
    res = cal_from_csv(path = '/home/taku/research/honda/achive/hitote/minamiyama.csv', need_flatten = False)
    dic['minamiyama'] = res[:, 2].tolist()
    res = cal_from_csv(path = '/home/taku/research/honda/achive/hitote/ozaki.csv', need_flatten = False)
    dic['ozaki'] = res[:, 2].tolist()
    res = cal_from_csv(path = '/home/taku/research/honda/achive/chatgpt/chatgpt_output.csv', need_flatten = False)
    dic['chatgpt'] = res[:, 2].tolist()
    res = cal_from_csv(path = '/home/taku/research/honda/achive/gpt4/gpt4output.csv', need_flatten = False)
    dic['gpt4'] = res[:, 2].tolist()
    from exp import taku
    fs = taku.run(indicator = 2, need_save_dic = False)[:10]
    dic['crf'] = fs
    # from common import save_dic
    # save_dic(dic, '/home/taku/research/honda/exp/t_test_manual_10_articles_all_methods.dic')
    # return dic
    # Rule base: 第一句话强调？
    # TODO:
    fake_model = Rule_Based_Model()
    rule_base_result = common_func_by_model(fake_model, need_flatten = False)
    

class Rule_Based_Model:
    def test(self, ds):
        from common import cal_prec_rec_f1_v2
        y_pred = []
        y_true = []
        for idx, item in enumerate(ds):
            labels = item[1]
            if idx == 0:
                y_pred += [1] * len(labels)
            else:
                y_pred += [0] * len(labels)
            y_true += labels
        return cal_prec_rec_f1_v2(y_pred, y_true)


############################# 打印模型在第七个文章上的强调结果 ####################
def print_case_7_common_func(instantiate_func, checkpoint_name, case_idx = 7, model_idx = 0):
    from printer import print_sentence
    # model = load_first_model(checkpoint_name, instantiate_func)
    model = load_model(checkpoint_name, instantiate_func, repeat_index = model_idx)
    arts = get_first_ten_article()
    ds = arts[case_idx]
    emphasizes = [model.emphasize(item) for item in ds]
    texts = [print_sentence(item, empha, need_title = False) for item, empha in zip(ds, emphasizes)]
    return ''.join(texts)

# 我们的手法
def print_case_7(case_idx = 7, model_idx = 0):
    from roberta import Sector_Roberta_Title_Append_Crf
    return print_case_7_common_func(Sector_Roberta_Title_Append_Crf, 'ROBERTA_TITLE_APPEND_CRF', case_idx, model_idx)

def print_case_7_without_title(case_idx = 7):
    from roberta import Sector_Roberta_Crf
    return print_case_7_common_func(Sector_Roberta_Crf, 'ROBERTA_CRF', case_idx)

def print_case_7_without_CRF():
    from roberta import Sector_Roberta_Title_Append
    return print_case_7_common_func(Sector_Roberta_Title_Append, 'ROBERTA_TITLE_APPEND')

def print_case_7_without_RoBERTa():
    from title_as_append import Sector_Title_Append_CRF
    return print_case_7_common_func(Sector_Title_Append_CRF, 'SECTOR_TITLE_APPEND_CRF')

def print_case_7_BILSTM():
    from compare_lstm import BILSTM
    return print_case_7_common_func(BILSTM, 'BILSTM')

def print_case_7_BERT():
    from taku_subword_expand import Sector
    return print_case_7_common_func(Sector, 'NORMAL')

def print_case_7_ROBERTA():
    from roberta import Sector_Roberta
    return print_case_7_common_func(Sector_Roberta, 'ROBERTA')

