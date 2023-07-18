from taku_reader3 import read_ds_all_with_title
from chatgpt import cal_from_csv
from title_as_append import Sector_Title_Append
import torch
from printer import *
from t_test import load_first_best_model
from common import flatten
from functools import lru_cache
import numpy as np

model_path = '/usr01/taku/checkpoint/honda/SECTOR_TITLE_APPEND_RP5_DS0_step2700_dev0.409_test0.436.checkpoint'

@lru_cache(maxsize=None)
def get_first_ten_article():
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
            if count > 10:
                break
    arts = []
    for i in range(10):
        arts.append(ds_all[starts[i]:starts[i+1]])
    return arts

def load_model():
    model = Sector_Title_Append()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.requires_grad_(False) # Freeze
    return model



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
    model = load_first_best_model(checkpoint_name, instance_func)
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
    from common import save_dic
    save_dic(dic, '/home/taku/research/honda/exp/t_test_manual_10_articles_all_methods.dic')
    return dic


