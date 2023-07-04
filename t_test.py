# 1. 根据五个数据集加载不同的模型
from scorer import best_checkpoints
from taku_reader3 import ds_5div_reconstructed_with_title
from taku_subword_expand import Sector
from taku_title import Sector_Title
from title_as_append import Sector_Title_Append
from common import save_dic, load_dic
import numpy as np
import torch

# return: (dataset_size = 5, repeat = 3)
def get_checkpoint_paths(name = 'SECTOR_TITLE_APPEND_CRF'):
    _, cps = best_checkpoints(directory_in_str = '/usr01/taku/checkpoint/honda/', type_names = [name], return_paths = True)
    return cps

# 2. 将数据集加载成arts的结构
# 如果title不同，换一个新的不就好了？
def dataset_5div_article():
    _, tests_reconstructed = ds_5div_reconstructed_with_title()
    test_datasets_by_art = []
    for tds in tests_reconstructed:
        arts = []
        prev_title = tds[0][-1]
        art_temp = []
        for item in tds:
            title = item[-1]
            if title != prev_title:
                arts.append(art_temp)
                art_temp = [item]
                prev_title = title
            else:
                art_temp.append(item)
        arts.append(art_temp)
        test_datasets_by_art.append(arts)
    return test_datasets_by_art
            
# 3. 加载模型并生成对应的f值，想想… 
def f_score_by_articles_BERT(dic = None):
    if dic is None:
        dic = {'BERT': [], 'BERT_TITLE': [], 'BERT_TITLE2': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('NORMAL')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = Sector()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BERT'] += temp_fs.mean(0).tolist()
    return dic

def f_score_by_articles_BERT_TITLE(dic = None):
    if dic is None:
        dic = {'BERT': [], 'BERT_TITLE': [], 'BERT_TITLE2': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('NORMAL_TITLE')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = Sector_Title()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BERT_TITLE'] += temp_fs.mean(0).tolist()
    return dic

def f_score_by_articles_BERT_TITLE2(dic = None):
    if dic is None:
        dic = {'BERT': [], 'BERT_TITLE': [], 'BERT_TITLE2': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('SECTOR_TITLE_APPEND')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = Sector_Title_Append()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BERT_TITLE2'] += temp_fs.mean(0).tolist()
    return dic

