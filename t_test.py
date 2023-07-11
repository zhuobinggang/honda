# 1. 根据五个数据集加载不同的模型
from scorer import best_checkpoints
from taku_reader3 import ds_5div_reconstructed_with_title
from common import save_dic, load_dic
import numpy as np
import torch

# return: (dataset_size = 5, repeat = 3)
def get_checkpoint_paths(name):
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
    from taku_subword_expand import Sector, Sector_CRF
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
    from taku_title import Sector_Title, Sector_CRF_Title
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
    from title_as_append import Sector_Title_Append, Sector_Title_Append_CRF
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

######################### CRF ##############################

def f_score_by_articles_BERT_CRF(dic = None):
    from taku_subword_expand import Sector, Sector_CRF
    if dic is None:
        dic = {'BERT_CRF': [], 'BERT_TITLE_CRF': [], 'BERT_TITLE2_CRF': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('CRF')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = Sector_CRF()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BERT_CRF'] += temp_fs.mean(0).tolist()
    return dic

def f_score_by_articles_BERT_TITLE_CRF(dic = None):
    from taku_title import Sector_Title, Sector_CRF_Title
    if dic is None:
        dic = {'BERT_CRF': [], 'BERT_TITLE_CRF': [], 'BERT_TITLE2_CRF': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('CRF_TITLE')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = Sector_CRF_Title()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BERT_TITLE_CRF'] += temp_fs.mean(0).tolist()
    return dic

def f_score_by_articles_BERT_TITLE2_CRF(dic = None):
    from title_as_append import Sector_Title_Append, Sector_Title_Append_CRF
    if dic is None:
        dic = {'BERT_CRF': [], 'BERT_TITLE_CRF': [], 'BERT_TITLE2_CRF': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('SECTOR_TITLE_APPEND_CRF')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = Sector_Title_Append_CRF()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BERT_TITLE2_CRF'] += temp_fs.mean(0).tolist()
    return dic




######################### BiLSTM ##############################
def f_score_by_articles_BILSTM(dic = None):
    from compare_lstm import BILSTM
    if dic is None:
        dic = {'BILSTM': [], 'BILSTM_TITLE': [], 'BILSTM_TITLE_CRF': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('BILSTM')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = BILSTM()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BILSTM'] += temp_fs.mean(0).tolist()
    return dic

def f_score_by_articles_BILSTM_TITLE(dic = None):
    from compare_lstm import BILSTM_TITLE
    if dic is None:
        dic = {'BILSTM': [], 'BILSTM_TITLE': [], 'BILSTM_TITLE_CRF': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('BILSTM_TITLE')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = BILSTM_TITLE()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BILSTM_TITLE'] += temp_fs.mean(0).tolist()
    return dic

def f_score_by_articles_BILSTM_TITLE_CRF(dic = None):
    from compare_lstm import BILSTM_TITLE_CRF
    if dic is None:
        dic = {'BILSTM': [], 'BILSTM_TITLE': [], 'BILSTM_TITLE_CRF': []}
    test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths('BILSTM_TITLE_CRF')
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = BILSTM_TITLE_CRF()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic['BILSTM_TITLE_CRF'] += temp_fs.mean(0).tolist()
    return dic


###################### RoBERTa ##################

def common_func(checkpoint_name, instance_func, dic = None, test_datasets_by_art = None):
    if dic is None:
        dic = {}
    dic[checkpoint_name] = []
    if test_datasets_by_art is None:
        test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths(checkpoint_name)
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            model = instance_func()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
            temp_fs.append(temp_temp_fs)
        temp_fs = np.array(temp_fs)
        dic[checkpoint_name] += temp_fs.mean(0).tolist()
    return dic

# 增加precision & recall
def common_func_additional(checkpoint_name, instance_func, dic = None, test_datasets_by_art = None, key_name = None):
    if dic is None:
        dic = {}
    if key_name is None:
        key_name = checkpoint_name
    dic[f'{key_name}_prec'] = []
    dic[f'{key_name}_rec'] = []
    dic[f'{key_name}_f'] = []
    if test_datasets_by_art is None:
        test_datasets_by_art = dataset_5div_article()
    # BERT
    checkpoints = get_checkpoint_paths(checkpoint_name)
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        temp_fs = [] # 3 * 67
        temp_precs = []
        temp_recs = []
        for path_repeat in paths_dataset:
            temp_temp_fs = []
            temp_temp_precs = []
            temp_temp_recs = []
            model = instance_func()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            for art_idx, article in enumerate(articles):
                prec, rec, f, _ = model.test(article)
                print(f'{dataset_idx} {art_idx} : {f}')
                temp_temp_fs.append(f)
                temp_temp_precs.append(prec)
                temp_temp_recs.append(rec)
            temp_fs.append(temp_temp_fs)
            temp_precs.append(temp_temp_precs)
            temp_recs.append(temp_temp_recs)
        temp_fs = np.array(temp_fs)
        temp_precs = np.array(temp_precs)
        temp_recs = np.array(temp_recs)
        dic[f'{key_name}_f'] += temp_fs.mean(0).tolist()
        dic[f'{key_name}_prec'] += temp_precs.mean(0).tolist()
        dic[f'{key_name}_rec'] += temp_recs.mean(0).tolist()
    return dic

def roberta(dic = None):
    from roberta import Sector_Roberta
    return common_func('ROBERTA', Sector_Roberta, dic)

def roberta_title(dic = None):
    from roberta import Sector_Roberta_Title
    return common_func('ROBERTA_TITLE', Sector_Roberta_Title, dic)

def roberta_title_crf(dic = None):
    from roberta import Sector_Roberta_Title_Crf
    return common_func('ROBERTA_TITLE_CRF', Sector_Roberta_Title_Crf, dic)

###################### TITLE as empty string #####################
def bert_title_append_crf_empty_title(dic = None, title = False):
    from title_as_append import Sector_Title_Append_CRF
    # NOTE: Set title to empty string
    empty_title_ds = dataset_5div_article()
    if not title:
        for ds in empty_title_ds:
            for art in ds:
                for idx, item in enumerate(art):
                    art[idx] = item[:-1] + ('',)
    else:
        print('KEEP TITLE')
    print(empty_title_ds[0][0][0])
    return common_func_additional('SECTOR_TITLE_APPEND_CRF', Sector_Title_Append_CRF, dic, test_datasets_by_art = empty_title_ds)


###################### 检定 ####################

def t_test(dic = None):
    from scipy import stats
    if dic is None:
        dic = load_dic('exp/t_test.dic')
    # 首先norm test
    assert stats.normaltest(dic['BERT']).pvalue < 0.05
    assert stats.normaltest(dic['BERT_TITLE']).pvalue < 0.05
    assert stats.normaltest(dic['BERT_TITLE2']).pvalue < 0.05
    # 然后wilcoxon符号测试
    stats.wilcoxon(dic['BERT'], dic['BERT_TITLE'])
    stats.wilcoxon(dic['BERT'], dic['BERT_TITLE2'])
    stats.wilcoxon(dic['BERT_TITLE'], dic['BERT_TITLE2'])
    # BERT
    dic1 = load_dic('exp/t_test.dic')
    # BERT_CRF
    dic2 = load_dic('exp/t_test_crf.dic')
    # CRF
    dic3 = load_dic('exp/t_test_crf_only.dic')
    # CRF
    dic4 = load_dic('exp/t_test_bilstm.dic')
    # RobertA
    dic5 = load_dic('exp/t_test_roberta.dic')
    



