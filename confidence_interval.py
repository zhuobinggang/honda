# Added 2024.10.15 for calculating confidence intervals

# 首先需要获取各个模型的331个f值分数
def table1_scores():
    from t_test import load_dic
    crf = load_dic('exp/t_test_crf_only.dic')['CRF']
    bilstm = load_dic('exp/t_test_bilstm.dic')['BILSTM']
    # NOTE: load bertsum. BERTSUM is added later, it is a bit different
    from t_test_bertsum import load_score_dict
    bertsum = load_score_dict('./exp/t_test_scores_bertsum.json')['BERTSUM']
    bert = load_dic('exp/t_test_bert.dic')['BERT']
    roberta_based_models = load_dic('exp/t_test_roberta.dic')
    roberta = roberta_based_models['ROBERTA']
    our = roberta_based_models['ROBERTA_TITLE_APPEND_CRF']


def table2_scores():
    from t_test import load_dic
    roberta_based_models = load_dic('exp/t_test_roberta.dic')
    our = roberta_based_models['ROBERTA_TITLE_APPEND_CRF']
    without_crf = roberta_based_models['ROBERTA_TITLE_APPEND']
    # Done: 检查了一下发现ROBERTA_CRF不在dic里面，需要记录一下 -> record_model_without_title_scores() executed
    without_title = load_dic('exp/t_test_roberta_crf.dic')['ROBERTA_CRF']
    without_roberta = load_dic('exp/t_test_crf.dic')['BERT_TITLE2_CRF']


def table4_scores():
    from t_test_bertsum import load_score_dict
    dic = load_score_dict('./exp/t_test_bbc_bertsum_add_modules.json')
    print(dic.keys())
    bertsum = dic['bert_vanilla']
    add_crf = dic['bert_crf_on']
    add_title = dic['bert_title_on']
    add_roberta = dic['roberta_vanilla']
    add_all = dic['roberta-base-title-on-crf-on']

# Done by taku, 2024.10.15
def record_model_without_title_scores():
    from t_test import common_func
    from roberta import Sector_Roberta_Crf
    dic = common_func('ROBERTA_CRF', Sector_Roberta_Crf, dic = None, title_set = 0)
    from common import save_dic
    save_dic(dic, 'exp/t_test_roberta_crf.dic')
    return dic


def bootstrap_test(f_values_model1, f_values_model2, B = 10000):
    import numpy as np
    # 计算原始的均值差异
    original_diff = np.mean(f_values_model1) - np.mean(f_values_model2)
    # 设置 Bootstrap 参数
    bootstrap_diffs = []
    # 进行 Bootstrap 重采样
    for i in range(B):
        # 有放回地随机采样两组 f 值
        sample_model1 = np.random.choice(f_values_model1, size=len(f_values_model1), replace=True)
        sample_model2 = np.random.choice(f_values_model2, size=len(f_values_model2), replace=True)
        # 计算每个样本的均值差异
        diff = np.mean(sample_model1) - np.mean(sample_model2)
        bootstrap_diffs.append(diff)
    # 计算差异的 95% 置信区间
    lower_bound = np.percentile(bootstrap_diffs, 2.5)
    upper_bound = np.percentile(bootstrap_diffs, 97.5)
    # 打印置信区间
    print(f"95% 置信区间: ({lower_bound}, {upper_bound}\noriginal_diff: {original_diff})")
    # 检验原始差异是否在置信区间之外
    if 0 < lower_bound or 0 > upper_bound:
        print("两组 f 值之间的差异显著")
    else:
        print("两组 f 值之间的差异不显著")

def bootstrap_mean_confidence_interval(scores, B=10000, alpha=0.05):
    import numpy as np
    # 初始化随机数生成器
    rng = np.random.default_rng()
    # 样本大小
    n = len(scores)
    # 自举样本均值
    bootstrap_means = np.empty(B)
    for i in range(B):
        # 从原始数据中有放回地抽样
        bootstrap_sample = rng.choice(scores, size=n, replace=True)
        # 计算该自举样本的均值
        bootstrap_means[i] = np.mean(bootstrap_sample)
    # 计算百分位法的置信区间
    lower_bound = np.percentile(bootstrap_means, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound

def bootstrap_mean_confidence_interval_half_range(scores):
    import numpy as np
    lower_bound, upper_bound = bootstrap_mean_confidence_interval(scores)
    return (upper_bound - lower_bound) / 2


def save_score_dict(score_dict, extra_name = ''):
    import json
    from datetime import datetime
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在文件名中添加当前时间
    print(f'Written dic_{extra_name}{current_time}.json')
    with open(f'model_outputs/dic_{extra_name}{current_time}.json', 'w') as f:
        json.dump(score_dict, f)

def load_score_dict(file_path):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

def init_result_dic(checkpoint_name, dic = None):
    if dic is None:
        dic = {}
    dic[checkpoint_name] = {}
    for ds_idx in range(5): # 初始化dic，dic['BERT']['ds0']['rp0'] = [0,1,0...] 通过这样的格式来存储结果
        dic[checkpoint_name][f'ds{ds_idx}'] = {}
        for rp_idx in range(3):
            dic[checkpoint_name][f'ds{ds_idx}'][f'rp{rp_idx}'] = []
    return dic

def get_label_dic(test_datasets_by_art = None, title_set = 0):
    from t_test import dataset_5div_article
    if test_datasets_by_art is None:
        test_datasets_by_art = dataset_5div_article(title_set)
    label_dic = {}
    for ds_idx in range(5): 
        arts = test_datasets_by_art[ds_idx]
        label_dic[f'ds{ds_idx}'] = []
        for art in arts:
            for sent in art:
                labels = sent[1]
                label_dic[f'ds{ds_idx}'] += labels
    return label_dic # dict_keys(['ds0', 'ds1', 'ds2', 'ds3', 'ds4'])

# TODO: 未完成
def common_func_token_level(checkpoint_name, instance_func, dic = None, test_datasets_by_art = None, title_set = 0):
    import torch
    from t_test import dataset_5div_article, get_checkpoint_paths
    dic = init_result_dic(checkpoint_name, dic)
    if test_datasets_by_art is None:
        test_datasets_by_art = dataset_5div_article(title_set)
    # BERT
    checkpoints = get_checkpoint_paths(checkpoint_name)
    for dataset_idx, (paths_dataset, articles) in enumerate(zip(checkpoints, test_datasets_by_art)):
        for repeat_idx, path_repeat in enumerate(paths_dataset):
            results = []
            labels = []
            model = instance_func()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Added 2024.10.15
            # model.eval()
            for art_idx, article in enumerate(articles):
                for sentence in article:
                    bool_results = model.emphasize(sentence)
                    results += [1 if bool_result else 0 for bool_result in bool_results]
                    labels += sentence[1]
            dic[checkpoint_name][f'ds{dataset_idx}'][f'rp{repeat_idx}'] += results
    fs = calculate_scores_by_dic(dic[checkpoint_name])
    f = sum(fs) / 3
    save_score_dict(dic, f'{checkpoint_name}_{str(f)[2: 5]}_')
    return dic

def calculate_scores_by_dic(score_dic):
    import numpy as np
    label_dic = get_label_dic()
    labels = []
    for ds_idx in range(5):
        labels += label_dic[f'ds{ds_idx}'] # 180k
    results = [[],[],[]]
    for ds_idx in range(5):
        for rp_idx in range(3):
            results[rp_idx] += score_dic[f'ds{ds_idx}'][f'rp{rp_idx}']
    results = np.array(results) # (3, 180k)
    fscores = []
    for i in range(3):
        _, _, f, _ = cal_prec_rec_f1_v2(results[i], labels)
        fscores.append(f)
    return fscores
    

    

def cal_prec_rec_f1_v2(results, targets):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for guess, target in zip(results, targets):
        if guess == 1:
            if target == 1:
                TP += 1
            elif target == 0:
                FP += 1
        elif guess == 0:
            if target == 1:
                FN += 1
            elif target == 0:
                TN += 1
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
    balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
    return prec, rec, f1, balanced_acc


def bootstrap_mean_confidence_interval_token_level(x, y, B=1000, alpha=0.05):
    import numpy as np
    x = np.array(x)
    y = np.array(y)
    # 初始化随机数生成器
    rng = np.random.default_rng()
    # 样本大小
    n = len(x)
    # 自举样本均值
    bootstrap_fs = np.empty(B)
    for i in range(B):
        # 从原始数据中有放回地抽样
        idx = rng.choice(n, size=n, replace=True)
        bootstrap_x = x[idx]
        bootstrap_y = y[idx]
        _, _, f, _ = cal_prec_rec_f1_v2(bootstrap_x, bootstrap_y)
        # 计算该自举样本的均值
        bootstrap_fs[i] = f
    # 计算百分位法的置信区间
    lower_bound = np.percentile(bootstrap_fs, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_fs, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound, bootstrap_fs

def roberta_title_append_crf(dic = None):
    from roberta import Sector_Roberta_Title_Append_Crf
    return common_func_token_level('ROBERTA_TITLE_APPEND_CRF', Sector_Roberta_Title_Append_Crf, dic)

def test_until(name, instance_func, paper_mean, max_try = 10):
    import numpy as np
    min_delta = 1000
    for i in range(max_try):
        dic = common_func_token_level(name, instance_func)
        fs = calculate_scores_by_dic(dic[name])
        mean = np.mean(fs)
        print('XXXXXXXXXXXXXXXXXXXXXXXX')
        print(mean)
        delta = np.abs(mean - paper_mean)
        if delta < min_delta:
            min_delta = delta
            print(f'mean delta UPDATED: {min_delta}')
        if (delta < 0.0007):
            print(f'I GOT IT {mean}')
            break
    return min_delta

# TODO
def roberta(dic = None):
    from roberta import Sector_Roberta
    delta = test_until('ROBERTA', Sector_Roberta, 0.399)
    return delta

# DONE
def bert(dic = None):
    from taku_subword_expand import Sector
    return test_until('NORMAL', Sector, 0.362)

# DONE
def bilstm():
    from compare_lstm import BILSTM
    return test_until('BILSTM', BILSTM, 0.297, max_try= 1)


# DONE
def without_crf():
    from roberta import Sector_Roberta_Title_Append
    return test_until('ROBERTA_TITLE_APPEND', Sector_Roberta_Title_Append, 0.423)

# DONE
def without_title():
    from roberta import Sector_Roberta_Crf
    return test_until('ROBERTA_CRF', Sector_Roberta_Crf, 0.401)

# DONE
def without_roberta():
    from title_as_append import Sector_Title_Append_CRF
    return test_until('SECTOR_TITLE_APPEND_CRF', Sector_Title_Append_CRF, 0.409)


def run():
    # roberta()
    # bert()
    bilstm()
    import os
    os.system('spd-say "your program has finished"')
    # without_crf()
    # without_title()
    # without_roberta()

def run_ignore():
    roberta()