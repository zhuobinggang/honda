# Added 2024.10.15 for calculating confidence intervals

# 首先需要获取各个模型的331个f值分数
def table1_scores():
    from t_test import load_dic
    crf = load_dic('exp/t_test_crf_only.dic')['CRF']
    bilstm = load_dic('exp/t_test_bilstm.dic')['BILSTM']
    # NOTE: load bertsum. BERTSUM is added later, it is a bit different
    from t_test_bertsum import load_score_dict
    bertsum = load_score_dict('/home/taku/research/honda/t_test/t_test_scores_20240930_170136.json')['BERTSUM']
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
    dic = load_score_dict('/home/taku/research/honda/exp/t_test_bbc_bertsum_add_modules.json')
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


