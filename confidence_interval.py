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
    pass

# Done by taku, 2024.10.15
def record_model_without_title_scores():
    from t_test import common_func
    from roberta import Sector_Roberta_Crf
    dic = common_func('ROBERTA_CRF', Sector_Roberta_Crf, dic = None, title_set = 0)
    from common import save_dic
    save_dic(dic, 'exp/t_test_roberta_crf.dic')
    return dic