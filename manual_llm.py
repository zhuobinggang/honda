### Created at 2024.3.21
### Problem: performance was calculated by just one random trained model before, now we need to average it from all 3 trained models.
from manual import get_first_ten_article
from t_test import get_checkpoint_paths, load_model
import numpy as np

def common_func(instance_func, checkpoint_name):
    model1 = load_model(checkpoint_name, instance_func, fold_index = 0, repeat_index = 0)
    model2 = load_model(checkpoint_name, instance_func, fold_index = 0, repeat_index = 1)
    model3 = load_model(checkpoint_name, instance_func, fold_index = 0, repeat_index = 2)
    cal = common_func_by_model
    res = np.array([cal(model1), cal(model2), cal(model3)])
    return res

def common_func_by_model(model, need_flatten = True):
    arts = get_first_ten_article()
    if need_flatten:
        score = model.test(flatten(arts))
        return score # shape: (10, 4)
    else: # by art
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

