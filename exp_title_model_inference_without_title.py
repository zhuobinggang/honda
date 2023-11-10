# 看文件名
from t_test import dataset_5div_article, common_func_detail, save_dic, load_dic, get_checkpoint_paths, ds_5div_reconstructed_with_title
import numpy as np

def get_empty_title_ds():
    empty_title_ds = dataset_5div_article()
    for ds in empty_title_ds:
        for art in ds:
            for idx, item in enumerate(art):
                art[idx] = item[:-1] + ('',)
    return empty_title_ds

def roberta_title_append_crf_empty_title(dic = None, title = False):
    from roberta import Sector_Roberta_Title_Append_Crf
    # NOTE: Set title to empty string
    ds = None
    if not title:
        key_name = 'ROBERTA_TITLE_APPEND_CRF_no_title'
        ds = get_empty_title_ds()
    else:
        key_name = 'ROBERTA_TITLE_APPEND_CRF_title'
        ds = dataset_5div_article()
        print('KEEP TITLE')
    print(ds[0][0][0])
    return common_func_detail('ROBERTA_TITLE_APPEND_CRF', Sector_Roberta_Title_Append_Crf, dic, test_datasets_by_art = ds, key_name = key_name)

def run():
    dic = roberta_title_append_crf_empty_title()
    _ = roberta_title_append_crf_empty_title(dic = dic, title = True)
    return dic

# 以token等级获取分数
def common_func_detail_token_level(checkpoint_name, instance_func):
    import torch
    # NOTE: token level
    _, tests_reconstructed = ds_5div_reconstructed_with_title()
    # BERT
    checkpoints = get_checkpoint_paths(checkpoint_name)
    res = np.zeros((5, 2, 3, 4))
    for dataset_idx, (paths_dataset, test_ds) in enumerate(zip(checkpoints, tests_reconstructed)):
        for repeat_idx, path_repeat in enumerate(paths_dataset): # 3
            model = instance_func()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            (prec, rec, f, _), emphas = model.test(test_ds, requires_ephasize_number = True)
            res[dataset_idx, 0, repeat_idx, 0] = prec
            res[dataset_idx, 0, repeat_idx, 1] = rec
            res[dataset_idx, 0, repeat_idx, 2] = f
            res[dataset_idx, 0, repeat_idx, 3] = emphas
            # NO title
            test_ds_without_title = [item[:-1] + ('',) for item in test_ds]
            (prec, rec, f, _), emphas = model.test(test_ds_without_title, requires_ephasize_number = True)
            res[dataset_idx, 1, repeat_idx, 0] = prec
            res[dataset_idx, 1, repeat_idx, 1] = rec
            res[dataset_idx, 1, repeat_idx, 2] = f
            res[dataset_idx, 1, repeat_idx, 3] = emphas
    return res


def roberta_title_append_crf_empty_title_token_level(dic = None, title = False):
    from roberta import Sector_Roberta_Title_Append_Crf
    return common_func_detail_token_level('ROBERTA_TITLE_APPEND_CRF', Sector_Roberta_Title_Append_Crf)


####### 11.10.2023 手动挑选比较漂亮的强调 ###

def print_empty_title_emphasize_by_model(m, ds_index, art_index):
    from printer import print_sentence
    ds = dataset_5div_article()[ds_index]
    ds_no_title = get_empty_title_ds()[ds_index]
    art = ds[art_index]
    art_no_title = ds_no_title[art_index]
    emp_normal = [m.emphasize(item) for item in art]
    emp_no_title = [m.emphasize(item) for item in art_no_title]
    # print
    text_normal = [print_sentence(item, empha) for item, empha in zip(art, emp_normal)]
    print(text_normal)
    text_no_title = [print_sentence(item, empha) for item, empha in zip(art_no_title, emp_no_title)]
    print(text_no_title)

def script_11102023():
    from roberta import Sector_Roberta_Title_Append_Crf
    import torch
    m = Sector_Roberta_Title_Append_Crf()
    # NOTE: DS0才能用于最初的test ds的观察
    cp = torch.load('/usr01/taku/checkpoint/honda/ROBERTA_TITLE_APPEND_CRF_RP0_DS0_step1800_dev0.462_test0.462.checkpoint')
    m.load_state_dict(cp['model_state_dict'])
    # 实际强调
    print_empty_title_emphasize_by_model(m, 0)


    



