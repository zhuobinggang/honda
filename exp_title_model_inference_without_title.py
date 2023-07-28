# 看文件名
from t_test import dataset_5div_article, common_func_detail, save_dic, load_dic, get_checkpoint_paths, ds_5div_reconstructed_with_title
import numpy as np

def roberta_title_append_crf_empty_title(dic = None, title = False):
    from roberta import Sector_Roberta_Title_Append_Crf
    # NOTE: Set title to empty string
    empty_title_ds = dataset_5div_article()
    if not title:
        key_name = 'ROBERTA_TITLE_APPEND_CRF_no_title'
        for ds in empty_title_ds:
            for art in ds:
                for idx, item in enumerate(art):
                    art[idx] = item[:-1] + ('',)
    else:
        key_name = 'ROBERTA_TITLE_APPEND_CRF_title'
        print('KEEP TITLE')
    print(empty_title_ds[0][0][0])
    return common_func_detail('ROBERTA_TITLE_APPEND_CRF', Sector_Roberta_Title_Append_Crf, dic, test_datasets_by_art = empty_title_ds, key_name = key_name)

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
