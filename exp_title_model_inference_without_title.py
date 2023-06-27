# 看文件名
from scorer import best_checkpoints
from title_as_append import Sector_Title_Append
from taku_reader3 import ds_5div_reconstructed_with_title
import numpy as np
import torch


def for_title_append():
    res = np.zeros((5, 3, 3))
    result_logits = []
    _, paths = best_checkpoints(type_names = ['SECTOR_TITLE_APPEND'], return_paths = True)
    _, org_ds_test = ds_5div_reconstructed_with_title()
    # NOTE: 这里魔改，把标题置换成空字符串
    empty_title_ds_tests = []
    for ds in org_ds_test:
        ds_empty_title = [(tokens, ls, x1, x2, '') for tokens, ls, x1, x2, title in ds]
        empty_title_ds_tests.append(ds_empty_title)
    for dataset_idx, dataset in enumerate(paths):
        print(f'{dataset_idx}\n\n\n\n')
        the_ds_test = empty_title_ds_tests[dataset_idx]
        for repeat_idx, repeat in enumerate(dataset):
            checkpoint = torch.load(repeat)
            model = Sector_Title_Append()
            model.load_state_dict(checkpoint['model_state_dict'])
            (prec, rec, f, _) = model.test(the_ds_test)
            res[dataset_idx, repeat_idx] = [prec, rec, f]
    return res
