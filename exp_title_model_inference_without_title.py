# 看文件名
from t_test import dataset_5div_article, common_func_detail, save_dic, load_dic

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


