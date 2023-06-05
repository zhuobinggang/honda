from taku_subword_expand import run as run1 # vanilla, crf
from taku_title import run as run2 # title_vanilla, title_crf
from compare_lstm import run_batch as run_batch_lstm
from scorer import dd2_all_info

# bert based
def bert():
    run1(seed = 10, indexs = range(0, 3))
    run1(seed = 10, indexs = range(0, 3))
    run2(seed = 10, indexs = range(3, 10)) # NOTE: 失误，因为种子一样，所以3~6的结果是重复的
    run2(seed = 10, indexs = range(3, 10)) # NOTE: 失误，因为种子一样，所以3~6的结果是重复的
    run1(seed = 11, indexs = range(3, 6))
    run2(seed = 11, indexs = range(3, 6))
    # NOTE: 由于生成了9套模型，重复的结果已经删除，所以直接用scorer即可
    res = dd2_all_info(type_names = ['NORMAL', 'CRF', 'NORMAL_TITLE','CRF_TITLE'], repeat_index_range = range(9)) # (5, 4, 9, 3), 最后一个维度分解为(prec, rec, f)

# bilstm + fasttext
def bilstm():
    run_batch(mtype = 0) # BILSTM
    run_batch(mtype = 1) # BILSTM + TITLE
    res = dd2_all_info(type_names=['BILSTM','BILSTM_TITLE'], repeat_index_range = range(3)) # (5, 2, 3, 3), 最后一个维度分解为(prec, rec, f)

def bert_title_onehot():
    pass
