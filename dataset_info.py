# NOTE: 该文件作为获取数据集情报的据点
from taku_reader2 import Loader
from functools import lru_cache
from fugashi import Tagger

@lru_cache(maxsize=None)
def read_ds_all():
    ld = Loader()
    ds_train = ld.read_trains(1)[0]
    ds_test = ld.read_tests(1)[0]
    ds_all = ds_test + ds_train
    return ds_all

def cal_emphasize_rate(ds_all = read_ds_all()):
    LENGTH_ALL = 0
    LENGTH_ONE = 0
    for item in ds_all:
        ls = item[1]
        LENGTH_ONE += sum(ls)
        LENGTH_ALL += len(ls)
    IS_TITLE_RATE = LENGTH_ONE / LENGTH_ALL
    return IS_TITLE_RATE

def infos():
    from t_test import dataset_5div_article
    ds = []
    testsets = dataset_5div_article()
    for arts in testsets:
        ds += arts
    sentence_count = [len(item) for item in ds]
    np.mean(sentence_count) # -> 21.56
    tokens_count = []
    for art in ds:
        tokens = 0
        for item in art:
            tokens += len(item[0])
        tokens_count.append(tokens)
    np.mean(sentence_count) # -> 546.62
    emphasis_count = []
    for art in ds:
        emphasis = 0
        for item in art:
            emphasis += sum(item[1])
        emphasis_count.append(emphasis)
    np.mean(emphasis_count) # -> 546.62

