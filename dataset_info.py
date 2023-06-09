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



