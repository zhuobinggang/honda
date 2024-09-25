# NOTE: 将5分割还原到下标获取
from dataset_info import read_ds_all, cal_emphasize_rate
from taku_reader2 import Loader
from raw_data import case_id_to_title, get_sents_with_meta
from functools import lru_cache
import re

def get_testset_ranges(test_lengths):
    starts = [0]
    ends = []
    for idx, length in enumerate(test_lengths):
        starts.append(starts[idx] + length)
        ends.append(starts[idx] + length)
    return list(zip(starts, ends))

@lru_cache(maxsize=None)
def ds_5div_reconstructed():
    # 手动找到每个test.txt的开始和最后一篇文章的下标就可以了
    ds_all = read_ds_all()
    test_lengths = [1587, 1730, 1257, 1181, 1384]
    test_ranges = get_testset_ranges(test_lengths)
    # test_ranges = [(0, 67), (67, 133), (133, 199), (199, 265), (265, 331)]
    tests_reconstructed = [ds_all[start:end] for start,end in test_ranges]
    trains_reconstructed = [ds_all[0:start] + ds_all[end:] for start,end in test_ranges]
    return trains_reconstructed, tests_reconstructed

def read_dataset(idx):
    trains_reconstructed, tests_reconstructed = ds_5div_reconstructed()
    return trains_reconstructed[idx], tests_reconstructed[idx]

def valid():
    trains_reconstructed, tests_reconstructed = ds_5div_reconstructed()
    tests = Loader().read_tests(5)
    for t1, t2 in zip(tests, tests_reconstructed):
        assert t1 == t2
    trains = Loader().read_trains(5)
    for t1, t2 in zip(trains, trains_reconstructed):
        assert t1 == t2


############################ 包含Title信息的loader #######################

def read_ds_all_with_title(title_set = 0):
    res = []
    for idx, item in enumerate(read_ds_all()):
        title = case_id_to_title(idx, title_set)
        res.append(item + (title, )) # tuple concat
    return res

# title_set: 0 = original, 1 = chatgpt
@lru_cache(maxsize=None)
def ds_5div_reconstructed_with_title(title_set = 0):
    # 手动找到每个test.txt的开始和最后一篇文章的下标就可以了
    ds_all = read_ds_all_with_title(title_set)
    test_lengths = [1587, 1730, 1257, 1181, 1384]
    test_ranges = get_testset_ranges(test_lengths)
    # test_ranges = [(0, 67), (67, 133), (133, 199), (199, 265), (265, 331)]
    tests_reconstructed = [ds_all[start:end] for start,end in test_ranges]
    trains_reconstructed = [ds_all[0:start] + ds_all[end:] for start,end in test_ranges]
    return trains_reconstructed, tests_reconstructed


# 增加可以读取article单位的函数(2024.9)
@lru_cache(maxsize=None)
def test_articles_by_fold(dataset_index):
    _, tests = ds_5div_reconstructed_with_title()
    prev_title = ''
    text = ''
    raw_datas = []
    start_idx = []
    arts = []
    for idx, item in enumerate(tests[dataset_index]):
        tokens = item[0]
        title = item[-1]
        if title != prev_title:
            arts.append([])
            raw_datas.append((prev_title, text))
            prev_title = title
            text = ''
            start_idx.append(idx)
        text += ''.join(tokens)
        arts[-1].append(item)
    return arts

# 增加可以读取article单位的函数(2024.9)
@lru_cache(maxsize=None)
def train_articles_by_fold(dataset_index):
    trains, _ = ds_5div_reconstructed_with_title()
    prev_title = ''
    text = ''
    raw_datas = []
    start_idx = []
    arts = []
    for idx, item in enumerate(trains[dataset_index]):
        tokens = item[0]
        title = item[-1]
        if title != prev_title:
            arts.append([])
            raw_datas.append((prev_title, text))
            prev_title = title
            text = ''
            start_idx.append(idx)
        text += ''.join(tokens)
        arts[-1].append(item)
    return arts


# NOTE: 通过文本检索原来的case
def text_to_flatten_index(text):
    res = []
    sents_with_meta = get_sents_with_meta()
    for idx, (sent, meta) in enumerate(sents_with_meta):
        if re.search(text, sent) is not None:
            res.append((sent, meta, idx))
    return res

############################# dataset info #########################
def cal_emphasize_rate_5div():
    trs, tes = ds_5div_reconstructed_with_title()
    res = []
    for ds in trs:
        dd = cal_emphasize_rate(ds)
        res.append(dd)
    return res

def get_flatten_labels(ds):
    ls = []
    for item in ds:
        ls += item[1]
    return ls

