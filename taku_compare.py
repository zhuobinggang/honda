from taku_reader2 import Loader, print_sentence
from common import flatten, cal_prec_rec_f1_v2
import numpy as np
# BILSTM + fasttext


def last_sentence_emphasize(ds):
    target_all = []
    result_all = []
    for i in range(len(ds)):
        _, ls, _, _ = ds[i]
        target_all.append(ls)
        j = i + 1
        if j >= len(ds):
            # 将ds[i]的token全部强调
            result_all.append([1] * len(ls))
        else: 
            _, _ , _, paragraph = ds[j]
            if paragraph[0] == 1: # 如果下一个句子是段落的开头
                # 将ds[i]的token全部强调
                result_all.append([1] * len(ls))
            else:
                # 将ds[i]的token全部非强调
                result_all.append([0] * len(ls))
    return result_all, target_all

def first_sentence_emphasize(ds):
    target_all = []
    result_all = []
    for i in range(len(ds)):
        _, ls, _, paras = ds[i]
        target_all.append(ls)
        if paras[0] == 1:
            result_all.append([1] * len(ls))
        else:
            result_all.append([0] * len(ls))
    return result_all, target_all


def performance(func):
    ld = Loader()
    tests = ld.read_tests()
    res = []
    for ds in tests:
        result_all, target_all = func(ds)
        # flatten & calculate
        results = flatten(result_all)
        targets = flatten(target_all)
        res.append(cal_prec_rec_f1_v2(results, targets))
    return res


# 计算一共有多少段落
def script():
    ld = Loader()
    datas = ld.read_tests(1)[0] # test集的第一个
    return ld.count_paragraphs(datas)

def random_emphasize(ds, neck = 0.3):
    target_all = []
    result_all = []
    for item in ds:
        _, ls, _, _ = item
        target_all.append(ls)
        result = []
        for _ in range(len(ls)):
            out = 1 if np.random.rand() < neck else 0
            result.append(out)
        result_all.append(result)
    return result_all, target_all

def try_print():
    ds = Loader().read_tests(1)[0]
    emphasizes, _ = random_emphasize(ds)
    texts = [print_sentence(item, empha) for item, empha in zip(ds, emphasizes)]
    return texts

def print_last_sentence_empha():
    ds = Loader().read_tests(1)[0]
    emphasizes, _ = last_sentence_emphasize(ds)
    texts = [print_sentence(item, empha) for item, empha in zip(ds, emphasizes)]
    return texts

    

