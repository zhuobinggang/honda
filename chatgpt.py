from dataset_info import read_ds_all
import numpy as np
from printer import mark_sentence

# TODO: 从数据集中随即选取20个例子用chatGPT进行强调，然后人工统计分数。
# NOTE: 随机获取20个case用以测试chatgpt性能
def ds_random_20(seed = 10):
    ds_all = read_ds_all()
    np.random.seed(seed)
    np.random.shuffle(ds_all)
    return ds_all[:20]

def prompt_ds(ds):
    texts = [mark_sentence(item) for item in ds]
    promt = 'This is a sentence in an article, the words in 【】 appear in the title, please find out the words or phrases that may be emphasized in the sentence, if any: '
    texts = [promt + text for text in texts]
    return texts

def prompt_ds2(ds):
    texts = [mark_sentence(item, need_title = False) for item in ds]
    promt = 'This is a sentence from an article, please find out the words or phrases that may be emphasized in the sentence, if any: '
    texts = [promt + text for text in texts]
    return texts

def script():
    ds = ds_random_20()
    texts = prompt_ds(ds)
    trues = [mark_sentence(item, need_title = False, need_ground_true = True) for item in ds]

