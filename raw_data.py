from bs4 import BeautifulSoup
from dataset_info import read_ds_all
from functools import lru_cache

# 将数据集的每个case追溯到源文件
# 解析sgml(中间级文件)
PATH_TEXTS = './data_five/CRL_NE_DATA_1.sgml' 
PATH_TITLES = './data_five/CRL_NE_DATA_1_TITLE.sgml' 

def sgml_text(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    text = ''.join(lines)
    # NOTE: 删除非常例外的符号
    text = text.replace('*','')
    return text

def article_raw():
    soup = BeautifulSoup(sgml_text(PATH_TEXTS), 'lxml')
    return soup.find_all('text')

def article_analyse(path = PATH_TEXTS):
    soup = BeautifulSoup(sgml_text(path), 'lxml')
    arts = [article.text.strip().split('\n') for article in soup.find_all('text')]
    # NOTE: strong mark is not analysed
    return arts

# 检查case的数量和sentence的数量一致 -> 不一致，需要找到对不上的地方
def check_case_equal_sentence():
    arts = article_analyse()
    ds = read_ds_all()
    _ = sum([len(art) for art in arts]) # 7457
    _ = len(ds) # 7139
    # 说明按照\n分割的话会多出来些奇怪的句子 -> 可能是空句子，排查一下


def check_empty_sent(): # 好像没有空句子，想当然应该被排除了
    arts = article_analyse()
    ds = read_ds_all()
    for art in arts:
        for sent in art:
            if sent.strip() == '':
                print('!')
    # 那按照长度将sentence和case对比一下，如果不对就停止循环，打印检查一下
    # -> NOTE: arts里边多出来一些孤立的句子，像是:   ■特にすばらしい点
    # -> 看来只能一个一个地删除，手动确认，不然很难解释，现在找不到生成sgml文件的逻辑了
    # -> 包含在<p>里的<a>也被删掉了, 应该是手动删除的吧？


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_processed_data_vs_raw_data(need_flatten = True):
    raws = flatten(article_analyse()) if need_flatten else article_analyse()
    products = [''.join(tokens) for tokens,_,_,_ in read_ds_all()]
    return products, raws

def sentences_length(arts):
    return sum([len(sents) for sents in arts])

def reversible_flatten_list(two_dim_list):
    list_with_meta_info = []
    for dim0_idx, dim0 in enumerate(two_dim_list):
        for dim1_idx, dim1 in enumerate(dim0):
            list_with_meta_info.append((dim1, (dim0_idx, dim1_idx)))
    return list_with_meta_info

# 将article保存为json类似的文件？ -> 不需要了，直接跑一遍比较好，不需要保存中间结果
def delete_until_equal(items, arts, log = True):
    while sentences_length(arts) > len(items):
        # NOTE: 怎么删到article里边？
        # NOTE: sentence idx要对应到(article idx, relative sentence idx)
        # NOTE: 使用reversible_flatten_list技术
        sents_with_meta = reversible_flatten_list(arts)
        for idx, (item, sent_with_meta) in enumerate(zip(items, sents_with_meta)):
            sent, (article_idx, relative_sent_idx) = sent_with_meta
            if len(item) != len(sent) or item != sent: # 即便长度相等也要认真比较一下
                if log:
                    print(f'WARNING {idx}: DIFFERENT SENTENCE!')
                    print(f'{item}')
                    print(f'{sent}\n')
                break
        del arts[article_idx][relative_sent_idx]
    return arts


def check_different_idx(items, arts):
    for idx, (t1, t2) in enumerate(zip(items, flatten(arts))):
        if t1 != t2:
            return idx

def check_deleted_raw_datas():
    items, arts = get_processed_data_vs_raw_data(need_flatten = False)
    arts_deleted = delete_until_equal(items, arts)
    check_different_idx(items, arts_deleted) # 出错，看看是哪里有问题 -> 是判断两个句子相等的算法问题 -> 增加了小小的例外符号排除算法


###################### 解析title sgml ################

@lru_cache(maxsize=None)
def get_sents_with_meta():
    print('Calculating ...')
    items, arts = get_processed_data_vs_raw_data(need_flatten = False)
    arts_deleted = delete_until_equal(items, arts, log = False)
    sents_with_meta = reversible_flatten_list(arts_deleted)
    print('Calculated')
    return sents_with_meta

@lru_cache(maxsize=None)
def titles_raw():
    suffix = ':ライフハッカー［日本版］'
    soup = BeautifulSoup(sgml_text(PATH_TITLES), 'lxml')
    titles = [title.text.strip().replace(suffix, '') for title in soup.find_all('text')]
    return titles

@lru_cache(maxsize=None)
def case_id_to_title(idx):
    sents_with_meta = get_sents_with_meta()
    titles = titles_raw()
    sent, (article_idx, relative_sent_idx) = sents_with_meta[idx]
    return titles[article_idx]


