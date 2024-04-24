# HOW TO USE: see scrit() function
import numpy as np
from common import flatten
log = np.log

def read_train_test_set(idx):
    from t_test import dataset_5div_article
    test_sets = dataset_5div_article()
    train_set = flatten(test_sets[:idx]) + flatten(test_sets[idx+1:])
    test_set = test_sets[idx]
    return train_set, test_set

# TODO: 读取5分割train_set
def doc_to_tokens(doc):
    return flatten([sentence[0] for sentence in doc])

# 建立词汇表
def build_vocabulary(train_set):
    doc_count = len(train_set)
    voc = {}
    for idx, doc in enumerate(train_set):
        for token in doc_to_tokens(doc):
            if token not in voc:
                voc[token] = [0] * doc_count
            voc[token][idx] = 1
    return voc

# 查询token idf值的函数
def cal_idf(voc, token):
    doc_count = len(voc['今日'])
    appear_in_how_many_docs = sum(voc[token]) if token in voc else 1 # If not apear in training set, then set to 1
    return log(doc_count / appear_in_how_many_docs)


def build_train_and_test(trainset_doc_level, testset_doc_level):
    voc = build_vocabulary(trainset_doc_level) # NOTE: 训练集合等级的voc
    # NOTE: tf值加入train + test集合。为什么不直接把idf也计算了呢？
    for doc in (trainset_doc_level + testset_doc_level):
        term_num_dic, total_token_num = term_num_per_doc(doc)
        for idx, item in enumerate(doc):
            tokens = item[0]
            tfs = [term_num_dic[token] / total_token_num for token in tokens]
            idfs = []
            # TODO: 计算idf
            for token in tokens:
                idfs.append(cal_idf(voc, token))
            tf_idfs = [tf * idf for tf, idf in zip(tfs, idfs)]
            doc[idx] = item + (tf_idfs,)
            # item.append(tfs) # NOTE: 追加tf到训练+测试集的item里面 
    # # 面对一个句子的时候，如何计算tf_idf值
    # for doc in testset_doc_level:
    #     for item in doc:
    #         tfs = item[-1]
    #         for idx, token in enumerate(item[0]):
    #             tf = tfs[idx]
    #             idf = cal_idf(voc, token)
    #             tf_idf = tf * idf
    return trainset_doc_level, testset_doc_level # tf值已得到更新, 但是idf值需要当场计算？
                
def tf(doc):
    tfs = []
    term_num = {}
    tokens = doc_to_tokens(doc)
    total_token_num = len(tokens)
    for token in tokens:
        if token not in term_num:
            term_num[token] = 0
        term_num[token] += 1
    for token in tokens:
        tf = term_num[token] / total_token_num
        tfs.append(tf)
    return tfs

def term_num_per_doc(doc):
    term_num_dic = {}
    tokens = doc_to_tokens(doc)
    total_token_num = len(tokens)
    for token in tokens:
        if token not in term_num_dic:
            term_num_dic[token] = 0
        term_num_dic[token] += 1
    return term_num_dic, total_token_num


# Usage
def script():
    trainset_doc_level, testset_doc_level = read_train_test_set(0) # 0 ~ 4, 5 splits
    trainset_doc_level, testset_doc_level = build_train_and_test(trainset, testset)
    # get topk tokens by tf_idf score
    from common import topk_tokens
    tokens = trainset_doc_level[0][0][0]
    tf_idfs = trainset_doc_level[0][0][5]
    tokens = topk_tokens(tokens, tf_idfs, 3)
    print(tokens)

    

