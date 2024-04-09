import numpy as np
log = np.log

# TODO: 读取5分割train_set
def doc_to_tokens(doc):
    pass

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
def idf(token):
    appear_in_how_many_docs = sum(voc[token]) if token in doc else 1 # If not apear in training set, then set to 1
    return log(doc_count / appear_in_how_many_docs)
