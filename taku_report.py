# 6.29写报告用的
from raw_data import *
arts = article_analyse()
last_lines = [ss[-1] for ss in arts]
import re
def dd(text):
    # patten 1
    pattern = re.match('\w+', text, flags = re.ASCII)
    if pattern:
        name = pattern[0]
        return name
    # patten 2
    pattern = re.match('（原文／訳：(.*?)）', text)
    if pattern:
        name = pattern[1]
        return name
    # patten 3
    pattern = re.match('（(.*?)）', text)
    if pattern:
        name = pattern[1]
        return name
    return ''


names = [dd(line) for line in last_lines]
def check(names, arts):
    for idx, (name, art) in enumerate(zip(names, arts)):
        if name == '':
            print(f'{idx}: {art[-2]}\n {art[-1]}\n\n')

titles = titles_raw()
# 根据人名整理文章数量
def name_dic(names, titles):
    dic = {}
    for idx, (name, title) in enumerate(zip(names, titles)):
        if name not in dic:
            dic[name] = {'art_ids': [],'titles': []}
        dic[name]['art_ids'].append(idx)
        dic[name]['titles'].append(title)
    return dic

# 根据文章数量排列
def rank_art_count_by_dic(dic):
    names = dic.keys()
    counts = [len(dic[name]['titles']) for name in names]
    ranked = list(reversed(sorted(zip(names, counts), key = lambda x: x[1])))
    return ranked

# 将整理的结果保存了
# TODO: 读取保存结果
def read_saved_meta(path = '/home/taku/research/honda/data_five/author_rank_by_article_count.meta'):
    ...





