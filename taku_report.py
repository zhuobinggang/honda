# 6.29写报告用的
from raw_data import *
from functools import lru_cache
import numpy as np
import re

def extract_name(text):
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


def get_last_lines(arts = None):
    arts = arts if arts is not None else article_analyse()
    last_lines = [ss[-1] for ss in arts]
    return last_lines

def get_names(last_lines = None):
    last_lines = last_lines if last_lines is not None else get_last_lines()
    names = [extract_name(line) for line in last_lines]
    return names

def check(names = None, arts = None):
    arts = arts if arts is not None else article_analyse()
    names = names if names is not None else get_names()
    for idx, (name, art) in enumerate(zip(names, arts)):
        if name == '':
            print(f'{idx}: {art[-2]}\n {art[-1]}\n\n')

# 根据人名整理文章数量
def name_dic(names = None, titles = None):
    names = names if names is not None else get_names()
    titles = titles if titles is not None else titles_raw()
    dic = {}
    for idx, (name, title) in enumerate(zip(names, titles)):
        if name not in dic:
            dic[name] = {'art_ids': [],'titles': []}
        dic[name]['art_ids'].append(idx)
        dic[name]['titles'].append(title)
    return dic

# 根据文章数量排列
def rank_art_count_by_dic(dic = None):
    dic = dic if dic is not None else name_dic()
    names = dic.keys()
    counts = [len(dic[name]['titles']) for name in names]
    ranked = list(reversed(sorted(zip(names, counts), key = lambda x: x[1])))
    return ranked

# 将整理的结果保存了
# 读取保存结果
@lru_cache(maxsize=None)
def read_saved_meta(path = '/home/taku/research/honda/data_five/author_rank_by_article_count.meta'):
    f = open(path)
    lines = f.readlines()
    f.close()
    # process
    splits = [item.split(':') for item in lines]
    names = [name.strip() for name, count in splits]
    counts = [int(count.strip()) for name, count in splits]
    return names, counts


def draw_pie_chart(labels, sizes, path = 'dd.png'):
    import matplotlib.pyplot as plt
    import japanize_matplotlib
    plt.clf()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels)
    plt.savefig(path)

def draw_pie_chart_author_article_count(path = 'dd.png'):
    names, counts = read_saved_meta()
    draw_pie_chart(names, counts, path)


# 将331个文章用chatGPT分类了，下面这个函数用于处理
@lru_cache(maxsize=None)
def read_article_categories(path = '/home/taku/research/honda/data_five/article_category.meta'):
    f = open(path)
    lines = f.readlines()
    f.close()
    splits = [item.split('->') for item in lines]
    category = [int(x2.strip()) for x1,x2 in splits]
    return category

def draw_category_pie(categories = None, path = 'category_pie.png'):
    categories = categories if categories is not None else read_article_categories()
    names = ['テクノロジー', '健康・ライフスタイル', 'インターネット・アプリケーション', '仕事・生産性', 'エンターテイメント']
    counts = [sum([1 for cate in categories if cate == i]) for i in range(1, 6)]
    # 将1和3合在一起
    del names[2]
    counts[0] += counts[2]
    del counts[2]
    draw_pie_chart(names, counts, path)

def draw_category_pie_by_author(name = 'AdamDachis', path = 'author_category_pie.png'):
    dic = name_dic()
    if name not in dic:
        print(f'{name}不在dic里面')
        return
    item = dic[name]
    art_ids = item['art_ids']
    categories = np.array(read_article_categories())
    author_cates = categories[art_ids]
    draw_category_pie(author_cates, path)

def title_nouns():
    from fugashi import Tagger
    tagger = Tagger('-Owakati')
    titles = titles_raw()
    nouns = []
    for title in titles:
        nodes = tagger.parseToNodeList(title)
        for node in nodes:
            if node.pos.startswith('名詞,普通名詞'):
                nouns.append(node.surface)
    return nouns

# 通过标题单词中出现的名字做WordCloud
# fugashi -> poor-gpt/lstm里面用过
def draw_word_cloud_by_title_words():
    from wordcloud import WordCloud
    # font_path = '/home/taku/research/japanize-matplotlib/japanize_matplotlib/fonts/ipaexg.ttf'
    # font_path = '/home/taku/Downloads/Hina_Mincho/HinaMincho-Regular.ttf' 
    font_path = '/home/taku/Downloads/ipag.ttf' 
    nouns = title_nouns()
    text = ' '.join(nouns)
    wc = WordCloud(width=960, height=640, background_color="white", font_path=font_path)
    wc.generate(text)
    wc.to_file('wc2.png')


        



