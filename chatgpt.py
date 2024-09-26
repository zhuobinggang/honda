from dataset_info import read_ds_all
from taku_reader3 import ds_5div_reconstructed_with_title
import numpy as np
from printer import mark_sentence
from functools import lru_cache
from common import cal_prec_rec_f1_v2
import re
import csv

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

############### 6.11 用程序来计算强调性能

def flatten(l):
    return [item for sublist in l for item in sublist]

def cal(output_text, article, mark = '**'):
    # TODO: 确保items中每一个token都能对应上output_text
    output_text = output_text.replace(' ', '')
    tokens = flatten([item[0] for item in article])
    start = 0
    emphasize_buff = []
    outputs = []
    for token in tokens:
        rest_text = output_text[start:]
        start += len(token)
        # if re.match(token, rest_text) is None: # 匹配不上，说明有强调或者出错
        if not rest_text.startswith(token): # 匹配不上，说明有强调或者出错
            if token == '。' and rest_text[0] == '.': # 处理句点
                rest_text = '。' + rest_text[1:]
            piece = rest_text[:len(token) + 2]
            if re.search('\*\*', piece) is not None:
                piece_sub_mark = piece.replace('**', '')
                if piece_sub_mark != token:
                    print(f'! {token} not match {piece}')
                    print(rest_text)
                    raise ValueError('piece_sub_mark != token')
                else: # 强调逻辑
                    start += 2 # 算上**的长度
                    if len(emphasize_buff) == 0: # 开区间
                        # print(f'EMPHASIZE START: {token}')
                        emphasize_buff.append(token)
                        outputs.append(1)
                    else: 
                        # 闭区间
                        emphasized_text = ''.join(emphasize_buff)
                        print(f'EMPHASIZE END: {emphasized_text}')
                        emphasize_buff = []
                        outputs.append(0)
            else: 
                print('WRONG OUTPUT TEXT:')
                print(rest_text)
                print(token)
                raise ValueError('匹配不上，还不包含强调符号')
        else: # 匹配成功，输出0
            if len(emphasize_buff) > 0:
                emphasize_buff.append(token)
                outputs.append(1)
            else:
                outputs.append(0)
    if start < len(output_text):
        emphasized_text = ''.join(emphasize_buff)
        print(f'EMPHASIZE END: {emphasized_text}')
    labels = flatten([item[1] for item in article])
    return outputs, labels
        

def test_cal():
    outputs = []
    labels = []
    _, tests = ds_5div_reconstructed_with_title()
    # 手动复制过来
    output_text = 'まだまだ寒い日が続きますが、本格的な春は刻一刻と近づいてきています。春の便りにウキウキ気分をおさえられない方も多いかと思いますが、一方で嬉しくない春の便りもあるもの。そう、「花粉症」です。花粉症、アレルギー性鼻炎、食物アレルギーなどの現代病に悩む人々は年々増加しています。そこでライフハッカーでは、花粉症やアレルギーの症状に悩む方々へ向けて **「花粉症・アレルギー対策」特集** を行います！5日（月）のお昼11時から順次アップしていきますので、乞うご期待ッッ！PhotobyThinkstock/GettyImages.（松井亮太）'
    items = tests[1][0:7]
    o, l = cal(output_text, items)
    outputs += o
    labels += l
    return outputs, labels

def title_text_to_prompt(title, text):
    prompt = f'Please refer to the title and put boldface emphasis on the text you think is important.\nTitle: {title} \nText: {text}'
    return prompt

def raw_data_to_prompts(raw_datas):
    return [title_text_to_prompt(title, text) for title, text in raw_datas]

def generate_prompt_from_arts(arts):
    res = []
    for art in arts:
        text = ''
        title = art[0][-1]
        for item in art:
            tokens = item[0]
            text += ''.join(tokens)
        prompt = title_text_to_prompt(title, text)
        res.append(prompt)
    return res


@lru_cache(maxsize=None)
def generate_prompt_from_dataset(dataset_index):
    _, tests = ds_5div_reconstructed_with_title()
    prev_title = ''
    text = ''
    raw_datas = []
    start_idx = []
    for idx, item in enumerate(tests[dataset_index]):
        tokens = item[0]
        title = item[-1]
        if title != prev_title:
            raw_datas.append((prev_title, text))
            prev_title = title
            text = ''
            start_idx.append(idx)
        text += ''.join(tokens)
    return raw_datas[1:], start_idx


def response_format(res_text):
    text = res_text.replace('<p>','').replace('</p>','').replace('<strong>','**').replace('</strong>','**').replace('\n','').replace(' ', '')
    text = re.sub('Title:.*?Text:', '', text)
    text = re.sub('Text:', '', text)
    return text


def test_as_hand_copy(ds_idx, case_idx, start_idxs, output_text):
    _, tests = ds_5div_reconstructed_with_title()
    # 手动复制过来
    items = tests[ds_idx][start_idxs[case_idx]:start_idxs[case_idx + 1]]
    outputs, labels = cal(output_text, items)
    return outputs, labels

    
# NOTE: 
# random 3 for sample: 19, 43, 50
def run(ds_idx = 1, case_idx = 3, output_text = None):
    datas, starts = generate_prompt_from_dataset(ds_idx)
    prompt = title_text_to_prompt(*datas[case_idx])
    if output_text is None:
        return prompt
    else:
        output_text = response_format(output_text)
        outputs, labels = test_as_hand_copy(ds_idx, case_idx, starts, output_text)
        print(f'{ds_idx},{case_idx},{output_text}')
        return outputs, labels


###### 读取chatgpt_output.csv用以计算结果
# 安部: '/home/taku/research/honda/achive/hitote/abe.csv' 
# 三木: '/home/taku/research/honda/achive/hitote/miki.csv' 
# 南山: '/home/taku/research/honda/achive/hitote/miki.csv' 
# 尾崎: '/home/taku/research/honda/achive/hitote/ozaki.csv' 
# chatgpt: '/home/taku/research/honda/achive/chatgpt/chatgpt_output.csv' 
# gpt4: '/home/taku/research/honda/achive/gpt4/gpt4output.csv' 
# NOTE: need_flatten = False, means will cal scores based on article level, instead of token level
def cal_from_csv(path = './achive/chatgpt_output.csv', need_flatten = True):
    datas = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            datas.append(row)
    # titles = datas[0]
    # datas = datas[1:]
    # 计算
    outputs = []
    labels = []
    for item in datas:
        if len(item) < 1:
            continue
        else:
            ds_idx = item[0]
            case_idx = item[1]
            text = ','.join(item[2:])
            o, l = run(int(ds_idx), int(case_idx), text)
            outputs.append(o)
            labels.append(l)
    if need_flatten:
        outputs = flatten(outputs)
        labels = flatten(labels)
        return cal_prec_rec_f1_v2(outputs, labels)
    else:
        scores = [cal_prec_rec_f1_v2(o, l) for o, l in zip(outputs, labels)] 
        return np.array(scores)



