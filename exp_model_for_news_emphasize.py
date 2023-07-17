# 用title_as_append模型来强调新闻
from title_as_append import Sector_Title_Append_CRF
import roberta
from roberta import Sector_Roberta_Title_Append_Crf
from t_test import get_checkpoint_paths
import torch
from functools import lru_cache

def read_raw_lines(path = '/home/taku/research/honda/data_five/news_exp.ds'):
    f = open(path)
    lines = f.readlines()
    f.close()
    return lines

def raw_lines_to_cases(raw_lines = None):
    if not raw_lines:
        raw_lines = read_raw_lines()
    ds = []
    counter = 0
    item = None
    for idx, line in enumerate(raw_lines):
        if len(line.strip()) == 0: 
            if item is not None: # Maybe END of case, but maybe not
                ds.append(item)
                item = None
        elif item is None: # Title
            item = {'idx': counter, 'title': line.strip(), 'paras': []}
            counter += 1
        else:
            ss = line.strip().split('。')
            for i in range(len(ss) - 1):
                ss[i] = ss[i] + '。'
            if len(ss[-1]) < 1:
                del ss[-1]
            item['paras'].append(ss)
    # 收尾工作
    if item is not None: # Maybe END of case, but maybe not
        ds.append(item)
    return ds


class Model(Sector_Title_Append_CRF):
    def test(self):
        print('NOT SUPPORT NOW')
    def get_ids_and_heads(self, item):
        toker = self.toker
        title = item['title']
        text = item['text']
        ids_text = toker.encode(text ,add_special_tokens = False)
        ids_title = toker.encode(title ,add_special_tokens = False)
        ids = [toker.cls_token_id] + ids_text + [toker.sep_token_id] + ids_title + [toker.sep_token_id]
        ids = torch.LongTensor(ids)
        heads = [idx + 1 for idx in list(range(len(ids_text)))]
        return ids, heads

class Model_Roberta(Sector_Roberta_Title_Append_Crf):
    def test(self):
        print('NOT SUPPORT NOW')
    def get_ids_and_heads(self, item):
        toker = self.toker
        ids_text = roberta.roberta_encode_token(item['text'], toker)
        ids_title = roberta.roberta_encode_token(item['title'], toker)
        ids = [toker.cls_token_id] + ids_text + [toker.sep_token_id] + ids_title + [toker.sep_token_id]
        ids = torch.LongTensor(ids)
        heads = [idx + 1 for idx in list(range(len(ids_text)))]
        return ids, heads


def flatten(l):
    return [item for sublist in l for item in sublist]

first_process_ds = raw_lines_to_cases

def second_process_ds(ds_org = None):
    if ds_org is None:
        ds_org = first_process_ds()
    ds = []
    for item_org in ds_org:
        title = item_org['title']
        paras = item_org['paras']
        items = [{'title': title, 'text': s} for s in flatten(paras)]
        ds += items
    return ds

def second_process_ds_by_path(path = '/home/taku/research/honda/data_five/news_exp2.ds'):
    lines = read_raw_lines(path)
    cases = raw_lines_to_cases(lines)
    return second_process_ds(cases)

# 稍微打印一下，从printer.py复制过来的
def token_transfer_by_emphasizes(tokens, emphasizes, i, last, then):
    last_is_emphasize = emphasizes[last] if last > -1 else False
    next_is_emphasize = emphasizes[then] if then < len(tokens) else False
    current_is_emphasize = emphasizes[i]
    if current_is_emphasize:
        if not last_is_emphasize: # 唯一需要特殊对待的情况
            # False True 的情况，增加左标记
            tokens[i] = '【' + tokens[i]
        if not next_is_emphasize:
            # True False 的情况，增加右标记
            tokens[i] = tokens[i] + '】'

def print_sentence(tokens, emphasizes):
    tokens = tokens.copy()
    for i in range(len(tokens)):
        last = i - 1
        then = i + 1
        token_transfer_by_emphasizes(tokens, emphasizes, i, last, then)
    text = ''.join(tokens)
    return text

def get_model_for_test(key = 'SECTOR_TITLE_APPEND_CRF', instance_func = Model):
    checkpoints = get_checkpoint_paths(key)
    path = checkpoints[0][0]
    checkpoint = torch.load(path)
    model = instance_func()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def emphasize(model = None, ds = None):
    if model is None:
        model = get_model_for_test()
    if ds is None:
        ds = second_process_ds()
    texts = []
    for item in ds:
        emphasizes = model.emphasize(item)
        ids, heads = model.get_ids_and_heads(item)
        ids = ids[heads]
        tokens = [model.toker.decode(idx) for idx in ids] 
        texts.append(print_sentence(tokens, emphasizes))
    text = ''.join(texts).replace(' ', '').replace('##', '')
    return text


def ds_without_title(ds):
    res = []
    for item in ds:
        res.append({'title': '', 'text': item['text']})
    return res

def common_script(model, ds, need_title = True):
    if not need_title:
        ds = ds_without_title(ds)
    texts = []
    for item in ds:
        emphasizes = model.emphasize(item)
        ids, heads = model.get_ids_and_heads(item)
        ids = ids[heads]
        tokens = [model.toker.decode(idx) for idx in ids] 
        texts.append(print_sentence(tokens, emphasizes))
    text = ''.join(texts).replace(' ', '').replace('##', '')
    return text

def run_bert():
    model = get_model_for_test()
    ds = second_process_ds_by_path(path = '/home/taku/research/honda/data_five/news_exp2.ds')
    text = common_script(model, ds, need_title = True)
    text2 = common_script(model, ds, need_title = False)

def run_roberta():
    model = get_model_for_test(key = 'ROBERTA_TITLE_APPEND_CRF', instance_func = Model_Roberta)
    ds = second_process_ds_by_path(path = '/home/taku/research/honda/data_five/news_exp2.ds')
    text = common_script(model, ds, need_title = True)
    text2 = common_script(model, ds, need_title = False)







