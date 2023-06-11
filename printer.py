from taku_reader2 import Loader
from taku_title import Sector_Title
import torch


# NOTE: 如果是标题就增加【和】
def token_transfer_by_titles(tokens, titles, i, last, then):
    last_is_title = titles[last] if last > -1 else False
    next_is_title = titles[then] if then < len(tokens) else False
    current_is_title = titles[i]
    if current_is_title:
        if not last_is_title: # 唯一需要特殊对待的情况
            # False True 的情况，增加左标记
            tokens[i] = '【' + tokens[i]
        if not next_is_title:
            # True False 的情况，增加右标记
            tokens[i] = tokens[i] + '】'


# NOTE: 如果是强调就增加<u>和</u>在两边
def token_transfer_by_labels(tokens, ls, i, last, then, left_mark = '<u>', right_mark = '</u>'):
    last_is_emphasize = ls[last] if last > -1 else False
    next_is_emphasize = ls[then] if then < len(tokens) else False
    current_is_emphasize = ls[i]
    if current_is_emphasize:
        if not last_is_emphasize: # 唯一需要特殊对待的情况
            # False True 的情况，增加左标记
            tokens[i] = left_mark + tokens[i]
        if not next_is_emphasize:
            # True False 的情况，增加右标记
            tokens[i] = tokens[i] + right_mark



# TODO: 根据模型输出的可能性来展示强调结果
# NOTE: 现在全部置换成True和False, 以后估计可以根据可能性来染色
def token_transfer_by_emphasizes(tokens, emphasizes, i, last, then):
    last_is_emphasize = emphasizes[last] if last > -1 else False
    next_is_emphasize = emphasizes[then] if then < len(tokens) else False
    current_is_emphasize = emphasizes[i]
    if current_is_emphasize:
        if not last_is_emphasize: # 唯一需要特殊对待的情况
            # False True 的情况，增加左标记
            tokens[i] = '<span style="background-color:rgba(255, 87, 51, 0.5);">' + tokens[i]
        if not next_is_emphasize:
            # True False 的情况，增加右标记
            tokens[i] = tokens[i] + '</span>'


def print_sentence(item, emphasizes = None):
    tokens, ls, titles, paras = item
    tokens = tokens.copy()
    for i in range(len(tokens)):
        last = i - 1
        then = i + 1
        token_transfer_by_titles(tokens, titles, i, last, then)
        token_transfer_by_labels(tokens, ls, i, last, then)
        if emphasizes:
            token_transfer_by_emphasizes(tokens, emphasizes, i, last, then)
    text = ''.join(tokens)
    # 段落情报
    if paras[0] == 1:
        text = '□' + text
    return text

def mark_sentence(item, need_title = True, need_ground_true = False):
    tokens, ls, titles, paras = item
    tokens = tokens.copy()
    for i in range(len(tokens)):
        last = i - 1
        then = i + 1
        if need_title:
            token_transfer_by_titles(tokens, titles, i, last, then)
        if need_ground_true:
            token_transfer_by_labels(tokens, ls, i, last, then, left_mark = '【', right_mark = '】')
    text = ''.join(tokens)
    return text

# NOTE: 将例子通过网页打印
def ds_printer(ds, LENGTH = None):
    if not LENGTH:
        LENGTH = len(ds)
    texts = []
    for i in range(LENGTH):
        text = print_sentence(ds[i])
        texts.append(text)
    return texts

def case_check(item):
    tokens = item[0]
    titles = item[2]
    for token, is_title in zip(tokens, titles):
        print(f'{token} {is_title}')




######################## Script


def script():
    the_path = '/usr01/taku/checkpoint/honda/NORMAL_TITLE_RP0_DS2_step1200_dev0.424_test0.336.checkpoint'
    checkpoint = torch.load(the_path)
    model = Sector_Title()
    model.load_state_dict(checkpoint['model_state_dict'])
    ds = Loader().read_tests(2)[1]
    emphasizes = [model.emphasize(item) for item in ds[:100]]
    texts = [print_sentence(item, empha) for item, empha in zip(ds[:100], emphasizes)]
    return texts
