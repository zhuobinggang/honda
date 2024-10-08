# 2024.10.08新增，为了回应查读comment进行新的考察
from scorer import best_checkpoints
from functools import lru_cache

# =============== Fundamental functions ==================
def get_checkpoint_paths(name):
    _, cps = best_checkpoints(directory_in_str = '/usr01/taku/checkpoint/honda/', type_names = [name], return_paths = True)
    return cps

# Add 2024.3.21
def load_model(checkpoint_name, instance_func, fold_index = 0, repeat_index = 0, strict = True):
    import torch
    # BERT
    checkpoints = get_checkpoint_paths(checkpoint_name) # shape as (5, 3)
    model = instance_func()
    checkpoint = torch.load(checkpoints[fold_index][repeat_index])
    model.load_state_dict(checkpoint['model_state_dict'], strict = strict)
    return model

def empty_title_item(item):
    return item[:-1] + ('',)
# =============== Fundamental functions End ==================

# =============== cal_sentence_level_empty_emphasis_rate =================
@lru_cache(maxsize=None)
def get_model(fold_index):
    from roberta import Sector_Roberta_Title_Append_Crf
    return load_model('ROBERTA_TITLE_APPEND_CRF', Sector_Roberta_Title_Append_Crf, fold_index, 0)

def get_model_and_dataset_by_fold(fold_index):
    # 这个函数可以获取一个模型和对应的数据集用以进行进一步的考察实验。
    from taku_reader3 import ds_5div_reconstructed_with_title
    _, tests = ds_5div_reconstructed_with_title()
    return get_model(fold_index), tests[fold_index]

def cal_sentence_level_empty_emphasis_rate(fold_index, with_title_input = True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model, ds = get_model_and_dataset_by_fold(fold_index)
    model.eval()
    emphasized_sentences_count = 0
    with torch.no_grad():
        for item in ds:
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            emphasized_sentences_count += 1 if sum(bool_values) > 0 else 0
    return emphasized_sentences_count, len(ds)

def cal_emphasis_rate_batch(cal_func):
    # f'{emphasized_sentences_count / len(ds) * 100 :.2f}%'
    # with title
    sentences_emphasized_count = 0
    sentences_couht = 0
    for fold_index in range(5):
        a, b = cal_func(fold_index, with_title_input= True)
        sentences_emphasized_count += a
        sentences_couht += b
    print(f'With title: {sentences_emphasized_count} / {sentences_couht} = {sentences_emphasized_count / sentences_couht * 100 :.2f}%')
    # without title
    sentences_emphasized_count = 0
    sentences_couht = 0
    for fold_index in range(5):
        a, b = cal_func(fold_index, with_title_input= False)
        sentences_emphasized_count += a
        sentences_couht += b
    print(f'Without title: {sentences_emphasized_count} / {sentences_couht} = {sentences_emphasized_count / sentences_couht * 100 :.2f}%')

# =============== cal_sentence_level_empty_emphasis_rate END =================

def cal_sentence_level_beginning_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model = get_model(fold_index)
    model.eval()
    from taku_reader3 import test_articles_by_fold
    ds = test_articles_by_fold(fold_index)
    emphasized_sentences_count = 0
    with torch.no_grad():
        for art in ds:
            beginning = art[0]
            beginning = beginning if with_title_input else empty_title_item(beginning)
            bool_values = model.emphasize(beginning)
            emphasized_sentences_count += 1 if sum(bool_values) > 0 else 0
    return emphasized_sentences_count, len(ds)


def cal_sentence_level_ending_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model = get_model(fold_index)
    model.eval()
    from taku_reader3 import test_articles_by_fold
    ds = test_articles_by_fold(fold_index)
    emphasized_sentences_count = 0
    with torch.no_grad():
        for art in ds:
            beginning = art[-1]
            beginning = beginning if with_title_input else empty_title_item(beginning)
            bool_values = model.emphasize(beginning)
            emphasized_sentences_count += 1 if sum(bool_values) > 0 else 0
    return emphasized_sentences_count, len(ds)


def cal_token_level_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model, ds = get_model_and_dataset_by_fold(fold_index)
    model.eval()
    emphasized_tokens_count = 0
    total_tokens_count = 0
    with torch.no_grad():
        for item in ds:
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            emphasized_tokens_count += sum(bool_values)
            total_tokens_count += len(bool_values)
    return emphasized_tokens_count, total_tokens_count

def cal_token_level_beginning_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model = get_model(fold_index)
    model.eval()
    from taku_reader3 import test_articles_by_fold
    ds = test_articles_by_fold(fold_index)
    emphasized_tokens_count = 0
    total_tokens_count = 0
    with torch.no_grad():
        for art in ds:
            item = art[0]
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            emphasized_tokens_count += sum(bool_values)
            total_tokens_count += len(bool_values)
    return emphasized_tokens_count, total_tokens_count


def cal_token_level_ending_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model = get_model(fold_index)
    model.eval()
    from taku_reader3 import test_articles_by_fold
    ds = test_articles_by_fold(fold_index)
    emphasized_tokens_count = 0
    total_tokens_count = 0
    with torch.no_grad():
        for art in ds:
            item = art[-1]
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            emphasized_tokens_count += sum(bool_values)
            total_tokens_count += len(bool_values)
    return emphasized_tokens_count, total_tokens_count