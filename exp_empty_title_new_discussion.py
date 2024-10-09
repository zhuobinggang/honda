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

def sign_test(after_weights, before_weights):
    import numpy as np
    from scipy import stats
    
    # Define before and after weights
    
    # Calculate differences
    differences = after_weights - before_weights
    
    # Remove zeros (no change)
    differences = differences[differences != 0]
    
    # Calculate the number of positive (weight gain) and negative (weight loss) differences
    n_pos = np.sum(differences > 0)
    n_neg = np.sum(differences < 0)
    
    # We use the smaller of n_pos and n_neg as our test statistic (for a two-tailed test)
    n = np.min([n_pos, n_neg])
    
    # Calculate p-value (two-tailed) using the binomial test
    p_value = stats.binom_test(n, n=n_pos + n_neg, p=0.5, alternative='two-sided')
    
    print(f'p-value: {p_value}')
    
    # Interpret the p-value
    if p_value < 0.05:
        print("We reject the null hypothesis: the diet program appears to have a significant effect on weight.")
    else:
        print("We fail to reject the null hypothesis: the diet program does not appear to have a significant effect on weight.")
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
    emphasized_count = 0
    item_count = 0
    with_title_results = []
    for fold_index in range(5):
        a, b = cal_func(fold_index, with_title_input= True)
        emphasized_count += a
        item_count += b
    print(f'With title: {emphasized_count} / {item_count} = {emphasized_count / item_count * 100 :.2f}%')
    # without title
    emphasized_count = 0
    item_count = 0
    for fold_index in range(5):
        a, b = cal_func(fold_index, with_title_input= False)
        emphasized_count += a
        item_count += b
    print(f'Without title: {emphasized_count} / {item_count} = {emphasized_count / item_count * 100 :.2f}%')
    # 进行wilcoxon检定

def cal_emphasis_rate_batch_sign_test(cal_func):
    # f'{emphasized_sentences_count / len(ds) * 100 :.2f}%'
    # with title
    results = []
    labels = []
    for fold_index in range(5):
        dic = cal_func(fold_index, with_title_input= True)
        results += dic['results']
        labels += dic['labels']
    print(f'With title: {sum(results)} / {len(results)} = {(sum(results) / len(results) * 100) :.2f}%')
    with_title_results = results
    # without title
    results = []
    labels = []
    for fold_index in range(5):
        dic = cal_func(fold_index, with_title_input= False)
        results += dic['results']
        labels += dic['labels']
    print(f'Without title: {sum(results)} / {len(results)} = {(sum(results) / len(results) * 100) :.2f}%')
    no_title_results = results
    # label token percentage
    print(f'Label percentage: {sum(labels)} / {len(labels)} = {(sum(labels) / len(labels)) * 100 :.2f}%')
    # 进行wilcoxon检定
    from scipy import stats
    return stats.wilcoxon(with_title_results, no_title_results)


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


# ================ token level ====================

def cal_token_level_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model, ds = get_model_and_dataset_by_fold(fold_index)
    model.eval()
    emphasized_tokens_count = 0
    dic = {}
    total_tokens_count = 0
    results = []
    labels = []
    with torch.no_grad():
        for item in ds:
            labels += item[1]
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            emphasized_tokens_count += sum(bool_values)
            total_tokens_count += len(bool_values)
            results += [1 if bool_value else 0 for bool_value in bool_values]
    dic['emphasized_tokens_count'] = emphasized_tokens_count
    dic['total_tokens_count'] = total_tokens_count
    dic['results'] = results
    dic['labels'] = labels
    dic['labels_count'] = sum(labels)
    return dic

def cal_token_level_beginning_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model = get_model(fold_index)
    model.eval()
    from taku_reader3 import test_articles_by_fold
    ds = test_articles_by_fold(fold_index)
    emphasized_tokens_count = 0
    total_tokens_count = 0
    results = []
    dic = {}
    labels = []
    with torch.no_grad():
        for art in ds:
            item = art[0]
            labels += item[1]
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            emphasized_tokens_count += sum(bool_values)
            total_tokens_count += len(bool_values)
            results += [1 if bool_value else 0 for bool_value in bool_values]
    dic['emphasized_tokens_count'] = emphasized_tokens_count
    dic['total_tokens_count'] = total_tokens_count
    dic['results'] = results
    dic['labels'] = labels
    dic['labels_count'] = sum(labels)
    return dic


def cal_token_level_ending_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model = get_model(fold_index)
    model.eval()
    from taku_reader3 import test_articles_by_fold
    ds = test_articles_by_fold(fold_index)
    emphasized_tokens_count = 0
    total_tokens_count = 0
    results = []
    dic = {}
    labels = []
    with torch.no_grad():
        for art in ds:
            item = art[-1]
            labels += item[1]
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            emphasized_tokens_count += sum(bool_values)
            total_tokens_count += len(bool_values)
            results += [1 if bool_value else 0 for bool_value in bool_values]
    dic['emphasized_tokens_count'] = emphasized_tokens_count
    dic['total_tokens_count'] = total_tokens_count
    dic['results'] = results
    dic['labels'] = labels
    dic['labels_count'] = sum(labels)
    return dic


def cal_token_level_titleword_emphasis_rate(fold_index, with_title_input= True):
    # 这个函数可以计算空白强调的比率，一旦输出中存在任何一处强调就不需要被纳入统计。
    import torch
    model, ds = get_model_and_dataset_by_fold(fold_index)
    model.eval()
    results = []
    dic = {}
    labels = []
    is_title_words = []
    with torch.no_grad():
        for item in ds:
            labels += item[1]
            item = item if with_title_input else empty_title_item(item)
            bool_values = model.emphasize(item)
            results += [1 if bool_value else 0 for bool_value in bool_values]
            is_title_words += item[2]
    dic['results'] = []
    dic['labels'] = []
    # TODO: 只记录是标题单词的强调与非强调
    for predict, label, is_title in zip(results, labels, is_title_words):
        if not is_title:
            continue
        dic['results'].append(predict)
        dic['labels'].append(label)
    dic['emphasized_tokens_count'] = sum(dic['results'])
    dic['total_tokens_count'] = len(dic['results'])
    dic['labels_count'] = sum(dic['results'])
    return dic