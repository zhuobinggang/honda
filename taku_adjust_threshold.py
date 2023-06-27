# 通过train数据集调整阈值
from scorer import best_checkpoints
from title_as_append import Sector_Title_Append
from taku_reader3 import ds_5div_reconstructed_with_title, cal_emphasize_rate, get_flatten_labels, cal_emphasize_rate_5div
from common import cal_prec_rec_f1_v2
import numpy as np
import torch
import pickle


def for_title_append():
    res = np.zeros((5, 3, 3))
    result_logits = []
    _, paths = best_checkpoints(type_names = ['SECTOR_TITLE_APPEND'], return_paths = True)
    ds_train, ds_test = ds_5div_reconstructed_with_title()
    for dataset_idx, dataset in enumerate(paths):
        logits_dataset = []
        threshold = 1 - cal_emphasize_rate(ds_train[dataset_idx])
        the_ds_test = ds_test[dataset_idx]
        print(f'Dataset {dataset_idx}: {threshold}')
        for repeat_idx, repeat in enumerate(dataset):
            checkpoint = torch.load(repeat)
            model = Sector_Title_Append()
            model.load_state_dict(checkpoint['model_state_dict'])
            score, logits = model.test(the_ds_test, threshold = threshold, return_logits = True)
            (prec, rec, f, _) = score
            res[dataset_idx, repeat_idx] = [prec, rec, f]
            logits_dataset.append(logits)
        result_logits.append(logits_dataset)
    return res, result_logits

def read_logits():
    f = open('/home/taku/research/honda/logits_for_taku_threshold_adjust_for_title_append.pkl', 'rb')
    logits = pickle.load(f)
    f.close()
    return logits


# logits: (5, 3, ?)
def cal_score_by_logits_and_different_threshold(logits = None):
    # res, logits = for_title_append()
    if logits is None:
        logits = read_logits()
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    res = np.zeros((len(thresholds), 5, 3, 3))
    ds_train, ds_test = ds_5div_reconstructed_with_title()
    for threshold_idx, threshold in enumerate(thresholds):
        for dataset_idx, dataset in enumerate(logits):
            labels = get_flatten_labels(ds_test[dataset_idx])
            for repeat_idx, repeat in enumerate(dataset):
                temp_output = [1 if item > threshold else 0 for item in repeat]
                prec, rec, f, _ = cal_prec_rec_f1_v2(temp_output, labels)
                res[threshold_idx, dataset_idx, repeat_idx] = [prec, rec, f]
    return res
        

def cal_true_rate_by_logits_and_threshold(logits, threshold = 0.5):
    dd = [1 if item > threshold else 0 for item in logits]
    return sum(dd) / len(logits)

def threshold_search_range():
    start = 0.10
    res = []
    for i in range(80):
        thres = start + (i+1) * 0.01
        res.append(thres)
    return res

def enumerate_threshold_and_get_best_fit(single_logits, emphasize_rate):
    distance = 9999
    the_threshold = -1
    for threshold_to_check in threshold_search_range():
        true_rate = cal_true_rate_by_logits_and_threshold(single_logits, threshold_to_check)
        new_distance = abs(emphasize_rate - true_rate)
        if new_distance < distance:
            distance = new_distance
            the_threshold = threshold_to_check
    return the_threshold

def auto_adjust_threshold(logits = None):
    if logits is None:
        logits = read_logits()
    ds_train, ds_test = ds_5div_reconstructed_with_title()
    emphasize_rates = cal_emphasize_rate_5div()
    res = np.zeros((5, 3, 3))
    for dataset_idx, dataset in enumerate(logits):
        labels = get_flatten_labels(ds_test[dataset_idx])
        emphasize_rate = emphasize_rates[dataset_idx]
        for repeat_idx, repeat in enumerate(dataset):
            best_threshold = enumerate_threshold_and_get_best_fit(repeat, emphasize_rate)
            print(f'{dataset_idx}.{repeat_idx}:{best_threshold}')
            adjusted_logits = [1 if item > best_threshold else 0 for item in repeat]
            prec, rec, f, _ = cal_prec_rec_f1_v2(adjusted_logits, labels)
            res[dataset_idx, repeat_idx] = [prec, rec, f]
    return res






