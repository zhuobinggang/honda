# NOTE: 该文件作为整理实验结果的据点
import os
import re
import numpy as np
import torch

directory_in_str = '/usr01/taku/checkpoint/honda/best/vanilla/'
directory_in_str2 = '/usr01/taku/checkpoint/honda/best/crf/'

TRAINED_MODEL_DIR = '/usr01/taku/checkpoint/honda/'
TRAINED_MODEL_DIR_TFIDF = '/usr01/taku/checkpoint/tfidf/'

def dd(directory_in_str):
    directory = os.fsencode(directory_in_str)
    filepaths = [] # 获取文件路径
    filenames = [] # 文件名
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
        filepaths.append(f'{directory_in_str}{filename}')
    # TODO: 匹配 VANILLA_RP?_DS{ds_idx}_* 
    # TODO: 读取checkpoint然后计算平均f, prec, rec
    fs5 = []
    precs5 = []
    recs5 = []   
    for dataset_index in range(5):
        fs = []
        precs = []
        recs = []
        print(f'dataset_index: {dataset_index}')
        for name, path in zip(filenames, filepaths): # 遍历, 无所谓，只是15个文件名字符串处理而已
            pattern = f'.*?_RP._DS{dataset_index}_.*'
            if len(re.findall(pattern, name)) > 0:
                print(f'load: {name}')
                checkpoint = torch.load(path)
                prec, rec, f, _ = checkpoint['score']['test']
                fs.append(f)
                precs.append(prec)
                recs.append(rec)
        fs5.append(fs)
        precs5.append(precs)
        recs5.append(recs)
    return precs5, recs5, fs5



# NOTE: 不需要手动，也不需要实际上加载checkpoint
def dd2(directory_in_str = TRAINED_MODEL_DIR_TFIDF, type_names = ['NORMAL', 'CRF', 'NORMAL_TITLE','CRF_TITLE'], repeat_index_range = range(3), return_paths = False):
    directory = os.fsencode(directory_in_str)
    filepaths = [] # 获取文件路径
    filenames = [] # 文件名
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
        filepaths.append(f'{directory_in_str}{filename}')
    res = np.zeros((5, len(type_names), len(repeat_index_range)))
    best_checkpoint_paths = []
    for dataset_index in range(5):
        best_checkpoint_by_dataset = []
        for type_index, type_name in enumerate(type_names):
            for index_counter, repeat_index in enumerate(repeat_index_range):
            # TODO: 找到性能最高的checkpoint
                best_dev = 0
                the_test = 0
                the_name = ''
                the_path = ''
                for name, path in zip(filenames, filepaths): # 遍历, 无所谓，只是15个文件名字符串处理而已
                    pattern = f'{type_name}_RP{repeat_index}_DS{dataset_index}_.*'
                    # if len(re.findall(pattern, name)) > 0:
                    if re.match(pattern, name): # 不同step的处理，根据文件名读取数字
                        dev = float(re.findall('dev(0\.?\d*)', name)[0])
                        if dev > best_dev:
                            the_name = name
                            the_path = path
                            best_dev = dev
                            test = float(re.findall('test(0\.?\d*)', name)[0])
                            if test < the_test:
                                print(f'! {test} is smaller than {the_test}')
                            the_test = test
                print(the_name)
                best_checkpoint_by_dataset.append(the_path)
                res[dataset_index, type_index, index_counter] = the_test
        best_checkpoint_paths.append(best_checkpoint_by_dataset) 
    if return_paths:
        return res, best_checkpoint_paths
    else:
        return res



# NOTE: 加载模型，计算对应prec, recall
def dd2_all_info(directory_in_str = '/usr01/taku/checkpoint/honda/', type_names = ['NORMAL', 'CRF', 'NORMAL_TITLE','CRF_TITLE'], repeat_index_range = range(3), RP = 'RP'):
    directory = os.fsencode(directory_in_str)
    filepaths = [] # 获取文件路径
    filenames = [] # 文件名
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
        filepaths.append(f'{directory_in_str}{filename}')
    res = np.zeros((5, len(type_names), len(repeat_index_range), 3))
    for dataset_index in range(5):
        for type_index, type_name in enumerate(type_names):
            for index_counter, repeat_index in enumerate(repeat_index_range):
            # TODO: 找到性能最高的checkpoint
                best_dev = 0
                the_test = 0
                the_path = ''
                for name, path in zip(filenames, filepaths): # 遍历, 无所谓，只是15个文件名字符串处理而已
                    pattern = f'{type_name}_{RP}{repeat_index}_DS{dataset_index}_.*'
                    if len(re.findall(pattern, name)) > 0: # 不同step的处理，根据文件名读取数字
                        dev = float(re.findall('dev(0\.?\d*)', name)[0])
                        if dev > best_dev:
                            the_path = path
                            best_dev = dev
                            test = float(re.findall('test(0\.?\d*)', name)[0])
                            if test < the_test:
                                print(f'! {test} is smaller than {the_test}')
                            the_test = test
                print(the_path)
                # TODO: 加载模型获取详细分数
                checkpoint = torch.load(the_path)
                prec, rec, f, _ = checkpoint['score']['test']
                res[dataset_index, type_index, index_counter][0] = prec
                res[dataset_index, type_index, index_counter][1] = rec
                res[dataset_index, type_index, index_counter][2] = f
    return res


############## 获取实验结果的脚本
def script():
    exp1 = dd2(repeat_index_range = range(0,3))
    exp2 = dd2(repeat_index_range = range(3,6))
    exp3 = dd2(repeat_index_range = range(6,9))
    avg = dd2(repeat_index_range = range(9))


# 获取所有结果包括prec, rec, f
def script2():
    # BERT based
    res = dd2_all_info(repeat_index_range = range(9)) # (5, 4, 9, 3), 最后一个维度分解为(prec, rec, f)
    prec_bert = res[:,0,:,0].mean()
    rec_bert = res[:,0,:,1].mean()
    f_bert = res[:,0,:,2].mean()
    # BILSTM based
    res = dd2_all_info(type_names=['BILSTM','BILSTM_TITLE'], repeat_index_range = range(3)) # (5, 2, 3, 3), 最后一个维度分解为(prec, rec, f)
    # Title as append
    res = dd2_all_info(type_names=['SECTOR_TITLE_APPEND'], repeat_index_range = range(9)) # (5, 2, 3, 3), 最后一个维度分解为(prec, rec, f)



############## 通过最高dev值加载模型 ###########
def best_checkpoints(directory_in_str = '/usr01/taku/checkpoint/honda/', type_names = ['NORMAL', 'CRF', 'NORMAL_TITLE','CRF_TITLE'], repeat_index_range = range(3), return_paths = True):
    return dd2(directory_in_str, type_names, repeat_index_range, return_paths)


### 删除所有不是最高的模型
def deletable_redundant_models(directory_in_str = '/usr01/taku/checkpoint/honda/', type_names = ['NORMAL', 'CRF', 'NORMAL_TITLE','CRF_TITLE'], repeat_index_range = range(3), return_paths = False, RP = 'RP'):
    directory = os.fsencode(directory_in_str)
    filepaths = [] # 获取文件路径
    filenames = [] # 文件名
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        filenames.append(filename)
        filepaths.append(f'{directory_in_str}{filename}')
    res = np.zeros((5, len(type_names), len(repeat_index_range)))
    best_checkpoint_paths = []
    deletable_paths = []
    for dataset_index in range(5):
        best_checkpoint_by_dataset = []
        for type_index, type_name in enumerate(type_names):
            for index_counter, repeat_index in enumerate(repeat_index_range):
            # TODO: 找到性能最高的checkpoint
                best_dev = 0
                the_test = 0
                the_name = ''
                the_path = ''
                for name, path in zip(filenames, filepaths): # 遍历, 无所谓，只是15个文件名字符串处理而已
                    pattern = f'{type_name}_{RP}{repeat_index}_DS{dataset_index}_.*'
                    # if len(re.findall(pattern, name)) > 0:
                    if re.match(pattern, name): # 不同step的处理，根据文件名读取数字
                        deletable_paths.append(path)
                        dev = float(re.findall('dev(0\.?\d*)', name)[0])
                        if dev > best_dev:
                            the_name = name
                            the_path = path
                            best_dev = dev
                            test = float(re.findall('test(0\.?\d*)', name)[0])
                            if test < the_test:
                                print(f'! {test} is smaller than {the_test}')
                            the_test = test
                print(the_name)
                best_checkpoint_by_dataset.append(the_path)
                deletable_paths.remove(the_path)
                res[dataset_index, type_index, index_counter] = the_test
        best_checkpoint_paths.append(best_checkpoint_by_dataset) 
    return deletable_paths



