from main import read_trains, read_tests, Sector_2022
from main_crf import Sector_2022_CRF
from common import train_and_save_checkpoints
import numpy as np
import torch

def data_size_curve():
    ds_trains = read_trains() 
    ds_tests = read_tests()
    ds_test = ds_tests[0] # 1587
    mess_train_dev = ds_trains[0] # 5552
    ds_train = mess_train_dev[:-500] # 5052
    ds_dev = mess_train_dev[-500:] # 500
    m = Sector_2022()
    train_and_save_checkpoints(m, 'VANILLA', ds_train, ds_dev, ds_test)
    m = Sector_2022_CRF()
    train_and_save_checkpoints(m, 'CRF', ds_train, ds_dev, ds_test)

def data_size_curve_5split():
    torch.manual_seed(10)
    np.random.seed(10)
    # 5 * 10 * 2 * 400 * 3 = 120GB
    for repeat in range(3):
        for idx, (mess_train_dev, ds_test) in enumerate(zip(read_trains(), read_tests())):
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            m = Sector_2022()
            train_and_save_checkpoints(m, f'VANILLA_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test)
            m = Sector_2022_CRF()
            train_and_save_checkpoints(m, f'CRF_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test)


