from taku_reader3 import read_ds_all_with_title
from chatgpt import cal_from_csv
from title_as_append import Sector_Title_Append
import torch
from printer import *

model_path = '/usr01/taku/checkpoint/honda/SECTOR_TITLE_APPEND_RP5_DS0_step2700_dev0.409_test0.436.checkpoint'

def get_first_ten_article():
    count = 0
    ds_all = read_ds_all_with_title()
    title = ''
    starts = []
    for idx,item in enumerate(ds_all):
        current_title = item[-1]
        if current_title != title:
            title = current_title
            starts.append(idx)
            count += 1
            if count > 10:
                break
    arts = []
    for i in range(10):
        arts.append(ds_all[starts[i]:starts[i+1]])
    return arts

def load_model():
    model = Sector_Title_Append()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.requires_grad_(False) # Freeze
    return model

def run_model():
    arts = get_first_ten_article()
    items = flatten(arts)
    model = load_model()
    # 打印
    # emphasizes = [model.emphasize(item) for item in arts[0]]
    # texts = [print_sentence(item, empha) for item, empha in zip(arts[0], emphasizes)]
    score = model.test(flatten(arts))
    return score

def run_others():
    path = {
        '安部': '/home/taku/research/honda/achive/hitote/output.csv', 
        'chatgpt': '/home/taku/research/honda/achive/chatgpt/chatgpt_output.csv', 
        'gpt4': '/home/taku/research/honda/achive/gpt4/gpt4output.csv'
    }
    cal_from_csv(path = path['安部'])
    cal_from_csv(path = path['chatgpt'])
    cal_from_csv(path = path['gpt4'])


