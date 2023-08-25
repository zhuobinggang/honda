# 基于2023.8.20之前完成的roberta + title + CRF进一步改造
from aux_task import Roberta_Title_Append_Crf_Aux
import torch
from torch import nn
import numpy as np
from manual import *
from common import train_and_save_checkpoints, cal_prec_rec_f1_v2, flatten

class Roberta_Title_Append_Aux(Roberta_Title_Append_Crf_Aux):
    def init_hook(self):
        #NOTE: 增加分类头，利用[cls]判断句子是否需要强调, 不能沿用classfier，因为任务不同了 
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.bert_size, 1),
            nn.Sigmoid()
        )
    def loss(self, item):
        out_bert_cls = self.forward(item)
        aux_loss = self.aux_loss(out_bert_cls, self.get_labels_from_input(item))
        return aux_loss
    def aux_task_to_label(self, item, keep_prob = False):
        out_bert_cls = self.forward(item)
        o = self.aux_task(out_bert_cls).item()
        if keep_prob:
            return o
        else:
            return 1 if o > 0.5 else 0
    def forward(self, item):
        ids, heads = self.get_ids_and_heads(item)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
        out_bert_cls = out_bert[:, 0, :]
        return out_bert_cls
    def test(self, ds, requires_ephasize_number = False):
        return self.aux_test(ds)
    def test_whole_sentence_emphasis(self, ds):
        target_all = []
        result_all = []
        for item in ds:
            labels = self.get_labels_from_input(item)
            zero_or_one = self.aux_task_to_label(item)
            results = [zero_or_one] * len(labels)
            target_all.append(labels)
            result_all.append(results)
        return cal_prec_rec_f1_v2(flatten(result_all), flatten(target_all))

def run_batch(seed = 10, indexs = range(3), splits_until = 5, mtype = 0):
    from taku_reader3 import ds_5div_reconstructed_with_title
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_trian_org, ds_test_org = ds_5div_reconstructed_with_title()
    for repeat in indexs:
        for idx, (mess_train_dev, ds_test) in list(enumerate(zip(ds_trian_org, ds_test_org)))[:splits_until]:
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            if mtype == 0:
                print('For aux task only')
                m = Roberta_Title_Append_Aux()
                train_and_save_checkpoints(m, f'Roberta_Title_Append_Aux_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)


# 单句强调任务的f值为0.5左右，不知道实用不实用
def cal_aux_score(instance_func = Roberta_Title_Append_Aux, checkpoint_name = 'Roberta_Title_Append_Aux'):
    import common_aux
    return common_aux.cal_aux_score(instance_func, checkpoint_name)

################## 整句强调，看看性能 #######################

def cal_emphasis_score_by_aux_task(instance_func = Roberta_Title_Append_Aux, checkpoint_name = 'Roberta_Title_Append_Aux'):
    import common_aux
    return common_aux.cal_emphasis_score_by_aux_task(instance_func, checkpoint_name)

