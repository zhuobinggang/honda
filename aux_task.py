# 基于2023.8.20之前完成的roberta + title + CRF进一步改造
from roberta import Sector_Roberta_Title_Append_Crf, CRF
import torch
from torch import nn
import numpy as np
from manual import *
from common import train_and_save_checkpoints, cal_prec_rec_f1_v2

# Instance: model = load_first_best_model('ROBERTA_TITLE_APPEND_CRF', Roberta_Title_Append_Crf_Aux, strict = False)
# 增加分类头，利用[cls]判断句子是否需要强调, 不能沿用classfier，因为任务不同了
# 需要重新实现init_hook, get_should_update
class Roberta_Title_Append_Crf_Aux(Sector_Roberta_Title_Append_Crf):
    def init_hook(self):
        self.classifier = nn.Sequential(  # 用于普通任务
            nn.Linear(self.bert_size, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 2)
        )
        self.crf = CRF(2, batch_first=True)
        #NOTE: 增加分类头，利用[cls]判断句子是否需要强调, 不能沿用classfier，因为任务不同了 
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.bert_size, 1),
            nn.Sigmoid()
        )
    def get_should_update(self):
        return self.parameters()
    def aux_task(self, out_bert_cls):
        o = self.aux_classifier(out_bert_cls).squeeze()
        return o
    def aux_task_to_label(self, item, keep_prob = False):
        _, out_bert_cls = self.forward(item, need_cls = True)
        o = self.aux_task(out_bert_cls).item()
        if keep_prob:
            return o
        else:
            return 1 if o > 0.5 else 0
    def aux_loss(self, out_bert_cls, labels):
        l = 1 if sum(labels) > 0 else 0
        o = self.aux_task(out_bert_cls)
        return torch.square(o - l)
    def aux_test(self, dataset):
        preds = []
        trues = []
        for item in dataset:
            pred = self.aux_task_to_label(item)
            true = 1 if sum(self.get_labels_from_input(item)) > 0 else 0
            preds.append(pred)
            trues.append(true)
        return cal_prec_rec_f1_v2(preds, trues)
    def loss(self, item):
        out_mlp, out_bert_cls = self.forward(item, need_cls = True)
        tags = torch.LongTensor([self.get_labels_from_input(item)]).cuda()
        main_loss = -self.crf(out_mlp, tags)
        # aux loss
        aux_loss = self.aux_loss(out_bert_cls, self.get_labels_from_input(item))
        loss = main_loss + aux_loss
        return loss
    def forward(self, item, need_cls = False):
        ids, heads = self.get_ids_and_heads(item)
        # TODO: 给title编码，然后追加在后边
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
        out_bert_cls = out_bert[:, 0, :]
        out_bert_heads = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert_heads)  # (1, seq_len, 2)
        if not need_cls:
            return out_mlp
        else:
             # (1, 768)
            return out_mlp, out_bert_cls


# 结果证明平均f值为0.433666，但是没有辅助损失的0.437，具体看数值也没有特别突出的表现…
# 看看辅助任务的得分有多少，算一下f值
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
                print('aux loss')
                m = Roberta_Title_Append_Crf_Aux()
                train_and_save_checkpoints(m, f'Roberta_Title_Append_Crf_Aux_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)

# 判断一个句子是否需要强调的f值为0.456
# 
# array([[0.47619048, 0.50591017, 0.48962656],
#        [0.42918455, 0.48899013, 0.50257732],
#        [0.39937598, 0.39530988, 0.43980738],
#        [0.44122966, 0.43886097, 0.4389313 ],
#        [0.46153846, 0.43478261, 0.50531915]])
# mean = 0.4565089
# 
# 单纯在辅助任务上训练的f值为0.5，如果将
# 
# array([[0.48186528, 0.50402145, 0.53902185],
#        [0.52655889, 0.53410405, 0.5480427 ],
#        [0.47798742, 0.49034175, 0.52835408],
#        [0.43137255, 0.46153846, 0.50082372],
#        [0.52615845, 0.49324324, 0.4893617 ]])
# mean = 0.50218
# 
def cal_aux_score(instance_func = Roberta_Title_Append_Crf_Aux, checkpoint_name = 'Roberta_Title_Append_Crf_Aux'):
    import common_aux
    return common_aux.cal_aux_score(instance_func, checkpoint_name)

################################ STAGE 2 #################################



