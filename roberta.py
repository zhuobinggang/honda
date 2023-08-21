from transformers import AutoTokenizer, RobertaModel
from main import Sector_2022
from taku_reader3 import ds_5div_reconstructed_with_title
import torch
from torch import nn
from torchcrf import CRF
import numpy as np
from common import flatten, train_and_save_checkpoints, cal_prec_rec_f1_v2


################## NOTE: 由于roberta会自动加9在后面，需要重新实现encode_plus ################
def roberta_encode_token(token, toker):
    ids = toker.encode(token, add_special_tokens = False)
    while 9 in ids:
        ids.remove(9)
    return ids

def encode_plus(tokens, toker):
    ids_expand = []
    head_indexs = []
    for token in tokens:
        head_indexs.append(len(ids_expand))
        ids = roberta_encode_token(token, toker)
        ids_expand += ids
    # Special Token
    ids_expand = [toker.cls_token_id] + ids_expand + [toker.sep_token_id]
    head_indexs = [idx + 1 for idx in head_indexs]
    return torch.LongTensor(ids_expand), head_indexs


# 将单词分解成subwords, 还要对应上【和】，展开后重新对应回来
def encode_plus_title_mark(toker, tokens, titles):
    # NOTE: 将tokens中所有的【和】换成双重引号
    tokens = [token.replace('【', '『').replace('】', '』') for token in tokens]
    left_marker = roberta_encode_token('【', toker)
    right_marker = roberta_encode_token('】', toker)
    ids_expand = []
    need_indexs = []
    for i in range(len(tokens)):
        # 获取前后的标题情报
        next_index = i + 1
        last_index = i - 1
        last_is_title = titles[last_index] if i > 0 else False
        next_is_title = titles[next_index] if next_index < len(tokens) else False
        current_is_title = titles[i]
        # 根据情况在token前面或者后面加标记，同时要更新need_index
        token = tokens[i]
        ids = roberta_encode_token(token, toker)
        need_index = len(ids_expand)
        if current_is_title:
            if not last_is_title: # 唯一需要特殊对待的情况
                # False True 的情况，增加左标记
                ids = left_marker + ids
                # 更新need_index
                need_index += len(left_marker)
            if not next_is_title:
                # True False 的情况，增加右标记
                ids = ids + right_marker
        need_indexs.append(need_index)
        ids_expand += ids
    # Special Token
    ids_expand = [toker.cls_token_id] + ids_expand + [toker.sep_token_id]
    need_indexs = [idx + 1 for idx in need_indexs]
    return torch.LongTensor(ids_expand), need_indexs

# with title
def encode_title_append(toker, tokens, title):
     ids_title = roberta_encode_token(title, toker)
     ids_title.append(toker.sep_token_id) # 增加[SEP]
     ids_title = torch.LongTensor(ids_title)
     ids, heads = encode_plus(tokens, toker)
     ids_concat = torch.cat((ids, ids_title))
     return ids_concat, heads

# tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base", use_fast=False)
# tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
# 
# model = RobertaModel.from_pretrained("rinna/japanese-roberta-base")

class Sector_Roberta(Sector_2022):
    def init_bert(self, wholeword=True):
        if wholeword:
            self.bert = RobertaModel.from_pretrained("rinna/japanese-roberta-base")
            self.bert.train()
            self.toker = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base", use_fast=False)
            self.toker.do_lower_case = True  # due to some bug of tokenizer config loading
        else:
            print('NOT SUPPORTED NOW!')
    def get_tokens(self, item):
        return item[0]
    def get_labels_from_input(self, item):
        return item[1]
    def get_ids_and_heads(self, item):
        ids, heads = encode_plus(self.get_tokens(item), self.toker) # NO TITLE
        return ids, heads
    def forward(self, item):
        ids, heads = self.get_ids_and_heads(item)
        # TODO: 给title编码，然后追加在后边
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp


class Sector_Roberta_Title_Append(Sector_Roberta):
    def get_title(self, item):
        return item[-1]
    def get_ids_and_heads(self, item):
        ids, heads = encode_title_append(self.toker, self.get_tokens(item), self.get_title(item)) # With title
        return ids, heads

class Sector_Roberta_Title(Sector_Roberta):
    def get_title(self, item):
        return item[2] # NOTE: get_title func is different between TITLE and TITLE2
    def get_ids_and_heads(self, item):
        ids, heads = encode_plus_title_mark(self.toker, self.get_tokens(item), self.get_title(item)) # With title
        return ids, heads



# >>> model.toker.decode(ids)
# '[CLS] w i n d o w s用のシステム設定アプリはたくさんありますが、なかでも米 l h編集部一番のお気に入りは 『ultimatewindowstweaker』です。[SEP] 米lh編集部イチオシのwindowsシステム設定アプリは『ultimatewindowstweaker』[SEP]'
class Sector_Roberta_Title_Append_Crf(Sector_Roberta_Title_Append):
    def init_hook(self):
        self.classifier = nn.Sequential(  # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
            nn.Linear(self.bert_size, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 2)
        )
        self.crf = CRF(2, batch_first=True)
    def get_should_update(self):
        return self.parameters()
    def loss(self, item):
        out_mlp = self.forward(item)  # (1, seq_len, 2)
        tags = torch.LongTensor([self.get_labels_from_input(item)]).cuda()
        loss = -self.crf(out_mlp, tags)
        return loss
    def forward(self, item):
        ids, heads = self.get_ids_and_heads(item)
        # TODO: 给title编码，然后追加在后边
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 2)
        return out_mlp
    # 和printer.py配合
    def emphasize(self, item, threshold = 0.5):
        out_mlp = self.forward(item)  # (1, seq_len, 2)
        out_mlp = self.crf.decode(out_mlp)[0]
        res = [True if res > threshold else False for res in out_mlp]
        return res
    def test(self, ds, requires_ephasize_number = False):
        target_all = []
        result_all = []
        for item in ds:
            target_all.append(self.get_labels_from_input(item))
            out_mlp = self.forward(item)  # (1, seq_len, 2)
            result_all.append(self.crf.decode(out_mlp)[0])
        # flatten & calculate
        results = flatten(result_all)
        targets = flatten(target_all)
        if requires_ephasize_number:
            return cal_prec_rec_f1_v2(results, targets), sum(results)
        else:
            return cal_prec_rec_f1_v2(results, targets)

class Sector_Roberta_Title_Crf(Sector_Roberta_Title_Append_Crf):
    def get_title(self, item):
        return item[2] # NOTE: get_title func is different between TITLE and TITLE2
    def get_ids_and_heads(self, item):
        ids, heads = encode_plus_title_mark(self.toker, self.get_tokens(item), self.get_title(item)) # With title
        return ids, heads

class Sector_Roberta_Crf(Sector_Roberta_Title_Append_Crf):
    def get_ids_and_heads(self, item):
        ids, heads = encode_plus(self.get_tokens(item), self.toker) # NO TITLE
        return ids, heads

def run1():
    model = Sector_Roberta()
    ds_trian, ds_test = ds_5div_reconstructed_with_title()
    mess_train_dev = ds_trian[0]
    ds_train = mess_train_dev[:-500]
    ds_dev = mess_train_dev[-500:]
    ds_test = ds_test[0]
    train_and_save_checkpoints(model, f'ROBERTA_TEST_DS0', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)

def run_batch(seed = 10, indexs = range(3), mtype = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_trian_org, ds_test_org = ds_5div_reconstructed_with_title()
    for repeat in indexs:
        for idx, (mess_train_dev, ds_test) in enumerate(zip(ds_trian_org, ds_test_org)):
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            if mtype == 0:
                print('ROBERTA')
                m = Sector_Roberta()
                train_and_save_checkpoints(m, f'ROBERTA_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
            elif mtype == 1:
                print('ROBERTA_TITLE')
                m = Sector_Roberta_Title()
                train_and_save_checkpoints(m, f'ROBERTA_TITLE_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
            elif mtype == 2:
                print('ROBERTA_TITLE_CRF')
                m = Sector_Roberta_Title_Crf()
                train_and_save_checkpoints(m, f'ROBERTA_TITLE_CRF_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
            elif mtype == 3:
                print('ROBERTA_TITLE_APPEND')
                m = Sector_Roberta_Title_Append()
                train_and_save_checkpoints(m, f'ROBERTA_TITLE_APPEND_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
            elif mtype == 4:
                print('ROBERTA_TITLE_APPEND_CRF')
                m = Sector_Roberta_Title_Append_Crf()
                train_and_save_checkpoints(m, f'ROBERTA_TITLE_APPEND_CRF_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
            elif mtype == 5:
                print('ROBERTA_CRF')
                m = Sector_Roberta_Crf()
                train_and_save_checkpoints(m, f'ROBERTA_CRF_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)

def run_all_1():
    run_batch(mtype = 0)
    run_batch(mtype = 1)
    run_batch(mtype = 2)

def run_all_2():
    # run_batch(mtype = 3)
    # run_batch(mtype = 4)
    run_batch(mtype = 5)


