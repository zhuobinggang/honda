from taku_reader import read_trains, read_tests
from taku_subword_expand import Sector, Sector_CRF
from common import train_and_save_checkpoints
import torch
import numpy as np
# 使用title word情报

# Plan A: 让BERT处理
class Sector_Title(Sector):
    def get_labels_from_input(self, item):
        return item[1]
    def get_tokens_from_input(self, item):
        return item[0]
    def get_titles_from_input(self, item):
        return item[2]
    def forward(self, item):
        tokens = self.get_tokens_from_input(item)
        ids, heads = encode_plus_title(self.toker, tokens, self.get_titles_from_input(item))
        assert len(heads) == len(tokens)
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp

class Sector_CRF_Title(Sector_CRF):
    def get_labels_from_input(self, item):
        return item[1]
    def get_tokens_from_input(self, item):
        return item[0]
    def get_titles_from_input(self, item):
        return item[2]
    def forward(self, item):
        tokens = self.get_tokens_from_input(item)
        ids, heads = encode_plus_title(self.toker, tokens, self.get_titles_from_input(item))
        assert len(heads) == len(tokens)
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # NOTE: (1, seq_len, 2)
        return out_mlp

# 将单词分解成subwords, 还要对应上【和】，展开后重新对应回来
def encode_plus_title(toker, tokens, titles):
    # NOTE: 将tokens中所有的【和】换成双重引号
    tokens = [token.replace('【', '『').replace('】', '』') for token in tokens]
    left_marker = toker.encode('【', add_special_tokens = False)[0]
    right_marker = toker.encode('】', add_special_tokens = False)[0]
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
        ids = toker.encode(token, add_special_tokens = False)
        need_index = len(ids_expand)
        if current_is_title:
            if not last_is_title: # 唯一需要特殊对待的情况
                # False True 的情况，增加左标记
                ids = [left_marker] + ids
                # 更新need_index
                need_index += 1
            if not next_is_title:
                # True False 的情况，增加右标记
                ids = ids + [right_marker]
        need_indexs.append(need_index)
        ids_expand += ids
    # Special Token
    ids_expand = [2] + ids_expand + [3]
    need_indexs = [idx + 1 for idx in need_indexs]
    return torch.LongTensor(ids_expand), need_indexs


def run(seed = 10, indexs = range(3, 10)):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 5 * 10 * 2 * 400 * 3 = 120GB
    for repeat in indexs:
        for idx, (mess_train_dev, ds_test) in enumerate(zip(read_trains(), read_tests())):
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            m = Sector_CRF_Title()
            train_and_save_checkpoints(m, f'CRF_TITLE_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
            m = Sector_Title()
            train_and_save_checkpoints(m, f'NORMAL_TITLE_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)

