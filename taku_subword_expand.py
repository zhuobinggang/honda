from main import read_trains, read_tests, Sector_2022
from main_crf import Sector_2022_CRF
from common import train_and_save_checkpoints
import torch
import numpy as np

def save_checkpoint(name, model, step, score_dev, score_test):
    PATH = f'/usr01/taku/checkpoint/honda/{name}_step{step + 1}_dev{round(score_dev[2], 3)}_test{round(score_test[2], 3)}.checkpoint'
    score = {'dev': score_dev, 'test': score_test}
    torch.save({
            'model_state_dict': model.state_dict(),
            'score': score,
            'step': step + 1
            }, PATH)

class Sector(Sector_2022):
    def forward(self, item):
        tokens, _ = item
        ids, heads = encode_plus(tokens, self.toker)
        assert len(heads) == len(tokens)
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp

class Sector_CRF(Sector_2022_CRF):
    def forward(self, item):
        tokens, _ = item
        ids, heads = encode_plus(tokens, self.toker)
        assert len(heads) == len(tokens)
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # NOTE: (1, seq_len, 2)
        return out_mlp


# 将单词分解成subwords，展开后重新对应回来
# NOTE: 已通过测试
def encode_plus(tokens, toker):
    ids_expand = []
    head_indexs = []
    for token in tokens:
        head_indexs.append(len(ids_expand))
        ids = toker.encode(token, add_special_tokens = False)
        ids_expand += ids
    # Special Token
    ids_expand = [2] + ids_expand + [3]
    head_indexs = [idx + 1 for idx in head_indexs]
    return torch.LongTensor(ids_expand), head_indexs


def run(seed = 10, indexs = range(3, 10)):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 5 * 10 * 2 * 400 * 3 = 120GB
    for repeat in indexs:
        for idx, (mess_train_dev, ds_test) in enumerate(zip(read_trains(), read_tests())):
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            m = Sector()
            train_and_save_checkpoints(m, f'NORMAL_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
            m = Sector_CRF()
            train_and_save_checkpoints(m, f'CRF_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)

