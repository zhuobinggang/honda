import torch
import numpy as np
from taku_subword_expand import Sector, encode_plus
from taku_reader3 import ds_5div_reconstructed_with_title
from common import train_and_save_checkpoints

class Sector_Title_Append(Sector):
    def get_tokens(self, item):
        return item[0]
    def get_title(self, item):
        return item[-1]
    def forward(self, item):
        ids, heads = encode_title_append(self.toker, self.get_tokens(item), self.get_title(item))
        # TODO: 给title编码，然后追加在后边
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp

def encode_title_append(toker, tokens, title):
     ids_title = toker.encode(title ,add_special_tokens = False)
     ids_title.append(3) # 增加[SEP]
     ids_title = torch.LongTensor(ids_title)
     ids, heads = encode_plus(tokens, toker)
     ids_concat = torch.cat((ids, ids_title))
     return ids_concat, heads

def run(seed = 10, indexs = range(3)):
    torch.manual_seed(seed)
    np.random.seed(seed)
    trains_reconstructed, tests_reconstructed = ds_5div_reconstructed_with_title()
    for repeat in indexs:
        for idx, (mess_train_dev, ds_test) in enumerate(zip(trains_reconstructed, tests_reconstructed)):
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            m = Sector_Title_Append()
            train_and_save_checkpoints(m, f'SECTOR_TITLE_APPEND_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)

