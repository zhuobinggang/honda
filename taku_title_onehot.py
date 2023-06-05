from taku_title import Sector_Title, Sector_CRF_Title
from taku_subword_expand import encode_plus
from taku_reader2 import Loader
from common import train_and_save_checkpoints
import torch
import numpy as np

# NOTE: 此时onehot是追加在BERT的输出后面的，相比起标记的方法，BERT对标题情报是无感知的，猜测此时性能应该比标记的方法更差, 除非使用bilstm重新再整合一下标题情报，以引入类似CRF的机制
class Sector_Title_Onehot(Sector_Title):
    def __init__(self):
        super().__init__(last_dim = 770)
    # out_bert: (seq_len, 768)
    # titles: seq_len * [BOOLEAN]
    def add_title_info(self, out_bert, titles):
        titles_one_hot = torch.Tensor([[0, 1] if boolean else [1, 0] for boolean in titles]) # (seq_len, 2)
        return torch.cat((out_bert, titles_one_hot.cuda()), dim = 1)
    def forward(self, item):
        tokens = self.get_tokens_from_input(item)
        ids, heads = encode_plus(tokens, self.toker) # NOTE: 使用taku_subword_expand的encode_plus，而不是使用【】标记的encode方式
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
        out_bert = out_bert[:, heads, :].view(len(tokens), 768)  # (seq_len, 768)
        # NOTE: 增加title info作为one hot向量
        out_bert_plus_title = self.add_title_info(out_bert, self.get_titles_from_input(item)) # (seq_len, 770)
        out_mlp = self.classifier(out_bert_plus_title)  # (seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp

def run_batch(seed = 10, indexs = range(3)):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ld = Loader()
    for repeat in indexs:
        for idx, (mess_train_dev, ds_test) in enumerate(zip(ld.read_trains(5), ld.read_tests(5))):
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            m = Sector_Title_Onehot()
            train_and_save_checkpoints(m, f'BERT_TITLE_ONEHOT_{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 3000)
