import fasttext
import torch
from torchcrf import CRF
import numpy as np
from torch import nn
from torch import optim
from fugashi import Tagger
from taku_reader2 import Loader
from common import train_and_save_checkpoints, flatten, cal_prec_rec_f1_v2

class BILSTM(nn.Module):
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
        self.init_all()
        self.opter = optim.AdamW(self.parameters(), lr = learning_rate)
    def init_all(self):
        self.ft = fasttext.load_model('/home/taku/cc.ja.300.bin')
        self.tagger = Tagger('-Owakati')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True, bidirectional = True)
        self.classifier = nn.Sequential(
            nn.Linear(600, 1),
            nn.Sigmoid()
        )
        self.BCE = nn.BCELoss()
        self.cuda()
    def get_titles_from_input(self, item):
        return item[2]
    def get_labels_from_input(self, item):
        return item[1]
    def get_tokens_from_input(self, item):
        return item[0]
    def forward(self, item):
        tokens = self.get_tokens_from_input(item)
        out_ft = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in tokens]).cuda() # (seq_len, 300)
        out_rnn, (_, _) = self.rnn(out_ft.cuda()) # (?, 600)
        out_mlp = self.classifier(out_rnn)  # (seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp
    def loss(self, item):
        labels = self.get_labels_from_input(item)
        out_mlp = self.forward(item)  # (seq_len)
        labels = torch.FloatTensor(self.get_labels_from_input(item)).cuda()  # (seq_len)
        loss = self.BCE(out_mlp, labels)
        return loss
    # 和printer.py配合
    def emphasize(self, item):
        out_mlp = self.forward(item)  # (seq_len)
        out_mlp = out_mlp.tolist()
        res = [1 if res > 0.5 else 0 for res in out_mlp]
        return res
    def test(self, ds):
        target_all = []
        result_all = []
        for item in ds:
            out_mlp = self.forward(item)  # (seq_len)
            target_all.append(self.get_labels_from_input(item))
            out_mlp = out_mlp.tolist()
            result_all.append(out_mlp)
        # flatten & calculate
        results = flatten(result_all)
        results = [1 if res > 0.5 else 0 for res in results]
        targets = flatten(target_all)
        return cal_prec_rec_f1_v2(results, targets)
    def opter_step(self):
        self.opter.step()
        self.opter.zero_grad()




################## LSTM加标题情报
class BILSTM_TITLE(BILSTM):
    def init_all(self):
        self.ft = fasttext.load_model('/home/taku/cc.ja.300.bin')
        self.tagger = Tagger('-Owakati')
        self.rnn = nn.LSTM(302, 302, 1, batch_first = True, bidirectional = True)
        self.classifier = nn.Sequential(
            nn.Linear(604, 1),
            nn.Sigmoid()
        )
        self.BCE = nn.BCELoss()
        self.cuda()
    def transfer_words(self, item):
        tokens = self.get_tokens_from_input(item)
        titles = self.get_titles_from_input(item)
        out_ft = torch.stack([torch.Tensor(np.concatenate((self.ft.get_word_vector(str(word)), [0, 1] if is_title else [1, 0]))) for word, is_title in zip(tokens, titles)]).cuda()
        return out_ft
    def forward(self, item):
        out_ft = self.transfer_words(item) # (seq_len, 302)
        out_rnn, (_, _) = self.rnn(out_ft.cuda()) # (?, 604)
        out_mlp = self.classifier(out_rnn)  # (seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp


class BILSTM_TITLE_CRF(BILSTM_TITLE):
    def init_all(self):
        self.ft = fasttext.load_model('/home/taku/cc.ja.300.bin')
        self.tagger = Tagger('-Owakati')
        self.rnn = nn.LSTM(302, 302, 1, batch_first = True, bidirectional = True)
        self.classifier = nn.Sequential(
            nn.Linear(604, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 2),
        )
        self.crf = CRF(2, batch_first=True)
        self.cuda()
    def forward(self, item):
        out_ft = self.transfer_words(item) # (seq_len, 302)
        out_rnn, (_, _) = self.rnn(out_ft.cuda()) # (?, 604)
        out_mlp = self.classifier(out_rnn)  # (seq_len, 2)
        return out_mlp
    def loss(self, item):
        labels = self.get_labels_from_input(item)
        out_mlp = self.forward(item)  # (seq_len, 2)
        out_mlp = out_mlp.unsqueeze(0) # (1, seq_len, 2)
        tags = torch.LongTensor([self.get_labels_from_input(item)]).cuda()
        loss = -self.crf(out_mlp, tags)
        return loss
    def emphasize(self, item):
        out_mlp = self.forward(item)  # (seq_len, 2)
        out_mlp = out_mlp.unsqueeze(0)
        out_mlp = self.crf.decode(out_mlp)[0]
        res = [True if res > 0.5 else False for res in out_mlp]
        return res
    def test(self, ds):
        target_all = []
        result_all = []
        for item in ds:
            out_mlp = self.forward(item)  # (seq_len, 2)
            out_mlp = out_mlp.unsqueeze(0)
            out_mlp = self.crf.decode(out_mlp)[0] # (seq_len, 0/1)
            target_all.append(self.get_labels_from_input(item))
            result_all.append(out_mlp)
        # flatten & calculate
        results = flatten(result_all)
        targets = flatten(target_all)
        return cal_prec_rec_f1_v2(results, targets)



######################## scripts ############################
# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# 手稿
def run(seed = 10, indexs = range(3), mtype = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ld = Loader()
    mess_train_dev = ld.read_trains(1)[0]
    ds_train = mess_train_dev[:-500]
    ds_dev = mess_train_dev[-500:]
    ds_test = ld.read_tests(1)[0]
    if mtype == 0:
        m = BILSTM()
        # 5 * 10 * 2 * 400 * 3 = 120GB
        train_and_save_checkpoints(m, f'bilstm', ds_train, ds_dev, ds_test, check_step = 300, total_step = 30000)
    elif mtype == 1:
        print('bilstm title')
        m = BILSTM_TITLE()
        # 5 * 10 * 2 * 400 * 3 = 120GB
        train_and_save_checkpoints(m, f'BILSTM_TITLE', ds_train, ds_dev, ds_test, check_step = 300, total_step = 30000)

def run_batch(seed = 10, indexs = range(3), mtype = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ld = Loader()
    for repeat in indexs:
        for idx, (mess_train_dev, ds_test) in enumerate(zip(ld.read_trains(5), ld.read_tests(5))):
            ds_train = mess_train_dev[:-500]
            ds_dev = mess_train_dev[-500:]
            if mtype == 0:
                m = BILSTM()
                train_and_save_checkpoints(m, f'BILSTM_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 12000)
            elif mtype == 1:
                print('BILSTM_TITLE')
                m = BILSTM_TITLE()
                train_and_save_checkpoints(m, f'BILSTM_TITLE_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 12000)
            elif mtype == 2:
                print('BILSTM_TITLE_CRF')
                m = BILSTM_TITLE_CRF()
                train_and_save_checkpoints(m, f'BILSTM_TITLE_CRF_RP{repeat}_DS{idx}', ds_train, ds_dev, ds_test, check_step = 300, total_step = 12000)

def script():
    # run_batch(mtype = 0) # BILSTM
    run_batch(mtype = 1) # BILSTM + TITLE
    # run_batch(mtype = 2) # BILSTM + TITLE + CRF

