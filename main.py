from itertools import chain
import datetime
from transformers import BertJapaneseTokenizer, BertModel
import numpy as np
from run_ner import readfile
import torch as t
torch = t
numpy = np
nn = t.nn
F = t.nn.functional
from functools import lru_cache

RANDOM_SEEDs = [21, 22, 8, 29, 1648, 1, 2]
DATASET_ORDER_SEED = 0

@lru_cache(maxsize=None)
def get_toker():
    toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    return toker

def label_to_number(case):
    tokens, labels = case
    # tokens = ''.join(tokens)
    numbers = [1 if label != 'O' else 0 for label in labels]
    return tokens, numbers


def create_model_with_seed(seed, cuda, wholeword):
    t.manual_seed(seed)
    np.random.seed(seed)
    m = Sector_2022(cuda=cuda, wholeword=wholeword)
    time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'created model with seed {seed} at time {time_string}')
    return m


def read_test(name='Matono/1/test.txt'):
    data = readfile(name)
    data = [label_to_number(case) for case in data]
    return data


def read_train(name='Matono/1/train.txt'):
    data = readfile(name)
    data = [label_to_number(case) for case in data]
    return data

# def read_tests(length = 1):
#    datas = []
#    start = 1
#    end = 1 + length
#    for i in range(start, end):
#        data = readfile(f'm1/test.txt')
#        data = [label_to_number(case) for case in data]
#        datas.append(data)
#    return datas

# def read_trains(length = 1):
#    datas = []
#    start = 1
#    end = 1 + length
#    for i in range(start, end):
#        data = readfile(f'm1/train.txt')
#        data = [label_to_number(case) for case in data]
#        datas.append(data)
#    return datas


def read_tests(length=5):
    datas = []
    start = 1
    end = 1 + length
    for i in range(start, end):
        data = readfile(f'data_five/{i}/test.txt')
        data = [label_to_number(case) for case in data]
        datas.append(data)
    return datas


def read_trains(length=5):
    datas = []
    start = 1
    end = 1 + length
    for i in range(start, end):
        data = readfile(f'data_five/{i}/train.txt')
        data = [label_to_number(case) for case in data]
        datas.append(data)
    return datas


def flatten(l):
    return [item for sublist in l for item in sublist]


class Sector_2022(nn.Module):
    def __init__(self, learning_rate=2e-5, cuda=True, wholeword=True, last_dim = 768):
        super().__init__()
        self.learning_rate = learning_rate
        self.bert_size = last_dim
        self.verbose = False
        self.init_bert(wholeword=wholeword)
        self.init_hook()
        self.opter = t.optim.AdamW(
            self.get_should_update(),
            self.learning_rate)
        if cuda:
            self.cuda()
        self.is_cuda = cuda
        self.BCE = t.nn.BCELoss()

    def init_bert(self, wholeword=True):
        if wholeword:
            self.bert = BertModel.from_pretrained(
                'cl-tohoku/bert-base-japanese-whole-word-masking')
            self.bert.train()
            self.toker = BertJapaneseTokenizer.from_pretrained(
                'cl-tohoku/bert-base-japanese-whole-word-masking')
        else:
            self.bert = BertModel.from_pretrained(
                'cl-tohoku/bert-base-japanese-char')
            self.bert.train()
            self.toker = BertJapaneseTokenizer.from_pretrained(
                'cl-tohoku/bert-base-japanese-char')

    def get_should_update(self):
        return chain(self.bert.parameters(), self.classifier.parameters())

    def init_hook(self):
        self.classifier = nn.Sequential(  # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
            nn.Linear(self.bert_size, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )

    def get_labels_from_input(self, item):
        return item[1]
    def get_tokens_from_input(self, item):
        return item[0]

    def forward(self, item):
        tokens = self.get_tokens_from_input(item)
        ids = encode(tokens, self.toker)
        assert ids.shape[0] == len(tokens) + 2
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, 1:-1, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp

    def loss(self, item):
        labels = self.get_labels_from_input(item)
        out_mlp = self.forward(item)  # (seq_len)
        labels = t.FloatTensor(self.get_labels_from_input(item)).cuda()  # (seq_len)
        loss = self.BCE(out_mlp, labels)
        return loss

    # 和printer.py配合
    def emphasize(self, item, threshold = 0.5):
        out_mlp = self.forward(item)  # (seq_len)
        out_mlp = out_mlp.tolist()
        res = [True if res > threshold else False for res in out_mlp]
        return res

    def test(self, ds, threshold = 0.5, return_logits = False, requires_ephasize_number = False):
        target_all = []
        result_all = []
        for item in ds:
            out_mlp = self.forward(item)  # (seq_len)
            target_all.append(self.get_labels_from_input(item))
            out_mlp = out_mlp.tolist()
            result_all.append(out_mlp)
        # flatten & calculate
        logits = flatten(result_all)
        results = [1 if res > threshold else 0 for res in logits]
        targets = flatten(target_all)
        score = cal_prec_rec_f1_v2(results, targets)
        if not return_logits and not requires_ephasize_number:
            return score
        else:
            res = (score, )
            if return_logits:
                res += (logits, )
            if requires_ephasize_number:
                res += (sum(results), )
            return res

    def opter_step(self):
        self.opter.step()
        self.opter.zero_grad()


# TODO: 将单词分解成subwords，展开后重新对应回来
def encode(text, toker):
    return t.LongTensor(toker.encode(text))


def focal_loss(o, l, fl_rate=0):
    assert len(l.shape) == 0
    assert len(o.shape) == 0
    pt = o if (l == 1) else (1 - o)
    loss = (-1) * t.log(pt) * t.pow((1 - pt), fl_rate)
    return loss


@torch.no_grad()
def test(m, ds_test_org):
    ds_test = ds_test_org.copy()
    toker = m.toker
    bert = m.bert
    target_all = []
    result_all = []
    for text, labels in ds_test:
        ids = encode(text, toker)
        if m.is_cuda:
            # (1, seq_len + 2, 768)
            out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state
        else:
            # (1, seq_len + 2, 768)
            out_bert = bert(ids.unsqueeze(0)).last_hidden_state
        out_bert = out_bert[:, 1:-1, :]  # (1, seq_len, 768)
        out_mlp = m.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.squeeze()  # (seq_len)
        target_all.append(labels)
        if len(out_mlp.shape) == 0:
            out_mlp = [out_mlp.item()]
            print(text)
            print(labels)
        else:
            out_mlp = out_mlp.tolist()
        result_all.append(out_mlp)
    return result_all, target_all


def train(
        m,
        ds_train_org,
        epoch=1,
        batch=16,
        iteration_callback=None,
        random_seed=True):
    ds_train = ds_train_org.copy()
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    opter = m.opter
    BCE = t.nn.BCELoss()
    for epoch_idx in range(epoch):
        print(f'Train epoch {epoch_idx}')
        ds = None
        if random_seed:
            numpy.random.seed(DATASET_ORDER_SEED)  # 固定训练顺序
            np.random.shuffle(ds_train)
            ds = ds_train
        else:
            ds = ds_train
        for row_idx, (text, labels) in enumerate(ds):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            ids = encode(text, toker)  # TODO: 优化展开后复原
            assert ids.shape[0] == len(text) + 2
            if m.is_cuda:
                # (1, seq_len + 2, 768)
                out_bert = bert(ids.unsqueeze(0).cuda()).last_hidden_state
                labels = t.FloatTensor(labels).cuda()  # (seq_len)
            else:
                # (1, seq_len + 2, 768)
                out_bert = bert(ids.unsqueeze(0)).last_hidden_state
                labels = t.FloatTensor(labels)  # (seq_len)
            out_bert = out_bert[:, 1:-1, :]  # (1, seq_len, 768)
            out_mlp = m.classifier(out_bert)  # (1, seq_len, 1)
            out_mlp = out_mlp[0, :, 0]  # (seq_len)
            loss = BCE(out_mlp, labels)
            loss.backward()
            # backward
            if (row_idx + 1) % batch == 0:
                if iteration_callback is not None:
                    iteration_callback()
                opter.step()
                opter.zero_grad()
    opter.step()
    opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time
    print(delta.seconds)
    return delta.seconds


def cal_prec_rec_f1_v2(results, targets):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for guess, target in zip(results, targets):
        if guess == 1:
            if target == 1:
                TP += 1
            elif target == 0:
                FP += 1
        elif guess == 0:
            if target == 1:
                FN += 1
            elif target == 0:
                TN += 1
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
    balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
    return prec, rec, f1, balanced_acc


def get_results_and_targets(m, ds_test):
    result_all, target_all = test(m, ds_test)
    results = flatten(result_all)
    results = [1 if res > 0.5 else 0 for res in results]
    targets = flatten(target_all)
    return results, targets


def test_chain(m, ds_test):
    results, targets = get_results_and_targets(m, ds_test)
    return cal_prec_rec_f1_v2(results, targets)


def run():
    # ds_train = read_train('data/data_five/2/train.txt')
    # ds_test = read_test('data/data_five/2/test.txt')
    ds_train = read_train('Matono/1/train.txt')
    ds_test = read_test('Matono/1/test.txt')
    results = []
    m = create_model_with_seed(20, cuda=True, wholeword=True)
    for _ in range(5):
        train(
            m,
            ds_train,
            epoch=1,
            batch=16,
            iteration_callback=None,
            random_seed=True)
        result = test_chain(m, ds_test)
        print(result)
        results.append(result)
    return results


def experiment(epoch=5, cuda=True, wholeword=True):
    results_5X5X5 = []
    train_dss = read_trains()
    test_dss = read_tests()
    for _, (ds_train, ds_test) in enumerate(zip(train_dss, test_dss)):
        fs_by_model = []
        for idx in range(5):
            m = create_model_with_seed(RANDOM_SEEDs[idx], cuda, wholeword)
            fs = []
            for e in range(epoch):
                train(
                    m,
                    ds_train,
                    epoch=1,
                    batch=16,
                    iteration_callback=None,
                    random_seed=True)
                result = test_chain(m, ds_test)
                print(result)
                _, _, f, _ = result
                # fs.append(f)
                fs.append(result)
            fs_by_model.append(fs)
        results_5X5X5.append(fs_by_model)
        print('results_5X5X5:')
        print(results_5X5X5)
    return results_5X5X5
