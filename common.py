import numpy as np

def save_dic(dic, path = 'dd.dic'):
    import pickle
    pickle.dump(dic, open(path,"wb"))

def load_dic(path = 'dd.dic'):
    import pickle
    return pickle.load(open(path,"rb"))

def flatten(l):
    return [item for sublist in l for item in sublist]

def draw_line_chart(x, ys, legends, path = 'dd.png', colors = None, xlabel = None, ylabel = None):
    import matplotlib.pyplot as plt
    plt.clf()
    for i, (y, l) in enumerate(zip(ys, legends)):
        if colors is not None:
            plt.plot(x[:len(y)], y, colors[i], label = l)
        else:
            plt.plot(x[:len(y)], y, label = l)
    plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(path)


# NOTE: 不再使用，作为原型参考
def save_checkpoint_proto(PATH, model, step, score):
    import torch
    torch.save({
            'model_state_dict': model.m.state_dict(),
            'score': score,
            'step': step
            }, PATH)

def load_checkpoint(PATH):
    import torch
    raise ValueError('手动load,不要用这个函数')
    model = Sector()
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    score = checkpoint['score']
    step = checkpoint['step']
    return model

def get_test_result(model, ld):
    import torch
    with torch.no_grad():
        # Test
        preds = []
        trues = []
        for idx, (ss, ls) in enumerate(ld):
            out, tar = model.forward(ss, ls)
            preds.append(out.argmax().item())
            trues.append(ls[2])
        result = cal_prec_rec_f1_v2(preds, trues)
        return result

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


class Infinite_Dataset():
    def __init__(self, ds, shuffle_seed = None):
        self.counter = 0
        if shuffle_seed is None:
            pass
        else:
            np.random.seed(shuffle_seed)
            np.random.shuffle(ds)
        self.ds = ds

    def next(self):
        item = self.ds[self.counter]
        self.counter += 1
        if self.counter >= len(self.ds):
            self.counter = 0
        return item
        

class Loss_Plotter():
    def __init__(self):
        self.lst = []

    def add(self, item):
        self.lst.append(item)

    def plot(self, path, name = 'loss'):
        draw_line_chart(range(len(self.lst)), [self.lst], [name], path = path)

class Score_Plotter():
    def __init__(self):
        self.l1 = []
        self.l2 = []

    def add(self, item1, item2):
        self.l1.append(item1)
        self.l2.append(item2)

    def plot(self, path, name1 = 'dev', name2 = 'test'):
        draw_line_chart(range(len(self.l1)), [self.l1, self.l2], [name1, name2], path = path)


class ModelWrapper():
    def __init__(self, m):
        self.m = m
    def forward(self, ss, ls):
        return self.m(ss,ls)
    def loss(self, out, tar):
        return self.m.loss(out, tar)
    def opter_step(self):
        self.m.opter.step()
        self.m.opter.zero_grad()

def beutiful_print_result(step, dev_res, test_res):
    dev_prec, dev_rec, dev_f, _ = dev_res
    test_prec, test_rec, test_f, _ = test_res
    print(f'STEP: {step + 1}\nDEV: {round(dev_f, 5)}\nTEST: {round(test_f, 5)}\n\n')


def parameters(model):
    return sum(p.numel() for p in model.parameters())

################## Fasttext ###################
def combine_ss(ss):
    res = ''
    for s in ss:
        if s is not None:
            res += s
    return res

############### 梯度消失确认
def check_gradient(m):
    print(m.self_attn.out_proj.weight.grad)


############## Freeze
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


############# 确认梯度
def check_gradient(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, param.grad.sum())
        else:
            print(name, param.grad)


############# NOTE: 训练过程，因任务的不同而变，专属于NER
def save_checkpoint(name, model, step, score_dev, score_test):
    import torch
    PATH = f'/usr01/taku/checkpoint/honda/{name}_step{step + 1}_dev{round(score_dev[2], 3)}_test{round(score_test[2], 3)}.checkpoint'
    score = {'dev': score_dev, 'test': score_test}
    torch.save({
            'model_state_dict': model.state_dict(),
            'score': score,
            'step': step + 1
            }, PATH)
def train_and_save_checkpoints(
        m, name, ds_train, ds_dev, ds_test, 
        batch_size = 16, check_step = 200, total_step = 2000):
    # Init Model
    # Start
    ld_train = Infinite_Dataset(ds_train.copy(), shuffle_seed = 10)
    loser = Loss_Plotter()
    score_plotter = Score_Plotter()
    best_dev = 0
    for step in range(total_step):
        batch_loss = []
        for _ in range(batch_size):
            loss = m.loss(ld_train.next())
            loss.backward()
            batch_loss.append(loss.item())
        loser.add(np.mean(batch_loss))
        m.opter_step()
        if (step + 1) % check_step == 0: # Evalue
            score_dev = m.test(ds_dev.copy())
            score_test = m.test(ds_test.copy())
            beutiful_print_result(step, score_dev, score_test)
            if score_dev[2] > best_dev:
                best_dev = score_dev[2]
                save_checkpoint(name, m, step, score_dev, score_test)
            # Plot & Cover
            loser.plot(f'checkpoint/{name}_loss.png')
            score_plotter.add(score_dev[2], score_test[2])
            score_plotter.plot(f'checkpoint/{name}_score.png')

################# 检查数据集中没有【, 】
# NOTE: 结果是有，所以在处理的时候
def check_dataset():
    train = read_trains(1)[0]
    test = read_tests(1)[0]
    ds = train + test
    for item in ds:
        tokens, ls = item
        text = ''.join(tokens)
        if '【' in text or '】' in text:
            print(text)


def topk_tokens(tokens, scores, k):
    scores = np.array(scores)
    max_idxs = scores.argsort()[::-1][:k]
    return [tokens[idx] for idx in max_idxs]


