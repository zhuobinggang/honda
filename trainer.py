import numpy as np

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


def draw_line_chart(x, ys, legends, path='./logs/dd.png', colors=None, xlabel=None, ylabel=None):
    import matplotlib.pyplot as plt
    plt.clf()
    for i, (y, l) in enumerate(zip(ys, legends)):
        if colors:
            plt.plot(x[:len(y)], y, colors[i], label=l)
        else:
            plt.plot(x[:len(y)], y, label=l)
    plt.legend()
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    plt.savefig(path)

def beutiful_print_result(step, dev_res, test_res):
    dev_prec, dev_rec, dev_f, _ = dev_res
    test_prec, test_rec, test_f, _ = test_res
    print(f'STEP: {step + 1}\nDEV: {round(dev_f, 5)}\nTEST: {round(test_f, 5)}\n\n')

class ModelWrapper:
    def __init__(self, m):
        self.m = m
        self.model = m
        self.path_prefix = '/usr01/taku/checkpoint/bbc_news_emphasis/'
        self.suffix = '.checkpoint'
        self.fold_index = 999
        self.repeat_index = 999
        self.name = 'unknown'
        self.meta_setted = False
    def loss(self, item):
        return self.m.get_loss(item)
    def opter_step(self):
        self.m.opter.step()
        self.m.opter.zero_grad()
    def set_meta(self, fold_index = 999, repeat_index = 999):
        self.fold_index = fold_index
        self.repeat_index = repeat_index
        self.name = f'{self.m.get_name()}-fold{fold_index}-repeat{repeat_index}'
        self.meta_setted = True
    def get_name(self):
        assert self.meta_setted
        return self.name
    def test(self, ds): # Exact precision, recall, f-score
        return calc_exact_score(self, ds)
    def predict(self, item):
        return self.m.predict(item)
    def load_checkpoint(self):
        assert self.meta_setted
        import torch
        PATH = f'{self.path_prefix}{self.get_name()}{self.suffix}'
        checkpoint = torch.load(PATH)
        self.m.load_state_dict(checkpoint['model_state_dict'])
    def save_checkpoint(self):
        import torch
        PATH = f'{self.path_prefix}{self.get_name()}{self.suffix}'
        torch.save({'model_state_dict': self.m.state_dict()}, PATH)
    def calc_rouge1(self, ds):
        return calc_rouge1(self, ds)
    def calc_rouges(self, ds):
        return calc_rouges(self, ds)
        

def calc_exact_score(model_wrapper, ds):
    labels = []
    results = []
    for item in ds:
        results.extend(model_wrapper.predict(item))
        labels.extend(item['labels'])
    return cal_prec_rec_f1_v2(results, labels)

class Infinite_Dataset:
    def __init__(self, ds):
        self.ds = ds
        self.counter = 0

    def next(self):
        item = self.ds[self.counter]
        self.counter = (self.counter + 1) % len(self.ds)  # 精简计数器逻辑
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

def fake_test_score():
    return -1.0, -1.0, -1.0, -1.0

class Logger:
    def __init__(self, model_name):
        self.loss_plotter = Loss_Plotter()
        self.score_plotter = Score_Plotter()
        self.batch_log = {'loss': []}
        self.checkpoint_log = {'scores': [], 'log_time': []}
        self.txt_log = ''
        self.model_name = model_name
        self.best_dev = 0
        self.best_test = 0
        self.score_func_dev = calc_exact_score
        self.score_func_test = calc_exact_score
    def start(self):
        import time  # 导入时间模块
        self.start_time = time.time()  # 记录开始时间
    def log_batch(self, batch_loss):
        mean_loss = np.mean(batch_loss)
        self.batch_log['loss'].append(mean_loss)
        self.loss_plotter.add(mean_loss)
    def log_checkpoint(self, dev_score):
        import time  # 导入时间模块
        meta = {'best_score_updated': False}
        self.checkpoint_log['scores'].append({'dev': dev_score, 'test': fake_test_score()})
        self.checkpoint_log['log_time'].append(time.time())
        self.score_plotter.add(dev_score[2], self.best_test)
        if dev_score[2] > self.best_dev:
            self.best_dev = dev_score[2]
            meta['best_score_updated'] = True
        return meta
    def update_best_test(self, test_score):
        self.best_test = test_score[2]
        self.checkpoint_log['scores'][-1]['test'] = test_score
    def end(self):
        import time  # 导入时间模块
        self.end_time = time.time()  # 记录结束时间
        self.score_plotter.plot(f'logs/{self.model_name}_score.png')
        self.loss_plotter.plot(f'logs/{self.model_name}_loss.png')
        self.log_txt()
    def log_txt(self):
        import time  # 导入时间模块
        best_dev = self.best_dev
        best_test = self.best_test
        with open(f'logs/{self.model_name}_log.txt', 'w') as f:
            # 输出批次损失
            # f.write(f'Batch loss: {self.batch_log["loss"]}\n\n')
            total_time = self.end_time - self.start_time  # 计算总时间
            f.write(f'total time: {total_time:.3f}s\n')
            f.write(f'best dev: {best_dev}\n')
            f.write(f'best test: {best_test}\n\n')
            for i, score in enumerate(self.checkpoint_log['scores']):
                log_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(self.checkpoint_log['log_time'][i]))
                f.write(f'# Check point {i}\n')
                f.write(f'log time: {log_time}\n')
                f.write('## dev score\n')
                f.write(f'precision: {score["dev"][0]:.3f}\n')
                f.write(f'recall: {score["dev"][1]:.3f}\n')
                f.write(f'f-score: {score["dev"][2]:.3f}\n')
                f.write('## test score\n')
                f.write(f'precision: {score["test"][0]:.3f}\n')
                f.write(f'recall: {score["test"][1]:.3f}\n')
                f.write(f'f-score: {score["test"][2]:.3f}\n\n')

def fake_test_score_rouge():
    return {'rouge1': {'precision': -1.0, 'recall': -1.0, 'fmeasure': -1.0},
            'rouge2': {'precision': -1.0, 'recall': -1.0, 'fmeasure': -1.0},
            'rougeL': {'precision': -1.0, 'recall': -1.0, 'fmeasure': -1.0}}

class Rouge_Logger(Logger):  # {{ edit_1 }}
    def __init__(self, model_name):
        super().__init__(model_name)  # 调用父类构造函数
        self.score_func_dev = calc_rouge1
        self.score_func_test = calc_rouges
    def log_checkpoint(self, dev_score_rouge):
        import time  # 导入时间模块
        meta = {'best_score_updated': False}
        self.checkpoint_log['scores'].append({'dev': dev_score_rouge, 'test': fake_test_score_rouge()})
        self.checkpoint_log['log_time'].append(time.time())
        dev_score_rouge1_fmeasure = dev_score_rouge['rouge1']['fmeasure']
        self.score_plotter.add(dev_score_rouge1_fmeasure, self.best_test)
        if dev_score_rouge1_fmeasure > self.best_dev:
            self.best_dev = dev_score_rouge1_fmeasure
            meta['best_score_updated'] = True
        return meta
    def update_best_test(self, test_score_rouge):
        self.best_test = test_score_rouge['rouge1']['fmeasure']
        self.checkpoint_log['scores'][-1]['test'] = test_score_rouge
    def log_txt(self):
        import time  # 导入时间模块
        best_dev = self.best_dev
        best_test = self.best_test
        with open(f'logs/{self.model_name}_log.txt', 'a') as f:  # 追加模式
            total_time = self.end_time - self.start_time  # 计算总时间
            f.write(f'total time: {total_time:.3f}s\n')
            f.write(f'best dev (rouge1 f-score): {best_dev:.3f}\n')
            f.write(f'best test (rouge1 f-score): {best_test:.3f}\n\n')
            for i, score in enumerate(self.checkpoint_log['scores']):
                f.write(f'# Check point {i}\n')
                f.write(f'log time: {time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(self.checkpoint_log["log_time"][i]))}\n')
                f.write('## dev score\n')
                f.write(f'rouge1 precision: {score["dev"]["rouge1"]["precision"]:.3f}\n')
                f.write(f'rouge1 recall: {score["dev"]["rouge1"]["recall"]:.3f}\n')
                f.write(f'rouge1 f-score: {score["dev"]["rouge1"]["fmeasure"]:.3f}\n')
                f.write('## test score\n')
                f.write(f'rouge1 precision: {score["test"]["rouge1"]["precision"]:.3f}\n')
                f.write(f'rouge1 recall: {score["test"]["rouge1"]["recall"]:.3f}\n')
                f.write(f'rouge1 f-score: {score["test"]["rouge1"]["fmeasure"]:.3f}\n')
                f.write(f'rouge2 precision: {score["test"]["rouge2"]["precision"]:.3f}\n')
                f.write(f'rouge2 recall: {score["test"]["rouge2"]["recall"]:.3f}\n')
                f.write(f'rouge2 f-score: {score["test"]["rouge2"]["fmeasure"]:.3f}\n')
                f.write(f'rougeL precision: {score["test"]["rougeL"]["precision"]:.3f}\n')
                f.write(f'rougeL recall: {score["test"]["rougeL"]["recall"]:.3f}\n')
                f.write(f'rougeL f-score: {score["test"]["rougeL"]["fmeasure"]:.3f}\n\n')
            
def save_checkpoint(model_wrapper):
    model_wrapper.save_checkpoint()

def calc_f1_scores_by_article(model_wrapper, ds):
    results = calc_rouges(model_wrapper, ds, scorer_keys = ['rouge1'], need_article_level_score = True)
    return [result['rouge1'].fmeasure for result in results]

def calc_rouge1(model_wrapper, ds):
    return calc_rouges(model_wrapper, ds, scorer_keys = ['rouge1'])

def calc_rouges(model_wrapper, ds, scorer_keys = ['rouge1', 'rouge2', 'rougeL'], need_article_level_score = False):
    from rouge_score import rouge_scorer
    import numpy as np
    scorer = rouge_scorer.RougeScorer(scorer_keys)
    results = []
    for item in ds:
        predicted = model_wrapper.predict(item)
        sentences_predicted = ' '.join([sentence for sentence, label in zip(item['sentences'], predicted) if label == 1])
        reference = ' '.join([sentence for sentence, label in zip(item['sentences'], item['labels']) if label == 1])
        result = scorer.score(sentences_predicted, reference)
        results.append(result)
    # 根据scorer_keys输出平均precision, recall, f-score
    if need_article_level_score:
        return results
    avg_results = {}
    for key in scorer_keys:
        avg_results[key] = {
            'precision': np.mean([result[key].precision for result in results]),
            'recall': np.mean([result[key].recall for result in results]),
            'fmeasure': np.mean([result[key].fmeasure for result in results])
        }
    return avg_results

def train_and_plot(
        model_wrapper, trainset, devset, testset, 
        batch_size = 4, check_step = 100, total_step = 2000, logger = None): # 大约4个epoch
    # Init Model
    # Start
    trainset = Infinite_Dataset(trainset.copy())
    if logger is None:
        logger = Logger(model_wrapper.get_name())
    logger.start()
    for step in range(total_step):
        batch_loss = []
        for _ in range(batch_size):
            loss = model_wrapper.loss(trainset.next())
            loss.backward()
            batch_loss.append(loss.item())
        logger.log_batch(batch_loss)
        model_wrapper.opter_step()
        if (step + 1) % check_step == 0: # Evalue
            score_dev = logger.score_func_dev(model_wrapper, devset.copy())
            print(score_dev)
            # Plot & Cover
            meta = logger.log_checkpoint(score_dev)
            if meta['best_score_updated']:
                score_test = logger.score_func_test(model_wrapper, testset.copy())
                logger.update_best_test(score_test)
                model_wrapper.save_checkpoint()
                print('Best score updated!')
    logger.end()