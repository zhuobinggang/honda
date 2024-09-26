# 完成BERTSUM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from torchcrf import CRF

# 首先读取五分割数据集
def read_dataset_by_article_with_devset(fold_index):
    from taku_reader3 import train_articles_by_fold, test_articles_by_fold
    train_dev_merged = train_articles_by_fold(fold_index)
    trains = train_dev_merged[:-20]
    devs = train_dev_merged[-20:]
    tests = test_articles_by_fold(fold_index)
    return trains, devs, tests

def encode_without_title_but_sentences_truncate(tokenizer, sentences):
    token_ids = []
    head_ids = []
    max_tokens_per_sentence = 500 // len(sentences)
    for sentence in sentences:
        encoded = tokenizer.encode(sentence, add_special_tokens=False)
        if len(encoded) > (max_tokens_per_sentence - 2):
            encoded = encoded[:max_tokens_per_sentence - 2]
        encoded = [tokenizer.cls_token_id] + encoded + [tokenizer.sep_token_id]
        token_ids.extend(encoded)
        head_ids.append(len(token_ids) - len(encoded))
    if len(head_ids) != len(sentences):
        print(f"head_ids length: {len(head_ids)}, original_sentences_num: {len(sentences)}")
        print(f"sentences: {sentences}")
        raise ValueError("head_ids和original_sentences_num的长度不匹配，程序终止。")
    if token_ids[head_ids[0]] != tokenizer.cls_token_id:
        raise ValueError("token_ids的第一个元素不是cls_token_id，程序终止。")
    if token_ids[-1] != tokenizer.sep_token_id:
        raise ValueError("token_ids的最后一个元素不是sep_token_id，程序终止。")
    if head_ids[0] != 0:
        raise ValueError("head_ids的第一个元素不是0，程序终止。注：encode_without_title的head_ids的第一个元素应该是0。")
    return token_ids, head_ids

# return size: (sentence_num, 768)
def get_embeddings(bert, token_ids, head_ids):
    # 获取嵌入
    outputs = bert(torch.tensor(token_ids).unsqueeze(0).cuda())
    cls_embeddings = outputs.last_hidden_state[0, head_ids]
    return cls_embeddings


def get_untrained_model_and_tokenizer():
    from transformers import BertJapaneseTokenizer, BertModel
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    return bert, tokenizer

class BERTSUM(nn.Module):
    def __init__(self, learning_rate=2e-5, cuda=True, last_dim = 768):
        super().__init__()
        self.learning_rate = learning_rate
        self.bert_size = last_dim
        self.verbose = False
        self.init_bert()
        self.init_hook()
        self.opter = optim.AdamW(
            self.get_should_update(),
            self.learning_rate)
        if cuda:
            self.cuda()
        self.is_cuda = cuda
        self.loss_fn = nn.CrossEntropyLoss()
        self.set_name()

    def set_name(self):
        self.name = 'BERTSUM'

    def get_sentences(self, article): # tested
        return [''.join(item[0]) for item in article]

    def get_labels(self, article): # tested
        return [1 if sum(item[1]) > 0 else 0 for item in article]

    def forward(self, article):
        sentences = self.get_sentences(article)
        token_ids, head_ids = encode_without_title_but_sentences_truncate(self.tokenizer, sentences)
        embeddings = get_embeddings(self.bert, token_ids, head_ids) # size: (sentence_num, 768)
        binary_class_logits = self.classifier(embeddings) # size: (sentence_num, 2)
        return binary_class_logits

    def init_bert(self):
        bert, tokenizer = get_untrained_model_and_tokenizer()
        self.bert = bert
        self.tokenizer = tokenizer

    def init_hook(self):
        self.classifier = nn.Sequential(  # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
            nn.Linear(self.bert_size, 384),
            nn.LeakyReLU(0.1),
            nn.Linear(384, 2)
        )

    def get_loss(self, article):
        labels = self.get_labels(article)
        binary_class_logits = self.forward(article) # size: (sentence_num, 2)
        labels = torch.LongTensor(labels).cuda() # size: (sentence_num)
        loss = self.loss_fn(binary_class_logits, labels)
        return loss

    def get_should_update(self):
        return self.parameters()
    
    def predict(self, item):
        binary_class_logits = self.forward(item) # size: (sentence_num, 2)
        return torch.argmax(binary_class_logits, dim=1) # size: (sentence_num)

    def get_name(self):
        return self.name

BERTSUM.__name__ = 'BERTSUM'


# =====================  因为很麻烦所以直接在同一个文件写运行代码 ========================

# TODO: 在wrapper里面有针对摘要任务设计的test()函数，需要更换为f值评价
def run():
    from trainer import ModelWrapper, train_and_plot, Rouge_Logger
    for fold_index in range(5):
        # 获取数据集
        train_set, dev_set, test_set = read_dataset_by_article_with_devset(fold_index=fold_index)
        for repeat_index in range(3):
            # 初始化模型
            model_wrapper = ModelWrapper(BERTSUM())
            model_wrapper.set_meta(fold_index=fold_index, repeat_index=repeat_index)
            print(model_wrapper.get_name())
            logger = Rouge_Logger(model_wrapper.get_name())  # 添加日志记录器
            # 训练模型并打印结果
            train_and_plot(model_wrapper, train_set, dev_set, test_set, batch_size=16, check_step=10, total_step=100, logger=logger)  # 添加logger参数
