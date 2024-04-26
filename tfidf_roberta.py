from roberta import Sector_Roberta_Title_Append, encode_title_append, roberta_encode_token, encode_plus, Sector_Roberta
import torch
from common import flatten, train_and_save_checkpoints, cal_prec_rec_f1_v2, topk_tokens

# printed: [CLS] まだまだ寒い日が続きますが、本格的な春は刻 一刻と近づいてきています。[SEP] 辛い症状に負けないための「花粉症・アレルギー対策」特集を行います![SEP] 刻,一刻,春[SEP]
def encode_title_append_topk_tokens_append(toker, tokens, title, topk_tokens):
    # title encode
    ids_title = roberta_encode_token(title, toker)
    ids_title.append(toker.sep_token_id) # 增加[SEP]
    ids_title = torch.LongTensor(ids_title)
    # topk tokens encode
    ids_topk_tokens = ','.join(topk_tokens)
    ids_topk_tokens = roberta_encode_token(ids_topk_tokens, toker)
    ids_topk_tokens.append(toker.sep_token_id) # 增加[SEP]
    ids_topk_tokens = torch.LongTensor(ids_topk_tokens)
    # original sentence encode
    ids, heads = encode_plus(tokens, toker)
    # concat
    ids_concat = torch.cat((ids, ids_title, ids_topk_tokens))
    # print(toker.decode(ids_concat))
    return ids_concat, heads

# NO CRF(因为CRF实际上不太值得信用，考虑一下)
class Sector_Roberta_Title_Append_Tfidf_Append(Sector_Roberta):
    def get_title(self, item):
        return item[4] # tokens, labels, is_title, para_first, title, tf_idf
    def get_top3_tfidf_tokens(self, item):
        tfidf_scores = item[5] # tokens, labels, is_title, para_first, title, tf_idf
        tokens = item[0]
        return topk_tokens(tokens, tfidf_scores, 3)
    def get_ids_and_heads(self, item):
        ids, heads = encode_title_append_topk_tokens_append(self.toker, self.get_tokens(item), self.get_title(item), self.get_top3_tfidf_tokens(item)) # With title
        return ids, heads

# 2024.4.24：跑一遍看看Sector_Roberta_Title_Append_Tfidf_Append的性能，应该是每个split三次就可以了吧
def run():
    from tfidf import read_and_build_dataset
    for repeat in range(3):
        for split in range(5):
            # Dataset prepare
            train_dev_set, test_set = read_and_build_dataset(split)
            train_dev_set = flatten(train_dev_set)
            trainset = train_dev_set[:-500]
            devset = train_dev_set[-500:]
            testset = flatten(test_set)
            # Train
            m = Sector_Roberta_Title_Append_Tfidf_Append()
            train_and_save_checkpoints(m, f'ROBERTA_TITLE_APPEND_TFIDF_RP{repeat}_DS{split}', trainset, devset, testset, check_step = 300, total_step = 3000)




