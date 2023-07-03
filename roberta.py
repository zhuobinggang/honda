from transformers import AutoTokenizer, RobertaModel
from main import Sector_2022
from title_as_append import encode_title_append
from taku_reader3 import ds_5div_reconstructed_with_title


# tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base", use_fast=False)
# tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
# 
# model = RobertaModel.from_pretrained("rinna/japanese-roberta-base")
_, dss_test = ds_5div_reconstructed_with_title()

class Sector_Roberta(Sector_2022):
    def get_tokens(self, item):
        return item[0]
    def get_title(self, item):
        return item[-1]
    def get_ids_and_heads(self, item):
        ids, heads = encode_title_append(self.toker, self.get_tokens(item), self.get_title(item))
        return ids, heads
    def forward(self, item):
        ids, heads = self.get_ids_and_heads(item)
        position_id_tensor = torch.LongTensor([list(range(0, ids.size(0)))])
        out_bert = self.bert(ids.unsqueeze(0).cuda(), position_ids = position_id_tensor.cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp
    def init_bert(self, wholeword=True):
        if wholeword:
            self.bert = RobertaModel.from_pretrained("rinna/japanese-roberta-base")
            self.bert.train()
            self.toker = AutoTokenizer.from_pretrained("rinna/japanese-roberta-base", use_fast=False)
            self.toker.do_lower_case = True  # due to some bug of tokenizer config loading
        else:
            print('NOT SUPPORTED NOW!')
