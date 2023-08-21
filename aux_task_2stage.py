

class Roberta_Title_Append_Aux(Roberta_Title_Append_Crf_Aux):
    def init_hook(self):
        #NOTE: 增加分类头，利用[cls]判断句子是否需要强调, 不能沿用classfier，因为任务不同了 
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.bert_size, 1),
            nn.Sigmoid()
        )
    def loss(self, item):
        out_bert_cls = self.forward(item)
        aux_loss = self.aux_loss(out_bert_cls, self.get_labels_from_input(item))
        return aux_loss
    def forward(self, item):
        ids, heads = self.get_ids_and_heads(item)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state # (1, seq_len + 2, 768)
        out_bert_cls = out_bert[:, 0, :]
        return out_bert_cls
    def test(self, ds, requires_ephasize_number = False):
        return self.aux_test(ds)

