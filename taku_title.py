from taku_reader import read_ds
from taku_subword_expand import Sector
# 使用title word情报

# Plan A: 让BERT处理
ds = read_ds('data_five/1/test.txt')
tokens, ls, titles = ds[0]

class Sector_Title(Sector):
    def forward(self, tokens):
        ids, heads = encode_plus(tokens, self.toker)
        assert len(heads) == len(tokens)
        # (1, seq_len + 2, 768)
        out_bert = self.bert(ids.unsqueeze(0).cuda()).last_hidden_state
        out_bert = out_bert[:, heads, :]  # (1, seq_len, 768)
        out_mlp = self.classifier(out_bert)  # (1, seq_len, 1)
        out_mlp = out_mlp.view(-1)  # (seq_len)
        return out_mlp

# NOTE: 根据toker版本的不同可能会改变

# 将单词分解成subwords, 还要对应上【和】，展开后重新对应回来
def encode_plus_title(toker, tokens, titles):
    left_marker = toker.encode('【', add_special_tokens = False)[0]
    right_marker = toker.encode('】', add_special_tokens = False)[0]
    ids_expand = []
    need_indexs = []
    for i in range(len(tokens)):
        # NOTE: 获取前后的标题情报
        next_index = i + 1
        last_index = i - 1
        last_is_title = titles[last_index] if i > 0 else False
        next_is_title = titles[next_index] if next_index < len(tokens) else False
        current_is_title = titles[i]
        # 根据情况在token前面或者后面加标记，同时要更新need_index
        token = tokens[i]
        ids = toker.encode(token, add_special_tokens = False)
        need_index = len(ids_expand)
        if current_is_title:
            if not last_is_title: # 唯一需要特殊对待的情况
                # False True 的情况，增加左标记
                ids = [left_marker] + ids
                # TODO: 更新need_index
                need_index += 1
            if not next_is_title:
                # True False 的情况，增加右标记
                ids = ids + [right_marker]
        need_indexs.append(need_index)
        ids_expand += ids
    return torch.LongTensor(ids_expand), need_indexs

