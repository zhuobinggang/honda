class Loader:
    def __init__(self):
        pass
    def read_sentences(self, filename):
        f = open(filename)
        data = []
        sentence = []
        label= []
        title_word = []
        paragraph_info = []
        for line in f:
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence, label, title_word, paragraph_info))
                    sentence = [] # 遇到换行的时候要重置句子
                    label = []
                    title_word = []
                    paragraph_info = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(1 if splits[-1][:-1] != 'O' else 0)
            # TITLE WORD
            title_word.append(True if splits[-2] == 'TITLE_WORD' else False)
            # PARAGRAPH INFO
            paragraph_info.append(int(splits[-3]))
        # 收尾工作
        if len(sentence) >0:
            data.append((sentence, label, title_word)) # 以句子为单位读取
            sentence = []
            label = []
        return data
    def count_paragraphs(self, data):
        count = 0
        for _,_,_,para in data:
            count += 1 if para[0] == 1 else 0
        return count
    def read_tests(self, length=5):
        datas = []
        for i in range(1, 1 + length):
            datas.append(self.read_sentences(f'data_five/{i}/test.txt'))
        return datas
    def read_trains(self, length=5):
        datas = []
        for i in range(1, 1 + length):
            datas.append(self.read_sentences(f'data_five/{i}/train.txt'))
        return datas


# NOTE: 如果是标题就增加【和】
def token_transfer_by_titles(tokens, titles, i, last, then):
    last_is_title = titles[last] if last > -1 else False
    next_is_title = titles[then] if then < len(tokens) else False
    current_is_title = titles[i]
    if current_is_title:
        if not last_is_title: # 唯一需要特殊对待的情况
            # False True 的情况，增加左标记
            tokens[i] = '【' + tokens[i]
        if not next_is_title:
            # True False 的情况，增加右标记
            tokens[i] = tokens[i] + '】'


# NOTE: 如果是强调就增加<u>和</u>在两边
def token_transfer_by_labels(tokens, ls, i, last, then):
    last_is_emphasize = ls[last] if last > -1 else False
    next_is_emphasize = ls[then] if then < len(tokens) else False
    current_is_emphasize = ls[i]
    if current_is_emphasize:
        if not last_is_emphasize: # 唯一需要特殊对待的情况
            # False True 的情况，增加左标记
            tokens[i] = '<u>' + tokens[i]
        if not next_is_emphasize:
            # True False 的情况，增加右标记
            tokens[i] = tokens[i] + '</u>'



# TODO: 根据模型输出的可能性来展示强调结果
# NOTE: 现在全部置换成True和False, 以后估计可以根据可能性来染色
def token_transfer_by_emphasizes(tokens, emphasizes, i, last, then):
    last_is_emphasize = emphasizes[last] if last > -1 else False
    next_is_emphasize = emphasizes[then] if then < len(tokens) else False
    current_is_emphasize = emphasizes[i]
    if current_is_emphasize:
        if not last_is_emphasize: # 唯一需要特殊对待的情况
            # False True 的情况，增加左标记
            tokens[i] = '<span style="background-color:rgba(255, 87, 51, 0.5);">' + tokens[i]
        if not next_is_emphasize:
            # True False 的情况，增加右标记
            tokens[i] = tokens[i] + '</span>'


def print_sentence(item, emphasizes = None):
    tokens, ls, titles, paras = item
    tokens = tokens.copy()
    for i in range(len(tokens)):
        last = i - 1
        then = i + 1
        token_transfer_by_titles(tokens, titles, i, last, then)
        token_transfer_by_labels(tokens, ls, i, last, then)
        if emphasizes:
            token_transfer_by_emphasizes(tokens, emphasizes, i, last, then)
    text = ''.join(tokens)
    # 段落情报
    if paras[0] == 1:
        text = '□' + text
    return text


# NOTE: 将例子通过网页打印
def ds_printer(ds, LENGTH = None):
    if not LENGTH:
        LENGTH = len(ds)
    texts = []
    for i in range(LENGTH):
        text = print_sentence(ds[i])
        texts.append(text)
    return texts


