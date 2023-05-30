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


