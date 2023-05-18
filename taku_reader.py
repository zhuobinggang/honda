
def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = [] # 遇到换行的时候要重置句子
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label)) # 以句子为单位读取
        sentence = []
        label = []
    return data


test_filename = 'data_five/1/test.txt'
def readfile_with_title_info(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    title_word = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label, title_word))
                sentence = [] # 遇到换行的时候要重置句子
                label = []
                title_word = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
        # TITLE WORD
        title_word.append(True if splits[-2] == 'TITLE_WORD' else False)

    if len(sentence) >0:
        data.append((sentence, label, title_word)) # 以句子为单位读取
        sentence = []
        label = []
    return data
