import fasttext
from fugashi import Tagger
from taku_reader import read_trains, read_tests

ft = fasttext.load_model('/home/taku/cc.ja.300.bin')
tagger = Tagger('-Owakati')
ds = read_tests(1)[0]
ss, ls, titles = ds[0]
vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in ss]) # (LEN, 300)
rnn = nn.LSTM(300, 300, 1, batch_first = True)


class BILSTM_MEAN(nn.Module):
    def __init__(self, learning_rate = 1e-4):
        super().__init__()
        self.ft = fasttext.load_model('cc.ja.300.bin')
        self.tagger = Tagger('-Owakati')
        self.rnn = nn.LSTM(300, 300, 1, batch_first = True, bidirectional = True)
        self.attention = nn.Sequential(
            nn.Linear(600, 1),
            nn.Softmax(dim = 0),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1200, 2),
        )
        self.adapter = Adapter(600, 100)
        self.CEL = nn.CrossEntropyLoss()
        self.cuda()
        self.opter = optim.AdamW(chain(self.parameters()), lr = learning_rate)
    def forward(model, ss, ls):
        mlp = model.mlp
        vec_cat = model.get_pooled_output(ss, ls) # (1200)
        out = mlp(vec_cat).unsqueeze(0) # (1, 2)
        # loss & step
        tar = torch.LongTensor([ls[2]]).cuda() # (1)
        return out, tar
    def loss(model, out ,tar):
        return model.CEL(out, tar)
    def create_vecs(self, text):
        vecs = torch.stack([torch.from_numpy(self.ft.get_word_vector(str(word))) for word in self.tagger(text)]) # (?, 300)
        vecs, (_, _) = self.rnn(vecs.cuda()) # (?, 600)
        # NOTE: ATT
        vecs = self.attention(vecs) * vecs
        return vecs
    def get_pooled_output(self, ss, ls):
        left_vecs = self.create_vecs(combine_ss(ss[:2]))
        right_vecs = self.create_vecs(combine_ss(ss[2:]))
        assert len(left_vecs.shape) == 2 
        assert left_vecs.shape[1] == 600
        # Mean
        left_vec = left_vecs.mean(dim = 0) # (600)
        right_vec = right_vecs.mean(dim = 0) # (600)
        vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze())) # (1200)
        return vec_cat

# 使用fugashi分词，然后使用fasttext获取emb，然后LSTM结合特征，最后MLP输出结果
# 手稿
def script():
    rnn = nn.LSTM(300, 300, 1, batch_first = True)
    model = nn.Sequential(
        nn.Linear(600, 2),
    )
    CEL = nn.CrossEntropyLoss()
    opter = optim.AdamW(chain(rnn.parameters(), model.parameters()), lr = 1e-3)
    tagger = Tagger('-Owakati')
    ld_train = loader.news.train()
    np.random.seed(10)
    np.random.shuffle(ld_train)
    ss,ls = ld_train[0]
    left = combine_ss(ss[:2])
    right = combine_ss(ss[2:])
    left_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(left)]) # (?, 300)
    right_vecs = torch.stack([torch.from_numpy(ft.get_word_vector(str(word))) for word in tagger(right)]) # (?, 300)
    _, (left_vec, _) = rnn(left_vecs.unsqueeze(0)) # (1, 1, 300)
    _, (right_vec, _) = rnn(right_vecs.unsqueeze(0)) # (1, 1, 300)
    vec_cat = torch.cat((left_vec.squeeze(), right_vec.squeeze()))
    out = model(vec_cat).unsqueeze(0) # (1, 2)
    # loss & step
    tar = torch.LongTensor([ls[2]]) # (1)
    loss = CEL(out, tar)


