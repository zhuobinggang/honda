from taku_reader2 import *
from taku_title import Sector_Title
import torch

def script():
    the_path = '/usr01/taku/checkpoint/honda/NORMAL_TITLE_RP0_DS2_step1200_dev0.424_test0.336.checkpoint'
    checkpoint = torch.load(the_path)
    model = Sector_Title()
    model.load_state_dict(checkpoint['model_state_dict'])
    ds = Loader().read_tests(2)[1]
    emphasizes = [model.emphasize(item) for item in ds[:100]]
    texts = [print_sentence(item, empha) for item, empha in zip(ds[:100], emphasizes)]
    return texts
