# 使い方

```py
from main_crf import *
import numpy as np

result_no_crf, result_with_crf = run_experiment()
result_no_crf = np.array(result_no_crf)
result_with_crf = np.array(result_with_crf)
```
## 結果の並べ方

|      |  model1  |  model2  |  model3  |  model4  |  model5  |
| ---- | ----     | ----     | ----     | ----     | ----     |
|  dataset1  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset2  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset3  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset4  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |
|  dataset5  |  (precsion, recall, f-score, balanced-accuracy)  | ... | ... | ... | ... |

## プログラムの仕組み

このプログラムには 3 つの重要なファイルしかありません。1 つは CRF を使用しない BERT プログラムで、main.py にコードが含まれています。2 番目は CRF を使用する BERT プログラムで、ファイルは main\_crf.py です。 3 番目はrun\_ner.pyで、データセットを読み取るファイルです。

main.py の Sector\_2022 クラスはモデルの本体であり、モデルのすべての情報とパラメーターが含まれています。train はモデルのトレーニングに使用される関数です。test はテストに使用される関数です。 run 関数は単純な例であり、experiment 関数は特定の実験のコードです。

## パラメータ設定

|   name   |  value |
| ---- | ---- |
|  optimizer  |  AdamW |
|  learning rate  |  2e-5 |
|  BERT(for word based dataset)  | cl-tohoku/bert-base-japanese-whole-word-masking |
|  BERT(for character based dataset)  | cl-tohoku/bert-base-japanese-char |


## 初期値設定方法

`main_crf.py`に入って、`RANDOM_SEEDs`というglobal variableがある、５つのモデルは順番でこの値で初期化されている。

`DATASET_ORDER_SEED`はtraining setをshuffleする初期値である。

## Training setのラベル1の割合を計算する方法

```py
from main_crf import *
label_count, label_one = cal_label_percent()
print(label_one / label_count)
```

## スコアだけでなく可能性も取る方法

結果をリストからdictに変換した。
`dic.keys()`を使ってdicのすべてのkeywordを見れる。

```py
result_no_crf_raw, result_with_crf_raw = run_experiment_get_raw_output()
case0_pred = result_no_crf_raw['dataset0']['model0']['y_pred'][0]
print(case0_pred)
# 出力はこんな感じ: [0.11263526231050491, 0.12720605731010437, 0.12502093613147736, 0.11578959226608276, 0.12038103491067886, 0.11955509334802628, 0.10841980576515198, 0.10992693901062012, 0.0982583612203598, 0.13114511966705322, 0.13183309137821198, 0.11970293521881104, 0.080902598798275, 0.05768054723739624, 0.06931666284799576, 0.06536028534173965, 0.05445347726345062, 0.05253376439213753, 0.07992718368768692, 0.13279256224632263, 0.1735897660255432, 0.1821373701095581, 0.11884185671806335, 0.1317211091518402, 0.1243244856595993, 0.12247199565172195, 0.18926098942756653, 0.18819020688533783, 0.1932806372642517, 0.23031292855739594, 0.23812806606292725, 0.17765586078166962, 0.028003821149468422]
case0_true = result_no_crf_raw['dataset0']['model0']['y_true'][0]
```

## 2024.9.25 打印论文中的case

```py
from printer import print_paper_case
print_paper_case(0)
```

## 2024.9.27 使用学习数据集中随机10个case来组装prompt用于缓和one-shot导致的性能下降

读取随机samples

```py
from chatgpt import random_ten_training_cases
for sample, title in random_ten_training_cases(cut = True):
    print(title)
    print(sample)
```

## 2024.9.30 从llm输出中计算分数的方法

1. 检查输出符合要求

主要使用`chatgpt.py`文件中的cal函数，该函数会自动比较文本，如果出现错误会报错，再手动更改原文件以符合要求即可。

```py
from chatgpt import cal_and_cal
from taku_reader3 import test_articles_by_fold
articles = test_articles_by_fold(0)[:10]
text = '复制文本'
cal_and_cal(text, articles[index])
```

2. 制作csv文件

确认所有的输出符合要求之后，将输出复制到csv文件中。这个需要一行一行复制。手工操作。

3. 计算整体得分

最后使用同一个文件中的cal_from_csv函数就可以计算整体分数。


## 2024.9.30 对BERTSUM和提案手法进行t检定

1. 首先读取提案手法的结果

```py
from t_test import load_dic
dic = load_dic('exp/t_test_roberta.dic')
our_scores = dic['ROBERTA_TITLE_APPEND_CRF'] # 331 f-scores
```

2. 计算bertsum的结果并保存

```py
from t_test_bertsum import t_test, save_score_dic
from bertsum import BERTSUM
dic = t_test([BERTSUM])
save_score_dic(dic)
bertsum_scores = dic['BERTSUM']
```

3. 使用scipy进行wilcoxon检定

```py
from scipy import stats
_, p_value = stats.wilcoxon(bertsum_scores, our_scores)
# >>> p_value
# 5.542858751971058e-20
```