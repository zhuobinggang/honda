# ファイルの説明

* `CRF_only.out`とは安倍さんの手法の出力です、`taku.py`の`run`関数で処理して331記事ごとのf値を手に入れる、一応その結果を`t_test_crf_only.dic`に保存した、`taku.py`の`load_dic`関数で読み取ることができる
* `t_test.dic`とはBERT, BERT+TITLE, BERT+TITLE2, ３つのBERT手法の記事ごとのf値です, `taku.py`の`load_dic`関数で読み取ることができる
* `t_test_crf.dic`とはBERT+CRF, BERT+TITLE+CRF, BERT+TITLE2+CRF, ３つのCRF手法の記事ごとのf値です, `taku.py`の`load_dic`関数で読み取ることができる
