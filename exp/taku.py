def flatten(l):
    return [item for sublist in l for item in sublist]

def cal_prec_rec_f1_v2(results, targets):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for guess, target in zip(results, targets):
        if guess == 1:
            if target == 1:
                TP += 1
            elif target == 0:
                FP += 1
        elif guess == 0:
            if target == 1:
                FN += 1
            elif target == 0:
                TN += 1
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
    balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
    balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
    return prec, rec, f1, balanced_acc

def save_dic(dic, path = 'dd.dic'):
    import pickle
    pickle.dump(dic, open(path,"wb"))

def load_dic(path = 'dd.dic'):
    import pickle
    return pickle.load(open(path,"rb"))

def run(indicator = 2, need_save_dic = True):
    f = open('crf_only.out', 'r')
    y_true = []
    y_pred = []
    y_true_temp = []
    y_pred_temp = []
    t = 0
    p = 0
    datalist = f.read().splitlines()
    f.close()
    for data in datalist:
      list = data.split()
      if 'DOCID' in list[0]:
        print("article")
        # NOTE: NEW article and append result
        y_true.append(y_true_temp)
        y_pred.append(y_pred_temp)
        y_true_temp = []
        y_pred_temp = []
      else:
        if list[8] == 'O':
          y_true_temp.append(0)
        elif 'STRONG' in list[8]:
          t = t + 1
          y_true_temp.append(1)
        if list[9] == 'O':
          y_pred_temp.append(0)
        elif 'STRONG' in list[9]:
          p = p + 1
          y_pred_temp.append(1)
    # print(cal_prec_rec_f1_v2(y_pred, y_true))
    article_fs = [cal_prec_rec_f1_v2(x, y)[indicator] for x, y in zip(y_pred, y_true)]
    if need_save_dic:
        save_dic({'CRF': article_fs}, 't_test_crf_only.dic')
    return article_fs

############################# 在论文中引用token等级的实验结果(整体评分的时候) ################################

def run_token_level(indicator = 2, need_save_dic = False):
    f = open('crf_only.out', 'r')
    y_true_temp = []
    y_pred_temp = []
    t = 0
    p = 0
    datalist = f.read().splitlines()
    f.close()
    for data in datalist:
      list = data.split()
      if 'DOCID' in list[0]:
          ...
      else:
        if list[8] == 'O':
          y_true_temp.append(0)
        elif 'STRONG' in list[8]:
          t = t + 1
          y_true_temp.append(1)
        if list[9] == 'O':
          y_pred_temp.append(0)
        elif 'STRONG' in list[9]:
          p = p + 1
          y_pred_temp.append(1)
    y_true = y_true_temp
    y_pred = y_pred_temp
    # print(cal_prec_rec_f1_v2(y_pred, y_true))
    res = cal_prec_rec_f1_v2(y_pred, y_true)
    if need_save_dic:
        # save_dic({'CRF': article_fs}, 't_test_crf_only.dic')
        ...
    return res

def get_first_ten_scores(indicator = 2, need_save_dic = True):
    f = open('crf_only.out', 'r')
    y_true = []
    y_pred = []
    y_true_temp = []
    y_pred_temp = []
    t = 0
    p = 0
    datalist = f.read().splitlines()
    f.close()
    for data in datalist:
      list = data.split()
      if 'DOCID' in list[0]:
        print("article")
        # NOTE: NEW article and append result
        y_true.append(y_true_temp)
        y_pred.append(y_pred_temp)
        y_true_temp = []
        y_pred_temp = []
      else:
        if list[8] == 'O':
          y_true_temp.append(0)
        elif 'STRONG' in list[8]:
          t = t + 1
          y_true_temp.append(1)
        if list[9] == 'O':
          y_pred_temp.append(0)
        elif 'STRONG' in list[9]:
          p = p + 1
          y_pred_temp.append(1)
    # print(cal_prec_rec_f1_v2(y_pred, y_true))
    xx = flatten(y_pred[:10])
    yy = flatten(y_true[:10])
    return cal_prec_rec_f1_v2(xx, yy)


################### CRF的输出例 ####################3

def print_case_7(indicator = 2):
    f = open('crf_only.out', 'r')
    datalist = f.read().splitlines()
    f.close()
    start = False
    text = ''
    is_last_strong = False
    for data in datalist:
      list = data.split()
      if 'DOCID' in list[0]:
          if list[0] == '<DOCID>2012_03_120217xtype.html</DOCID>':
              print('DDDDDDDDDDDD')
              print(list[0])
              start = True
          else:
              start = False
      elif start:
        token = list[0]
        if list[9] == 'O':
            if is_last_strong:
                is_last_strong = False
                token = ']' + token 
            else:
                ...
        elif 'STRONG' in list[9]:
            if is_last_strong:
                ...
            else:
                is_last_strong = True
                token = '[' + token
        text += token
    return text
