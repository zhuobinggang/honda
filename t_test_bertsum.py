def get_test_set(fold_index):
    from taku_reader3 import test_articles_by_fold
    test_set = test_articles_by_fold(fold_index)
    return test_set

def get_checkpoint(class_func, fold_index, repeat_index):
    from trainer import ModelWrapper
    model_wrapper = ModelWrapper(class_func())
    model_wrapper.set_meta(fold_index=fold_index, repeat_index=repeat_index)
    model_wrapper.load_checkpoint()
    return model_wrapper

def calculate_f1_scores_by_article(test_set, model_wrapper):
    from trainer import calc_f1_scores_by_article
    model_wrapper.model.eval()
    model_wrapper.model.cuda()
    f1_scores = calc_f1_scores_by_article(model_wrapper, test_set)
    return f1_scores

def save_score_dict(score_dict):
    import json
    from datetime import datetime
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在文件名中添加当前时间
    with open(f'exp/t_test_bertsum_{current_time}.json', 'w') as f:
        json.dump(score_dict, f)

def save_output_dict(result_dict):
    import json
    from datetime import datetime
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在文件名中添加当前时间
    with open(f'model_outputs/bertsum_lifehacker_{current_time}.json', 'w') as f:
        json.dump(result_dict, f)

def load_score_dict(file_path):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)


def output_token_level(class_func):
    scores = [[], [], [], [], []]  # 用于存储每个fold的分数
    article_count_by_fold = [0, 0, 0, 0, 0]
    for fold_index in range(5):
        test_set = get_test_set(fold_index)
        article_count_by_fold[fold_index] += len(test_set)
        for repeat_index in range(3):
            checkpoint = get_checkpoint(class_func, fold_index, repeat_index)
            # TODO
            f1_scores = calculate_f1_scores_by_article(test_set, checkpoint) # size为len(articles)
            scores[fold_index].append(f1_scores)
    scores_mean_by_repeat_flatten = []
    for fold_index, scores_by_fold in enumerate(scores): # size: (3, len(articles))
        scores_mean_by_repeat = np.array(scores_by_fold)
        assert scores_mean_by_repeat.shape == (3, article_count_by_fold[fold_index])
        scores_mean_by_repeat_flatten += scores_mean_by_repeat.mean(axis=0).tolist()
    scores_mean_by_repeat_flatten = np.array(scores_mean_by_repeat_flatten)  # size为total_articles
    assert scores_mean_by_repeat_flatten.shape == (sum(article_count_by_fold),)
    score_dict[class_func.__name__] = scores_mean_by_repeat_flatten.tolist() # size为total_articles

def t_test(classes):
    import numpy as np
    # from main import Sector_2024
    # from ablation import Sector_without_crf, Sector_without_title, Sector_without_roberta
    # classes = [Sector_2024, Sector_without_crf, Sector_without_title, Sector_without_roberta]
    score_dict = {}
    for class_func in classes:
        scores = [[], [], [], [], []]  # 用于存储每个fold的分数
        article_count_by_fold = [0, 0, 0, 0, 0]
        for fold_index in range(5):
            test_set = get_test_set(fold_index)
            article_count_by_fold[fold_index] += len(test_set)
            for repeat_index in range(3):
                checkpoint = get_checkpoint(class_func, fold_index, repeat_index)
                f1_scores = calculate_f1_scores_by_article(test_set, checkpoint) # size为len(articles)
                scores[fold_index].append(f1_scores)
        scores_mean_by_repeat_flatten = []
        for fold_index, scores_by_fold in enumerate(scores): # size: (3, len(articles))
            scores_mean_by_repeat = np.array(scores_by_fold)
            assert scores_mean_by_repeat.shape == (3, article_count_by_fold[fold_index])
            scores_mean_by_repeat_flatten += scores_mean_by_repeat.mean(axis=0).tolist()
        scores_mean_by_repeat_flatten = np.array(scores_mean_by_repeat_flatten)  # size为total_articles
        assert scores_mean_by_repeat_flatten.shape == (sum(article_count_by_fold),)
        score_dict[class_func.__name__] = scores_mean_by_repeat_flatten.tolist() # size为total_articles
    save_score_dict(score_dict)
    return score_dict

def init_result_dic():
    dic = {}
    for ds_idx in range(5): # 初始化dic，dic['ds0']['art0']['rp0'] = [0,1,0...] 通过这样的格式来存储结果
        dic[f'ds{ds_idx}'] = {}
        test_set = get_test_set(ds_idx)
        for art_idx in range(len(test_set)):
            dic[f'ds{ds_idx}'][f'art{art_idx}'] = {}
            for rp_idx in range(3):
                dic[f'ds{ds_idx}'][f'art{art_idx}'][f'rp{rp_idx}'] = []
    return dic

def cal_and_save_outputs():
    from bertsum import BERTSUM
    class_func = BERTSUM
    result_dic = init_result_dic()
    for fold_index in range(5):
        test_set = get_test_set(fold_index)
        for repeat_index in range(3):
            checkpoint = get_checkpoint(class_func, fold_index, repeat_index)
            checkpoint.m.eval()
            for art_idx, article in enumerate(test_set):
                sentence_level_results = checkpoint.predict(article).tolist()
                result_dic[f'ds{fold_index}'][f'art{art_idx}'][f'rp{repeat_index}'] = sentence_level_results
    save_output_dict(result_dic)
    return result_dic

def result_dic_to_result_array(result_dic): # (5, article_count, repeat_count, sentence_count)
    results_by_ds = [[],[],[],[],[]] # 5
    for ds_idx in range(5): # 初始化dic，dic['ds0']['art0']['rp0'] = [0,1,0...] 通过这样的格式来存储结果
        test_set = get_test_set(ds_idx)
        for art_idx in range(len(test_set)):
            results_by_ds[ds_idx].append([])
            for rp_idx in range(3):
                art_level_result = result_dic[f'ds{ds_idx}'][f'art{art_idx}'][f'rp{rp_idx}']
                results_by_ds[ds_idx][art_idx].append(art_level_result)                
    return results_by_ds

def calculate_score_from_result_dic(result_dic, need_detail_score = False, need_raw_result = False):
    result_array = result_dic_to_result_array(result_dic)  # (5, article_count, repeat_count, sentence_count)
    from common import flatten, cal_prec_rec_f1_v2
    import numpy as np
    temp = flatten(result_array) # 331, 3, sentence_count
    # flip
    result_by_repeat = [[],[],[]] # 3, 331, sentence_count
    for rp_idx in range(3):
        for art_idx in range(331):
            a = temp[art_idx][rp_idx]
            result_by_repeat[rp_idx].append(a)
    # 将句子等级的结果扩展成token等级
    articles = flatten([get_test_set(i) for i in range(5)]) # 331
    labels = []
    for art_idx, article in enumerate(articles):
        for sent_idx, sent in enumerate(article):
            labels += sent[1]
    results = [[],[],[]]
    for rp_idx in range(3):
        for art_idx, article in enumerate(articles):
            for sent_idx, sent in enumerate(article):
                label_length = len(sent[1])
                results[rp_idx] += [result_by_repeat[rp_idx][art_idx][sent_idx]] * label_length
    precs =[]
    recalls = []
    fs = []
    for repeat_results in results:
        prec, recall , f, _ = cal_prec_rec_f1_v2(repeat_results, labels)
        precs.append(prec)
        recalls.append(recall)
        fs.append(f)
    result_to_return = {'prec': precs, 'rec': recalls, 'f': fs} if need_detail_score else fs
    if not need_raw_result:
        return result_to_return
    else:
        return result_to_return, results, labels
    

def get_xy():
    result_dic = load_score_dict('/home/taku/research/honda/model_outputs/bertsum_lifehacker_321.json')
    fs, results, labels = calculate_score_from_result_dic(result_dic, need_raw_result = True)
    from common import flatten
    return flatten(results), labels * 3


def print_p_value(score_dict):
    import numpy as np
    from scipy import stats
    class_names = list(score_dict.keys())
    
    # 打印表头
    print("|         | " + " | ".join([f"{class_name} ({np.mean(score_dict[class_name]):.3f})" for class_name in class_names]) + " |")
    print("|---------|" + "|".join(["---------"] * len(class_names)) + "|")
    
    for class_func_1 in class_names:
        row = [class_func_1 + f" ({np.mean(score_dict[class_func_1]):.3f})"]
        for class_func_2 in class_names:
            if class_func_1 != class_func_2:
                scores_1 = np.array(score_dict[class_func_1])
                scores_2 = np.array(score_dict[class_func_2])
                # use wilcoxon test
                _, p_value = stats.wilcoxon(scores_1, scores_2)
                row.append(f"{p_value}")
            else:
                row.append("-")  # 对角线填充 "-"
        print("| " + " | ".join(row) + " |")

def t_test_ablation():
    from main import Sector_2024
    from ablation import Sector_without_crf, Sector_without_title, Sector_without_roberta
    classes = [Sector_2024, Sector_without_crf, Sector_without_title, Sector_without_roberta]
    t_test(classes)


def t_test_add_modules():
    from main import Sector_2024
    from add_module import Sector_bert_vanilla, Sector_bert_crf_on, Sector_bert_title_on, Sector_roberta_vanilla
    classes = [Sector_bert_vanilla, Sector_bert_crf_on, Sector_bert_title_on, Sector_roberta_vanilla, Sector_2024]
    t_test(classes)

# 2024.9.26 output -> array([[0.25679503, 0.46941941, 0.31864531]])
def average_score():
    import numpy as np
    from bertsum import BERTSUM
    classes = [BERTSUM]
    scores = np.zeros((1, 5, 3, 3))
    for class_index, class_func in enumerate(classes):
        for fold_index in range(5):
            test_set = get_test_set(fold_index)
            for repeat_index in range(3):
                checkpoint = get_checkpoint(class_func, fold_index, repeat_index)
                checkpoint.model.eval()
                checkpoint.model.cuda()
                score = checkpoint.test(test_set)
                scores[class_index, fold_index, repeat_index, 0] = score[0]
                scores[class_index, fold_index, repeat_index, 1] = score[1]
                scores[class_index, fold_index, repeat_index, 2] = score[2]
    scores_mean_by_repeat = np.mean(scores, axis=2)
    scores_mean_by_dataset = np.mean(scores_mean_by_repeat, axis=1)
    print(scores_mean_by_dataset)
    return scores_mean_by_dataset, scores

if __name__ == "__main__":
    t_test_add_modules()