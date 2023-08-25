
# NEED: model.aux_test(testset)
def cal_aux_score(instance_func, checkpoint_name):
    import torch
    from taku_reader3 import ds_5div_reconstructed_with_title
    from t_test import get_checkpoint_paths
    _, test_datasets = ds_5div_reconstructed_with_title()
    checkpoints = get_checkpoint_paths(checkpoint_name)
    fs = []
    for dataset_idx, (paths_dataset, testset) in enumerate(zip(checkpoints, test_datasets)):
        temp_fs = [] # 3
        for path_repeat in paths_dataset:
            model = instance_func()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            prec, rec, f, _ = model.aux_test(testset)
            print(f'{dataset_idx} : {f}')
            temp_fs.append(f)
        fs.append(temp_fs)
    return fs


# Text emphasis score by emphasize whole sentence
# NEED: model.test_whole_sentence_emphasis(dataset)
def cal_emphasis_score_by_aux_task(instance_func, checkpoint_name):
    import torch
    from taku_reader3 import ds_5div_reconstructed_with_title
    from t_test import get_checkpoint_paths
    _, test_datasets = ds_5div_reconstructed_with_title()
    checkpoints = get_checkpoint_paths(checkpoint_name)
    fs = []
    for dataset_idx, (paths_dataset, testset) in enumerate(zip(checkpoints, test_datasets)):
        temp_fs = [] # 3
        for path_repeat in paths_dataset:
            model = instance_func()
            checkpoint = torch.load(path_repeat)
            model.load_state_dict(checkpoint['model_state_dict'])
            prec, rec, f, _ = model.test_whole_sentence_emphasis(testset) # NOTE: The only difference
            print(f'{dataset_idx} : {f}')
            temp_fs.append(f)
        fs.append(temp_fs)
    return fs

