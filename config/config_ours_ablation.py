from pathlib import Path
class Config(object):
    dataset_type = 'lumbar'
    run_name = 'date_ours_ablation'
    # path
    raw_data_folder = Path('/data2/citi/ykq/dataset/multiview/DRR/')
    debug_folder = Path('debug')

    gpus = [3]
    weightFile = "weights/lumbar/ours_ablation.pkl"
    backbone_weightFile = "none" #"/data2/citi/ykq/multiview/debug/SCN/model/1107_2dbackbone/model80.pkl"
    # parameters
    batch_size = 4
    max_epochs = 1500
    save_interval = 1
    # solver params
    steps = [10000]
    scales = [0.1]
    learning_rate = 1e-3
    volume_net_lr = 1e-3
    momentum = 0.9
    decay = 5e-4
    betas = (0.9, 0.98)
    # model config
    model_type = 'ours_ablation' 
    num_classes = 5
    use_confidences = True
    volume_aggregation_method = "conf"