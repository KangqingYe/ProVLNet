from pathlib import Path
class Config(object):
    dataset_type = 'thoracic'
    run_name = 'date_vol'
    # path
    # raw_data_folder = Path('/data2/citi/ykq/dataset/multiview/DRR/')
    raw_data_folder = Path('/media/cygzz/data/ykq/dataset/verse_231119/')
    debug_folder = Path('debug')

    gpus = [0]
    weightFile = "weights/thoracic/vol.pkl" #"none"
    backbone_weightFile = "none"
    # parameters
    batch_size = 4
    max_epochs = 1500 # 1500 for lumbar, 500 for thoracic
    save_interval = 1
    # solver params
    steps = [10000]
    scales = [0.1]
    learning_rate = 1e-5
    volume_net_lr = 1e-3
    momentum = 0.9
    decay = 5e-4
    betas = (0.9, 0.98)
    # model config
    model_type = 'vol'
    res_alg_confidences = False
    res_vol_confidences = True
    res_num_layers = 152
    res_init_weights = True
    res_checkpoint = "weights/thoracic/2d_res.pkl"
    
    num_classes = 4 # 5 for lumbar, 4 for thoracic
    use_confidences = True
    volume_aggregation_method = "conf"