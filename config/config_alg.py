from pathlib import Path
class Config(object):
    dataset_type = 'thoracic'
    run_name = 'date_alg'
    # path
    # raw_data_folder = Path('/data2/citi/ykq/dataset/multiview/DRR/')
    raw_data_folder = Path('/media/cygzz/data/ykq/dataset/verse_231119/')
    debug_folder = Path('debug')

    gpus = [0]
    weightFile = "weights/thoracic/alg.pkl" #"none"
    # parameters
    batch_size = 4
    max_epochs = 100
    save_interval = 1
    # solver params
    steps = [10000]
    scales = [0.1]
    learning_rate = 1e-5
    momentum = 0.9
    decay = 5e-4
    betas = (0.9, 0.98)
    # model config
    model_type = 'alg'
    res_alg_confidences = True
    res_vol_confidences = False
    res_num_layers = 152
    res_init_weights = True
    res_checkpoint = "weights/thoracic/2d_res.pkl"
    # res_checkpoint = "weights/pose_resnet_4.5_pixels_human36m.pth"
    num_classes = 4 # 5 for lumbar, 4 for thoracic
    use_confidences = True