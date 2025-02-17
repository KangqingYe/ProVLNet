from pathlib import Path
class Config(object):
    dataset_type = 'thoracic'
    run_name = 'date_2dscn'
    # path
    # raw_data_folder = Path('/data2/citi/ykq/dataset/multiview/DRR/')
    raw_data_folder = Path('/media/cygzz/data/ykq/dataset/verse_231119/')
    debug_folder = Path('debug')

    gpus = [0]
    weightFile = "weights/thoracic/2d_scn.pkl"#"none"
    # parameters
    batch_size = 4
    max_epochs = 300 # 100 for lumbar, 300 for thoracic
    save_interval = 1
    # solver params
    steps = [10000]
    scales = [0.1]
    learning_rate = 1e-3
    momentum = 0.9
    decay = 5e-4
    betas = (0.9, 0.98)
    # model config
    model_type = '2dscn'
    num_classes = 4 # 5 for lumbar, 4 for thoracic
    use_confidences = False