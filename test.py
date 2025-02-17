"""
Test script for 2D ResNet, 2D SCN, Alg, Vol, AdaFuse, Ours, Ours_ablation(2D SCN+Fusion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from config.config_adafuse import Config #config_2dres,config_2dscn,onfig_alg,config_vol,config_adafuse,config_ours,config_ours_ablation
from datasets import TwoViewTestDataset,E2EThoracicDataset
from model import AlgebraicTriangulationNet,AlgResNet,AlgAdafuseNet,VolumetricTriangulationNet,Vol3dResNet,VolResNet
from utils.metric import get_pe
from tqdm import tqdm

if __name__ == '__main__':
    config = Config()
    model_type = config.model_type#'2dres','2dscn,'alg','vol','adafuse','ours','ours_ablation'
    dataset_type = config.dataset_type #'lumbar','thoracic' 
    model_dir_path = config.debug_folder/'model'/config.run_name
    # when evaluating the best model, choose model [sorted_files[-1]]
    # model_files = model_dir_path.glob("best_model*.pkl")
    # sorted_files = sorted(model_files, key=lambda x: int(x.name[10:14]))

    fw = open(model_dir_path/'result_test.txt', 'w')

    name_idx = 0
    for model_num in tqdm(range(1,2,1)):
        torch.cuda.set_device(config.gpus[0])
        device = torch.device("cuda:%s" % str(config.gpus[0]))     

        if model_type == '2dres':
            model_path = Path("weights/"+dataset_type+"/2d_res.pkl")
            model = AlgResNet(config)
        if model_type == '2dscn':
            model_path = Path("weights/"+dataset_type+"/2d_scn.pkl")
            model = AlgebraicTriangulationNet(config)
        if model_type == 'alg':
            model_path = Path("weights/"+dataset_type+"/alg.pkl")
            model = AlgResNet(config)
        if model_type == 'vol':
            model_path = Path("weights/"+dataset_type+"/vol.pkl")
            model = VolResNet(config,device)
        if model_type == 'adafuse':
            model_path = Path("weights/"+dataset_type+"/adafuse.pkl")
            model = AlgAdafuseNet(config,model_path,device)
        if model_type == 'ours':
            model_path = Path("weights/"+dataset_type+"/ours.pkl")
            model = VolumetricTriangulationNet(config)
        if model_type == 'ours_ablation':
            model_path = Path("weights/"+dataset_type+"/ours_ablation.pkl")
            model = Vol3dResNet(config)

        fw_detail =  open(model_path.parent/'result'/'result_detail.txt', 'w')
        fw_name = open(model_path.parent/'result'/'name.txt', 'w')
        fw_coord_pred = open(model_path.parent/'result'/'coord_pred.txt', 'w')
        fw_coord_gt = open(model_path.parent/'result'/'coord_gt.txt', 'w')

        model = nn.DataParallel(model, device_ids=config.gpus)
        model = model.to(device=device)

        if model_type in ['alg', 'vol', 'adafuse', 'ours', 'ours_ablation']:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.module.load_state_dict(checkpoint['state_dict'],strict=False)

        model.eval()

        if dataset_type == 'lumbar':
            dsets = TwoViewTestDataset(raw_data_folder=config.raw_data_folder,
                            train_phase='test',
                            input_h=768,
                            input_w=512)
        if dataset_type == 'thoracic':
            dsets = E2EThoracicDataset(raw_data_folder=config.raw_data_folder,
                            train_phase='test')
        
        data_loader = DataLoader(dsets, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        pe_3d_list = []

        save_flag = False
        test_3d_distance_thres = 5
        with torch.no_grad():
            for data_dict in tqdm(data_loader):
                for name in data_dict:
                    if name in ['name','AP_id','LAT_id']:
                        continue
                    data_dict[name] = data_dict[name].to(device=device)

                if model_type in ['vol','ours','ours_ablation']:
                    keypoints_3d_pred,_,_,_,_ = model(data_dict['AP_image'],
                                            data_dict['AP_K'],
                                            data_dict['AP_T'],
                                            data_dict['LAT_image'],
                                            data_dict['LAT_K'],
                                            data_dict['LAT_T'])
                else:
                    keypoints_3d_pred = model(data_dict['AP_image'],
                                            data_dict['AP_K'],
                                            data_dict['AP_T'],
                                            data_dict['LAT_image'],
                                            data_dict['LAT_K'],
                                            data_dict['LAT_T'])

                pe_3d_each = get_pe(keypoints_3d_pred,data_dict['gt_pts_3d'])#[1,5,3],[1,5,3]
                pe_3d_list.append(pe_3d_each)

                if save_flag:
                    within_distance_3d_mask_each = pe_3d_each<test_3d_distance_thres
                    correct_3d_pe = pe_3d_each[within_distance_3d_mask_each]
                    correct_3d_landmark_each_num = correct_3d_pe.numel()
                    total_3d_landmark_each_num = pe_3d_each.numel()

                    print( "%04d, name: %s, AP_id: %d, LAT_id: %d" %
                        (name_idx, data_dict['name'][0], data_dict['AP_id'][0].item(), data_dict['LAT_id'][0].item()),file=fw_name,flush = True)
                    
                    print("%04d, 3d: %d/%d, %.4f±%.4f" %
                        (name_idx, correct_3d_landmark_each_num, total_3d_landmark_each_num, 
                        pe_3d_each.mean().item(), pe_3d_each.std().item()),file=fw_detail,flush = True)
                    
                    keypoints_3d_pred_np = keypoints_3d_pred.detach().cpu().numpy()
                    print("%04d, %s"%
                        (name_idx, ",".join(map(str, keypoints_3d_pred_np.flatten()))),file=fw_coord_pred,flush = True)

                    keypoints_3d_gt_np = data_dict['gt_pts_3d'].detach().cpu().numpy()
                    print("%04d, %s"%
                        (name_idx, ",".join(map(str, keypoints_3d_gt_np.flatten()))),file=fw_coord_gt,flush = True)

                    name_idx += 1
                    
        pe_3d_list = torch.cat(pe_3d_list, dim=0)
        
        within_distance_3d_mask_list = pe_3d_list<test_3d_distance_thres
        correct_3d_pe_list = pe_3d_list[within_distance_3d_mask_list]
        correct_3d_landmark_num = correct_3d_pe_list.numel()
        total_3d_landmark_num = pe_3d_list.numel()

        print("model: %3d, 3d: %d/%d, %.4f±%.4f" % 
                (int(model_num), correct_3d_landmark_num, total_3d_landmark_num, pe_3d_list.mean().item(), pe_3d_list.std().item()))#,file=fw,flush = True
