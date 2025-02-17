"""
Train script for Alg.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from config.config_alg import Config
from datasets import E2ELumbarDataset,E2EThoracicDataset
from model import AlgResNet
import loss
from utils.metric import get_pe
import numpy as np
import cv2
def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    np.random.seed(seed)

def validate(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        pe_3d_list = []
        for data_dict in val_loader:
            for name in data_dict:
                if name in ['name','AP_id','LAT_id']:
                    continue
                data_dict[name] = data_dict[name].to(device=device)
            keypoints_3d_pred = model(data_dict['AP_image'],
                                        data_dict['AP_K'],
                                        data_dict['AP_T'],
                                        data_dict['LAT_image'],
                                        data_dict['LAT_K'],
                                        data_dict['LAT_T'])
            pe_3d_each = get_pe(keypoints_3d_pred,data_dict['gt_pts_3d'])
            pe_3d_list.append(pe_3d_each)
        pe_3d_list = torch.cat(pe_3d_list, dim=0)
        return pe_3d_list.mean().item()

if __name__ == '__main__':
    set_seed(20)
    config = Config()
    run_name = config.run_name
    dataset_type = config.dataset_type
    os.system('cp config/config_alg.py '+ str(config.debug_folder) + '/model/'+run_name+'/config.py')
    writer = SummaryWriter(config.debug_folder/'log'/run_name)

    kwargs = {'num_workers': 2, 'pin_memory': True}
    device = torch.device("cuda:%s" % str(config.gpus[0]))    
    torch.cuda.set_device(config.gpus[0])

    model = AlgResNet(config)
    model = nn.DataParallel(model, device_ids=config.gpus)
    model = model.cuda(device=config.gpus[0])

    optimizer = optim.Adam(model.parameters(), config.learning_rate, betas=config.betas, weight_decay=config.decay, amsgrad=True)

    best_val_metric = float('inf')
    init_epoch = 0
    if config.weightFile != 'none':
        checkpoint = torch.load(config.weightFile, map_location='cpu')
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        init_epoch = checkpoint['epoch'] + 1

    if dataset_type == 'lumbar':
        train_dataset = E2ELumbarDataset(raw_data_folder = config.raw_data_folder,
                                    train_phase='train',
                                    input_h = 768,
                                    input_w = 512)
        train_loader = DataLoader(train_dataset,batch_size = config.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_dataset = E2ELumbarDataset(raw_data_folder = config.raw_data_folder,
                                    train_phase='val',
                                    input_h = 768,
                                    input_w = 512)
        val_loader = DataLoader(val_dataset,batch_size = config.batch_size, shuffle=False, drop_last=False, **kwargs)
    elif dataset_type == 'thoracic':
        train_dataset = E2EThoracicDataset(raw_data_folder = config.raw_data_folder,
                                train_phase="train")
        train_loader = DataLoader(train_dataset,batch_size = config.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_dataset = E2EThoracicDataset(raw_data_folder = config.raw_data_folder,
                                    train_phase="val")
        val_loader = DataLoader(val_dataset,batch_size = config.batch_size, shuffle=False, drop_last=False, **kwargs)

    step_tensorboard = 0
    for epoch in tqdm(range(init_epoch, config.max_epochs)):
        model.train()
        running_loss = 0
        for data_dict in train_loader:
            for name in data_dict:
                if name in ['name']:
                    continue
                data_dict[name] = data_dict[name].to(device=device)
            optimizer.zero_grad()
            keypoints_3d_pred = model(data_dict['AP_image'],
                                        data_dict['AP_K'],
                                        data_dict['AP_T'],
                                        data_dict['LAT_image'],
                                        data_dict['LAT_K'],
                                        data_dict['LAT_T'])
            
            loss_total = loss.KeypointsMSESmoothLoss()(keypoints_3d_pred,data_dict['gt_pts_3d'])

            running_loss += loss_total.item()

            loss_total.backward()
            optimizer.step()
            print(' %d, coord3d_mse: %f' % (epoch, loss_total.item()))
            writer.add_scalar('loss', loss_total, step_tensorboard)
            step_tensorboard+=1
        epoch_loss = running_loss / len(train_loader)
        print('{} epoch loss: {}'.format(epoch, epoch_loss))
        if (epoch + 1) % config.save_interval == 0:
            val_loss = validate(model, val_loader, device)
            writer.add_scalar('val_loss', val_loss, step_tensorboard)
            print('{} epoch validation Loss: {:.4f}'.format(epoch, val_loss))
            if val_loss < best_val_metric:
                best_val_metric = val_loss
                torch.save({'epoch': epoch,
                            'optimizer': optimizer.state_dict(),
                            'state_dict': model.module.state_dict()},
                            '%s/best_model%04d_%.3f.pkl' % (config.debug_folder/'model'/run_name,epoch+1,val_loss))
            torch.save({'epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.module.state_dict()},
                        '%s/model_last.pkl' % (config.debug_folder/'model'/run_name))