'''
Train script for 2D ResNet and 2D SCN.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from config.config_2dscn import Config #config_2dscn
from datasets import BackboneLumbarDataset,BackboneThoracicDataset
from model import Loc_SCN, get_pose_net
from utils import op
from utils.metric import get_pe
from utils.multiview import project_3d_points_to_image_plane
import loss
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def validate(model, val_loader, device, dataset_type):
    model.eval()
    with torch.no_grad():
        pe_2d_list = []
        for data_dict in tqdm(val_loader):
            for name in data_dict:
                if name in ['name','id']:
                    continue
                data_dict[name] = data_dict[name].to(device=device)
            if model_type == 'res':
                output,_,_,_ = model(data_dict['image'].repeat(1,3,1,1))
            elif model_type == 'scn':
                output,_,_,_ = model(data_dict['image'])
            if dataset_type == 'lumbar':
                output = F.interpolate(output, size = (768,512), mode="bilinear")
                output = torch.relu(output)
                
                keypoints_2d_pred, output_softmax = op.integrate_tensor_2d(heatmaps=output*50,softmax=True)

                keypoints_2d_gt = data_dict['gt_pts'][:,:,1:]
                pe_each = get_pe(keypoints_2d_pred,keypoints_2d_gt)
            elif dataset_type == 'thoracic':
                output = F.interpolate(output, size=(512,512), mode='bilinear', align_corners=False)

                keypoints_2d_pred = keypoints_2d_pred[0,:,[1,0]]

                keypoints_2d_gt_proj = project_3d_points_to_image_plane(data_dict['P'].squeeze(),
                                                                            data_dict['gt_pts_3d'].squeeze())
                pe_each = get_pe(keypoints_2d_pred,keypoints_2d_gt_proj)
            pe_2d_list.append(pe_each)
        pe_2d_list = torch.cat(pe_2d_list, dim=0)
        return pe_2d_list.mean().item()

if __name__ == '__main__':
    set_seed(20)
    config = Config()
    model_type = config.model_type#'2dres', '2dscn'
    dataset_type = config.dataset_type
    run_name = config.run_name
    writer = SummaryWriter(config.debug_folder/'log'/run_name)
    os.system('cp config/config_2dres.py '+ str(config.debug_folder) + '/model/'+run_name+'/config.py')

    kwargs = {'num_workers': 2, 'pin_memory': True}
    device = torch.device("cuda:%s" % str(config.gpus[0]))    
    torch.cuda.set_device(config.gpus[0])

    if model_type == '2dres':
        model = get_pose_net(config,device)
    elif model_type == '2dscn':
        model = Loc_SCN(config.num_classes,1)
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
        train_dataset = BackboneLumbarDataset(raw_data_folder = config.raw_data_folder,
                                    train_phase = 'train',
                                    input_h = 768,
                                    input_w = 512)
        train_loader = DataLoader(train_dataset,batch_size = config.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_dataset = BackboneLumbarDataset(raw_data_folder = config.raw_data_folder,
                                    train_phase = 'val',
                                    input_h = 768,
                                    input_w = 512)
        val_loader = DataLoader(val_dataset,batch_size = 1, shuffle=False, drop_last=False, **kwargs)
    elif dataset_type == 'thoracic':
        train_dataset = BackboneThoracicDataset(raw_data_folder = config.raw_data_folder,
                                    train_phase = 'train')
        train_loader = DataLoader(train_dataset,batch_size = config.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_dataset = BackboneThoracicDataset(raw_data_folder = config.raw_data_folder,
                                    train_phase = 'val')
        val_loader = DataLoader(train_dataset,batch_size = 1, shuffle=False, drop_last=False, **kwargs)

    step_tensorboard = 0
    for epoch in tqdm(range(init_epoch, config.max_epochs)):
        model.train()
        running_loss = 0
        for data_dict in train_loader:
            for name in data_dict:
                if name in ['name','id','id']:
                    continue
                data_dict[name] = data_dict[name].to(device=device)
            optimizer.zero_grad()
            if model_type == '2dres':
                output,_,_,_ = model(data_dict['image'].repeat(1,3,1,1))
            elif model_type == '2dscn':
                output,_,_,_ = model(data_dict['image'])
            if dataset_type == 'lumbar':
                output = F.interpolate(output, size = (768,512), mode="bilinear")
            elif dataset_type == 'thoracic':
                output = F.interpolate(output, size = (512,512), mode='bilinear', align_corners=False)
            loss_mse = nn.MSELoss(reduction='mean')(output,data_dict['hmap_gt'])
            loss_dice = loss.DiceLoss()(torch.relu(output),data_dict['hmap_gt'])
            loss_total = (1 - loss_dice) + loss_mse
            running_loss += loss_total.item()

            loss_total.backward()
            optimizer.step()

            print(' %d, dice: %f, mse: %f, loss: %f'
              % (epoch, loss_dice.item(), loss_mse.item(), loss_total.item()))

            writer.add_scalar('dice', loss_dice, step_tensorboard)
            writer.add_scalar('mse', loss_mse, step_tensorboard)
            writer.add_scalar('loss', loss_total, step_tensorboard)
            step_tensorboard+=1
        epoch_loss = running_loss / len(train_loader)
        print('{} epoch loss: {}'.format(epoch, epoch_loss))

        if (epoch + 1) % config.save_interval == 0:
            val_loss = validate(model, val_loader, device, dataset_type)
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