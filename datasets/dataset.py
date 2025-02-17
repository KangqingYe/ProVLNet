import os
import cv2
import numpy as np
import json
from torch.utils.data import Dataset
from datasets import pre_proc
from utils import HeatmapImageGenerator
from utils.multiview import project_3d_points_to_image_plane

class BackboneLumbarDataset(Dataset):
    def __init__(self,raw_data_folder,train_phase,input_h=768,input_w=512):
        # train_phase: 'train','val','test'
        self.raw_data_folder = raw_data_folder
        self.train_phase = train_phase
        self.input_h = input_h
        self.input_w = input_w
        self.img_dir = self.raw_data_folder/train_phase
        self.img_list = [subfolder.joinpath(f"{i}.png") for subfolder in self.img_dir.iterdir() if subfolder.is_dir() for i in range(20) if (subfolder / f"{i}.png").exists()]
        self.img_list = sorted(self.img_list)

    def load_image(self, index):
        image = cv2.imread(str(self.img_list[index]))
        return image
    
    def get_name_id(self, index):
        name = self.img_list[index].parent.stem[:-4]
        id = self.img_list[index].stem
        return name, id

    def load_gt_pts(self, name, id):
        return np.loadtxt(str(self.img_dir/(name+'.nii.gz')/(str(id)+'_coord.txt')))

    def generate_heatmap(self, index):
        name, id = self.get_name_id(index)
        image = self.load_image(index)
        img_shape = image.shape[:2]
        gt_pts = self.load_gt_pts(name, id)
        heatmap_image_generator = HeatmapImageGenerator(image_size=list(img_shape),
                                                        sigma=20,
                                                        scale_factor=1)
        hmap_list = []
        for ch in range(5):
            hmap_list.append(heatmap_image_generator.generate_heatmap(gt_pts[ch,1:],sigma_scale_factor=1))
        hmaps = np.stack(hmap_list, axis=0)
        hmaps = hmaps.transpose(1,2,0)#(1024,1024,6)
        return hmaps

    def __getitem__(self, index):
        image = self.load_image(index)
        hmap_gt = self.generate_heatmap(index)
        if self.train_phase == 'train':
            data_dict = pre_proc.process_image_hmap_simple(image = image,
                                                hmap = hmap_gt,
                                                image_h = self.input_h,
                                                image_w = self.input_w,
                                                aug_flag = True)
            name, id = self.get_name_id(index)
            data_dict['gt_pts'] = self.load_gt_pts(name,id)
            return data_dict
                
        if self.train_phase == 'val' or self.train_phase == 'test':
            data_dict = pre_proc.process_image_hmap_simple(image = image,
                                                hmap = hmap_gt,
                                                image_h = self.input_h,
                                                image_w = self.input_w,
                                                aug_flag = False)
            name, id = self.get_name_id(index)
            data_dict['name'] = name
            data_dict['id'] = id
            data_dict['gt_pts'] = self.load_gt_pts(name,id)
            return data_dict

    def __len__(self):
        return len(self.img_list)

class TwoViewTestDataset(Dataset):
    def __init__(self,raw_data_folder,train_phase,input_h=768,input_w=512):
        self.raw_data_folder = raw_data_folder
        self.input_h = input_h
        self.input_w = input_w
        self.img_dir = self.raw_data_folder/train_phase
        self.name_list = [subdir.name[:-7] for subdir in self.img_dir.iterdir() if subdir.is_dir()]
        self.pairs = [(i, j) for i in range(0,10) for j in range(10,20)]
    
    def load_image(self, name, id):
        image = cv2.imread(str(self.img_dir/(name+'.nii.gz')/(str(id)+'.png')))
        return image
    
    def get_name(self, index):
        return self.name_list[index]

    def load_3d_gt_pts(self, name):
        gt_pts_3d_raw = np.loadtxt(str(self.img_dir/(name+'.nii.gz')/'center_3D_coord.txt'))
        return gt_pts_3d_raw[:,1:]

    def get_transform_matrix(self, name, id):
        with open(self.img_dir/(name+'.nii.gz')/(str(id)+'.txt'), 'r') as f:
            lines = f.readlines()
        K = np.array([list(map(float, line.split())) for line in lines[:3]])
        K = np.hstack([K, np.zeros((3, 1))])

        T_S_CT = np.zeros((4, 4))
        T_S_CT[:3, :3] = np.array([list(map(float, line.split())) for line in lines[3:6]])
        T_S_CT[:3, 3] = np.array(list(map(float, lines[6].split()))).reshape(-1, 1)[:,0]
        T_S_CT[3, 3] = 1.0

        T_C_S = np.zeros((4, 4))
        T_C_S[:3, :3] = np.array([list(map(float, line.split())) for line in lines[7:10]])
        T_C_S[:3, 3] = np.array(list(map(float, lines[10].split()))).reshape(-1, 1)[:,0]
        T_C_S[3, 3] = 1.0
        return K, T_C_S, T_S_CT

    def get_P(self, name, id):
        K, T_C_S, T_S_CT = self.get_transform_matrix(name, id)
        return K@T_C_S@T_S_CT
    
    def get_K_T(self, name, id):
        K, T_C_S, T_S_CT = self.get_transform_matrix(name, id)
        return K, T_C_S@T_S_CT

    def __getitem__(self, index):
        subject_index = index // 100
        pair_index = index % 100
        name = self.get_name(subject_index)
        AP_id, LAT_id = self.pairs[pair_index]

        gt_pts_3d = self.load_3d_gt_pts(name)

        AP_image = self.load_image(name, AP_id)
        AP_P = self.get_P(name, AP_id)
        AP_K, AP_T = self.get_K_T(name, AP_id)
        LAT_image = self.load_image(name, LAT_id)
        LAT_P = self.get_P(name, LAT_id)
        LAT_K, LAT_T = self.get_K_T(name, LAT_id)

        AP_image = pre_proc.process_image(AP_image)
        LAT_image = pre_proc.process_image(LAT_image)

        data_dict = {
            "name":name,
            "gt_pts_3d":gt_pts_3d.astype(np.float32),
            "AP_image": AP_image,
            "AP_P": AP_P.astype(np.float32),
            "AP_K": AP_K.astype(np.float32),
            "AP_T": AP_T.astype(np.float32),
            "AP_id": AP_id,
            "LAT_image": LAT_image,
            "LAT_P": LAT_P.astype(np.float32),
            "LAT_K": LAT_K.astype(np.float32),
            "LAT_T": LAT_T.astype(np.float32),
            "LAT_id": LAT_id
        }
        return data_dict
    def __len__(self):
        return len(self.name_list)*100

class E2ELumbarDataset(Dataset):
    def __init__(self,raw_data_folder,train_phase,input_h=768,input_w=512):
        self.raw_data_folder = raw_data_folder
        self.input_h = input_h
        self.input_w = input_w
        self.img_dir = self.raw_data_folder/train_phase
        self.name_list = [subdir.name[:-7] for subdir in self.img_dir.iterdir() if subdir.is_dir()]
    
    def load_image(self, name, id):
        image = cv2.imread(str(self.img_dir/(name+'.nii.gz')/(str(id)+'.png')))
        return image
    
    def get_name(self, index):
        return self.name_list[index]

    def generate_AP_LAT_id(self):
        return np.random.randint(0,10),np.random.randint(10,20)

    def generate_heatmap(self, name, id):
        image = self.load_image(name, id)
        img_shape = image.shape[:2]
        gt_pts_3d = self.load_3d_gt_pts(name)
        P = self.get_P(name, id)
        gt_pts = project_3d_points_to_image_plane(P,gt_pts_3d)
        gt_pts = gt_pts[:,::-1]
        heatmap_image_generator = HeatmapImageGenerator(image_size=list(img_shape),
                                                        sigma=20,
                                                        scale_factor=1)
        hmap_list = []
        for ch in range(5):
            hmap_list.append(heatmap_image_generator.generate_heatmap(gt_pts[ch],sigma_scale_factor=1))
        hmaps = np.stack(hmap_list, axis=0)
        return hmaps

    def load_3d_gt_pts(self, name):
        gt_pts_3d_raw = np.loadtxt(str(self.img_dir/(name+'.nii.gz')/'center_3D_coord.txt'))
        return gt_pts_3d_raw[:,1:]

    def get_transform_matrix(self, name, id):
        with open(self.img_dir/(name+'.nii.gz')/(str(id)+'.txt'), 'r') as f:
            lines = f.readlines()
        K = np.array([list(map(float, line.split())) for line in lines[:3]])
        K = np.hstack([K, np.zeros((3, 1))])

        T_S_CT = np.zeros((4, 4))
        T_S_CT[:3, :3] = np.array([list(map(float, line.split())) for line in lines[3:6]])
        T_S_CT[:3, 3] = np.array(list(map(float, lines[6].split()))).reshape(-1, 1)[:,0]
        T_S_CT[3, 3] = 1.0

        T_C_S = np.zeros((4, 4))
        T_C_S[:3, :3] = np.array([list(map(float, line.split())) for line in lines[7:10]])
        T_C_S[:3, 3] = np.array(list(map(float, lines[10].split()))).reshape(-1, 1)[:,0]
        T_C_S[3, 3] = 1.0
        return K, T_C_S, T_S_CT

    def get_K_T(self, name, id):
        K, T_C_S, T_S_CT = self.get_transform_matrix(name, id)
        return K, T_C_S@T_S_CT

    def get_P(self, name, id):
        K, T_C_S, T_S_CT = self.get_transform_matrix(name, id)
        return K@T_C_S@T_S_CT

    def __getitem__(self, index):
        name = self.get_name(index)
        AP_id, LAT_id = self.generate_AP_LAT_id()

        gt_pts_3d = self.load_3d_gt_pts(name)

        AP_image = self.load_image(name, AP_id)
        AP_hmap_gt = self.generate_heatmap(name, AP_id)
        AP_K, AP_T = self.get_K_T(name, AP_id)
        LAT_image = self.load_image(name, LAT_id)
        LAT_hmap_gt = self.generate_heatmap(name, LAT_id)
        LAT_K, LAT_T = self.get_K_T(name, LAT_id)

        AP_image = pre_proc.process_image(AP_image)
        LAT_image = pre_proc.process_image(LAT_image)

        data_dict = {
            "name":name,
            "gt_pts_3d":gt_pts_3d.astype(np.float32),
            "AP_image": AP_image,
            "AP_hmap_gt": AP_hmap_gt.astype(np.float32),
            "AP_K": AP_K.astype(np.float32),
            "AP_T": AP_T.astype(np.float32),
            "LAT_image": LAT_image,
            "LAT_hmap_gt": LAT_hmap_gt.astype(np.float32),
            "LAT_K": LAT_K.astype(np.float32),
            "LAT_T": LAT_T.astype(np.float32),
        }
        return data_dict
    def __len__(self):
        return len(self.name_list)
    
class E2EThoracicDataset(Dataset):
    def __init__(self,raw_data_folder,train_phase):
        self.raw_data_folder = raw_data_folder
        self.img_dir = self.raw_data_folder/train_phase
        self.name_list = [subdir.name for subdir in self.img_dir.iterdir() if subdir.is_dir()]
    
    def load_image(self, name, id):
        image = cv2.imread(str(self.img_dir/name/(str(id)+'.png')))
        return image
    
    def get_name(self, index):
        return self.name_list[index]

    def generate_AP_LAT_id(self):
        return 0,1

    def generate_heatmap(self, name, id):
        image = self.load_image(name, id)
        img_shape = image.shape[:2]
        gt_pts_3d = self.load_3d_gt_pts(name)
        P = self.get_P(name, id)
        gt_pts = project_3d_points_to_image_plane(P,gt_pts_3d)
        gt_pts = gt_pts[:,::-1]
        heatmap_image_generator = HeatmapImageGenerator(image_size=list(img_shape),
                                                        sigma=20,
                                                        scale_factor=1)
        hmap_list = []
        for ch in range(4):
            hmap_list.append(heatmap_image_generator.generate_heatmap(gt_pts[ch],sigma_scale_factor=1))
        hmaps = np.stack(hmap_list, axis=0)
        return hmaps

    def load_3d_gt_pts(self, name):
        gt_pts_3d_raw = np.loadtxt(str(self.img_dir/name/'center_3D_coord.txt'))
        return gt_pts_3d_raw[:,1:]

    def get_transform_matrix(self, name, id):
        with open(self.img_dir/name/(str(id)+'.txt'), 'r') as f:
            lines = f.readlines()
        K = np.array([list(map(float, line.split())) for line in lines[:3]])
        K = np.hstack([K, np.zeros((3, 1))])

        T_S_CT = np.zeros((4, 4))
        T_S_CT[:3, :3] = np.array([list(map(float, line.split())) for line in lines[3:6]])
        T_S_CT[:3, 3] = np.array(list(map(float, lines[6].split()))).reshape(-1, 1)[:,0]
        T_S_CT[3, 3] = 1.0

        T_C_S = np.zeros((4, 4))
        T_C_S[:3, :3] = np.array([list(map(float, line.split())) for line in lines[7:10]])
        T_C_S[:3, 3] = np.array(list(map(float, lines[10].split()))).reshape(-1, 1)[:,0]
        T_C_S[3, 3] = 1.0
        return K, T_C_S, T_S_CT

    def get_K_T(self, name, id):
        K, T_C_S, T_S_CT = self.get_transform_matrix(name, id)
        return K, T_C_S@T_S_CT

    def get_P(self, name, id):
        K, T_C_S, T_S_CT = self.get_transform_matrix(name, id)
        return K@T_C_S@T_S_CT

    def __getitem__(self, index):
        name = self.get_name(index)
        AP_id, LAT_id = self.generate_AP_LAT_id()

        gt_pts_3d = self.load_3d_gt_pts(name)

        AP_image = self.load_image(name, AP_id)
        AP_hmap_gt = self.generate_heatmap(name, AP_id)
        AP_K, AP_T = self.get_K_T(name, AP_id)
        LAT_image = self.load_image(name, LAT_id)
        LAT_hmap_gt = self.generate_heatmap(name, LAT_id)
        LAT_K, LAT_T = self.get_K_T(name, LAT_id)

        AP_image = pre_proc.process_image(AP_image)
        LAT_image = pre_proc.process_image(LAT_image)

        data_dict = {
            "name":name,
            "gt_pts_3d":gt_pts_3d.astype(np.float32),
            "AP_image": AP_image,
            "AP_hmap_gt": AP_hmap_gt.astype(np.float32),
            "AP_K": AP_K.astype(np.float32),
            "AP_T": AP_T.astype(np.float32),
            "LAT_image": LAT_image,
            "LAT_hmap_gt": LAT_hmap_gt.astype(np.float32),
            "LAT_K": LAT_K.astype(np.float32),
            "LAT_T": LAT_T.astype(np.float32),
        }
        return data_dict
    def __len__(self):
        return len(self.name_list)
    
class BackboneThoracicDataset(Dataset):
    def __init__(self,raw_data_folder,train_phase):
        # train_phase: 'train','val','test'
        self.raw_data_folder = raw_data_folder
        self.train_phase = train_phase
        self.img_dir = self.raw_data_folder/train_phase
        self.img_list = [subfolder.joinpath(f"{i}.png") for subfolder in self.img_dir.iterdir() if subfolder.is_dir() for i in range(2) if (subfolder / f"{i}.png").exists()]
        self.img_list = sorted(self.img_list)

    def load_image(self, index):
        image = cv2.imread(str(self.img_list[index]))
        return image
    
    def get_name_id(self, index):
        name = self.img_list[index].parent.stem
        id = self.img_list[index].stem
        return name, id

    def get_P(self, name, id):
        K, T_C_S, T_S_CT = self.get_transform_matrix(name, id)
        return K@T_C_S@T_S_CT

    def get_transform_matrix(self, name, id):
        with open(self.img_dir/name/(str(id)+'.txt'), 'r') as f:
            lines = f.readlines()
        K = np.array([list(map(float, line.split())) for line in lines[:3]])
        K = np.hstack([K, np.zeros((3, 1))])

        T_S_CT = np.zeros((4, 4))
        T_S_CT[:3, :3] = np.array([list(map(float, line.split())) for line in lines[3:6]])
        T_S_CT[:3, 3] = np.array(list(map(float, lines[6].split()))).reshape(-1, 1)[:,0]
        T_S_CT[3, 3] = 1.0

        T_C_S = np.zeros((4, 4))
        T_C_S[:3, :3] = np.array([list(map(float, line.split())) for line in lines[7:10]])
        T_C_S[:3, 3] = np.array(list(map(float, lines[10].split()))).reshape(-1, 1)[:,0]
        T_C_S[3, 3] = 1.0
        return K, T_C_S, T_S_CT

    def generate_heatmap(self, index):
        image = self.load_image(index)
        img_shape = image.shape[:2]
        name, id = self.get_name_id(index)
        gt_pts_3d = self.load_3d_gt_pts(name)
        P = self.get_P(name, id)
        gt_pts = project_3d_points_to_image_plane(P,gt_pts_3d)
        gt_pts = gt_pts[:,::-1]
        heatmap_image_generator = HeatmapImageGenerator(image_size=list(img_shape),
                                                        sigma=20,
                                                        scale_factor=1)
        hmap_list = []

        for ch in range(4):
            hmap_list.append(heatmap_image_generator.generate_heatmap(gt_pts[ch],sigma_scale_factor=1))
        hmaps = np.stack(hmap_list, axis=0)
        # for i in range(hmaps.shape[0]):
        #     cv2.imwrite('debug/hmap'+str(i)+'.png',hmaps[i]*255)
        # cv2.imwrite('debug/image.png',image)
        # hmaps = hmaps.transpose(1,2,0)#(1024,1024,6)
        return hmaps

    def load_3d_gt_pts(self, name):
        gt_pts_3d_raw = np.loadtxt(str(self.img_dir/name/'center_3D_coord.txt'))
        return gt_pts_3d_raw[:,1:]

    def __getitem__(self, index):
        image = self.load_image(index)
        name,id = self.get_name_id(index)
        hmap_gt = self.generate_heatmap(index)
        gt_pts_3d = self.load_3d_gt_pts(name)
        P = self.get_P(name, id)

        image = pre_proc.process_image(image)
        data_dict = {
            "name":name,
            "id":id,
            "image":image,
            "hmap_gt":hmap_gt.astype(np.float32),
            "gt_pts_3d":gt_pts_3d.astype(np.float32),
            "P":P.astype(np.float32)
        }

        return data_dict

    def __len__(self):
        return len(self.img_list)