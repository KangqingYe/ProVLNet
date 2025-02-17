import numpy as np
from numpy import random
import cv2
import monai.transforms as monai_transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_list):
        for t in self.transforms:
            data_list = t(data_list)
        return data_list
    
class ConvertImgFloat(object):
    def __call__(self, data_list):
        if len(data_list) == 2:
            return [data_list[0].astype(np.float32), data_list[1].astype(np.float32)]
        
class Normalize(object):
    def __call__(self, data_list):
        if len(data_list) == 2:
            return [data_list[0]/255., data_list[1]]
    
class Resize(object):
    def __init__(self, h, w):
        self.dsize = (w,h)

    def __call__(self, data_list):
        data_list[0] = cv2.resize(data_list[0], dsize=self.dsize)
        data_list[1] = cv2.resize(data_list[1], dsize=self.dsize)
        if len(data_list) == 3:
            h,w,c = data_list[0].shape
            data_list[2][:, 0] = data_list[2][:, 0]/w*self.dsize[0]
            data_list[2][:, 1] = data_list[2][:, 1]/h*self.dsize[1]
            data_list[2] = np.asarray(data_list[2])
        return data_list
