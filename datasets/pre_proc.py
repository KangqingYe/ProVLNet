import cv2
import numpy as np
import random
from datasets import transform

def process_image_hmap_simple(image, hmap, image_h = 640, image_w = 640, aug_flag = False):
    if aug_flag:
        data_pro = transform.Compose([transform.ConvertImgFloat(),
                                      transform.Normalize(),
                                      transform.Resize(h = image_h, w = image_w)])
    else:
        data_pro = transform.Compose([transform.ConvertImgFloat(),
                                      transform.Normalize(),
                                      transform.Resize(h = image_h, w = image_w)])
    
    [out_image, out_hmap] = data_pro([image.copy(), hmap.copy()])
    out_image = np.clip(out_image, a_min=0., a_max=1.)
    data_dict = {
        "image":np.asarray(np.expand_dims(out_image[:,:,0],axis=0), np.float32),
        "hmap_gt":np.asarray(np.transpose(out_hmap, (2,0,1)), np.float32)
    }
    return data_dict

def process_image(image):
    image = image[:,:,0].astype(np.float32)
    image = image/255
    return np.expand_dims(image,axis=0)

def process_image_aug(image):
    image = image[:,:,0].astype(np.float32)
    image = image/255
    if np.random.random()<0.3:
        noise = np.random.poisson(image * 255) / 255

        # Add the noise to the original image
        image_noisy = image + noise

        # Clip the values to be between 0 and 1
        image = np.clip(image_noisy, 0, 1).astype(np.float32)

    # Expand the dimensions to add a channel dimension
    return np.expand_dims(image, axis=0)# add poisson distributed noise