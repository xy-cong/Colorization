import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image
import random
import cv2

class ColorDataset(data.Dataset):
    def __init__(self, conf):
        super(ColorDataset, self).__init__()
        data_root, grey, ref, split = conf.data_dir, conf.grey, conf.ref, conf.split
        self.image_size = conf.image_size
        self.data_root = data_root
        self.grey = grey
        self.ref = ref
        self.split = split
        self.image_grey_paths = sorted(self.glob_imgs(os.path.join(self.data_root, self.grey))) 
        self.image_ref_paths = sorted(self.glob_imgs(os.path.join(self.data_root, self.ref))) 
        self.image_paths = sorted(self.glob_imgs(os.path.join(self.data_root, self.split))) 
        

    def __getitem__(self, index):
        img_size = (self.image_size, self.image_size)
        image_RGB_path = self.image_paths[index]
        image_GREY_path = self.image_grey_paths[index]
        image_ref_path = self.image_ref_paths[index]
        image_name = image_RGB_path.split('/')[-1]
        assert image_GREY_path.split('/')[-1] == image_name
        assert image_ref_path.split('/')[-1] == image_name
        img_RGB_raw = Image.open(image_RGB_path).convert('RGB')
        original_size = img_RGB_raw.size
        img_RGB = np.array(img_RGB_raw.resize(img_size)).astype(np.float32) / 255.0
        img_GREY = np.array(Image.open(image_GREY_path).resize(img_size))[:,:,0] / 255.0
        img_ref = np.array(Image.open(image_ref_path).convert('RGB').resize(img_size)).astype(np.float32) / 255.0
        ret = {
            'img_RGB': img_RGB,
            'img_GREY': img_GREY,
            'img_REF': img_ref,
            'original_size': original_size,
            'img_name': image_name
        }
        
        return ret

    def __len__(self):
        return len(self.image_grey_paths)
    
    def glob_imgs(self, path):
        from glob import glob
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs