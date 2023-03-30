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
        data_root, split, fake, grey = conf.data_dir, conf.split, conf.fake, conf.grey
        self.image_size = conf.image_size
        self.data_root = data_root
        self.split = split
        self.fake = fake
        self.grey = grey
        self.image_paths = sorted(self.glob_imgs(os.path.join(self.data_root, self.split)))
        self.image_fake_paths = sorted(self.glob_imgs(os.path.join(self.data_root, self.fake)))
        self.image_grey_paths = sorted(self.glob_imgs(os.path.join(self.data_root, self.grey))) 
        

    def __getitem__(self, index):
        img_size = (self.image_size, self.image_size)
        # import ipdb; ipdb.set_trace()
        # image_path = self.image_paths[index]
        # img_RGB = np.array(Image.open(image_path).convert('RGB').resize(img_size)).astype(np.float32) / 255.0
        # img_GREY = np.array(Image.open(image_path).convert('L').resize(img_size)).astype(np.float32)
        # # import ipdb; ipdb.set_trace()
        # image_fake_path = self.image_fake_paths[random.randint(0, len(self.image_fake_paths)-1)]
        # img_FAKE = np.array(Image.open(image_fake_path).convert('RGB').resize(img_size)).astype(np.float32) / 255.0
        # # img_RGB = cv2.resize(img_RGB, img_size, interpolation=cv2.INTER_CUBIC)
        # # img_GREY = cv2.resize(img_GREY, img_size, interpolation=cv2.INTER_CUBIC)
        # # img_FAKE = cv2.resize(img_FAKE, img_size, interpolation=cv2.INTER_CUBIC)
        image_RGB_path = self.image_paths[index]
        image_GREY_path = self.image_grey_paths[index]
        image_fake_path = self.image_fake_paths[random.randint(0, len(self.image_fake_paths)-1)]
        image_name = image_RGB_path.split('/')[-1]
        assert image_GREY_path.split('/')[-1] == image_name
        img_RGB_raw = Image.open(image_RGB_path).convert('RGB')
        original_size = img_RGB_raw.size
        img_RGB = np.array(img_RGB_raw.resize(img_size)).astype(np.float32) / 255.0
        img_GREY = np.array(Image.open(image_GREY_path).resize(img_size))[:,:,0] / 255.0
        img_FAKE = np.array(Image.open(image_fake_path).convert('RGB').resize(img_size)).astype(np.float32) / 255.0
        ret = {
            'img_RGB': img_RGB,
            'img_GREY': img_GREY,
            'img_FAKE': img_FAKE,
            'original_size': original_size,
            'img_name': image_name
        }
        
        return ret

    def __len__(self):
        return len(self.image_paths)
    
    def glob_imgs(self, path):
        from glob import glob
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs