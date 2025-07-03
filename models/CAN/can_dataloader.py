import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
import random
import os
from PIL import Image
import numpy as np
import cv2

resize = {
    'shang':(1024,768),
    'ucf_50':(1024,768)
}

def load_data(img_path,dataset_type, train = True):
    img = Image.open(img_path).convert('RGB')
    #if dataset_type in resize.keys():
      #  img = img.resize(resize[dataset_type])
    # gt_file = h5py.File(gt_path,'r')
    # target = np.asarray(gt_file['density'])
    try:
        target = np.load(img_path.replace('.jpg','.npy').replace('images','ground_truth_npy'))
    except:
        target = np.load(img_path.replace('.png','.npy').replace('images','ground_truth_npy'))

    # target = np.load('/Users/lucascostafavaro/PycharmProjects/CrowdCounting/ShanghaiTech/part_A/train_data/ground_truth_npy/IMG_3.npy')
    #if dataset_type in resize.keys():
     #   target = cv2.resize(target, resize[dataset_type], interpolation=cv2.INTER_CUBIC)
    
    if train:
        ratio = 0.5
        crop_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))
        rdn_value = random.random()
        if rdn_value<0.25:
            dx = 0
            dy = 0
        elif rdn_value<0.5:
            dx = int(img.size[0]*ratio)
            dy = 0
        elif rdn_value<0.75:
            dx = 0
            dy = int(img.size[1]*ratio)
        else:
            dx = int(img.size[0]*ratio)
            dy = int(img.size[1]*ratio)

        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        _, target = cv2.threshold(cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64,0,0,cv2.THRESH_TOZERO)
    return img,target


class listDataset(Dataset):
    def __init__(self, root,dataset_type, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_type = dataset_type
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.dataset_type,self.train)
        
        if self.transform is not None:
            img = self.transform(img)
        return img,target


