import torch
import torchvision
import torch.nn as nn
import os
import glob
from .dsnet_model import DenseScaleNet as DSNet
import torchvision.transforms as transforms
from .dsnet_dataloader import RawDataset
import torch.nn.functional as F
import warnings
import pickle 
import sys
import time
import random
import numpy as np

warnings.filterwarnings("ignore")

paths_model = {
    'ucf_50':'./data/UCF/data/',
    'shang':'./data/ShanghaiTech/data/',
    'ucsd':'./data/UCSD/data/',
    'mall': './data/mall/data/',
    'drone':'./data/VisDrone2020-CC/data/',
    'rio':'./data/rio/data/'
}


device=torch.device("cuda")
model = DSNet('').to(device)  
model_param_path='./checkpoints/DSNet/mall/split_0.param'
model.load_state_dict(torch.load(model_param_path))

images = glob.glob(paths_model['mall']+'ground_truth_npy/*')

with torch.no_grad():
    img = Image.open(images[0]).convert('RGB').to(device)
    output = model.features(img)
    print(output)


