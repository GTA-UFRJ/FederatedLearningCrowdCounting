import torch
import torch.nn as nn
import os
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
from data.data_settings import data_path

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def find_bin(value, bin_edges):
    """Finds the bin index where a given value falls."""
    for i in range(len(bin_edges) - 1):
        delta_min = 0 
        delta_max = 0
        if i == 0:
            delta_min = 0.001
        if i == len(bin_edges)-2:
            delta_max = 0.001
        if round(bin_edges[i],4)-delta_min <= round(float(value),4) < round(bin_edges[i + 1],4)+delta_max:
            return i
    return None

def val(model_param_path, test_path):
    model = DSNet('').to(DEVICE)  
    model.load_state_dict(torch.load(model_param_path))

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_loader = torch.utils.data.DataLoader(
        RawDataset(test_path, transform, ratio=1, aug=False, ), 
        shuffle=False, batch_size=1, num_workers=4)
    time.sleep(1)

    model.eval()
    mae = 0.0
    mre = 0.0
    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(DEVICE)
            output = model(img)
            est_count = output.sum().item()
            mae += abs(est_count - target.sum())
            mre += abs(est_count - target.sum())/target.sum()
            if random.random() > 0.9:
                print(est_count,target.sum())
    mae /= len(test_loader)
    mre /= len(test_loader)
    print("len:"+str(len(test_loader))+" MAE:"+str(mae))
    time.sleep(1)
    return float(mae),float(mre) 


def test_dsnet(args = sys.argv):
    torch.backends.cudnn.enabled=False
    
    dataset_type = args[1]
    type_eval = args[2]
    n_split = args[3]
    base_path = data_path[type_eval]
    with open(f'{base_path}/test_splits/test_{n_split}.pkl',"rb") as fp:
        test_list = pickle.load(fp)
        test_list = [f'{base_path}/images/'+item for item in  test_list if item != '.DS_Store']

    model_param_path='./checkpoints/DSNet/'+dataset_type+'/split_'+n_split+'.param'
    return val(model_param_path,test_list)


if __name__ == '__main__':
    test_dsnet()


