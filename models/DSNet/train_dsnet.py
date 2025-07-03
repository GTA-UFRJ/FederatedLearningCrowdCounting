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
from data.data_settings import data_path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


def cal_lc_loss(output, target, sizes=(1,2,4)):
    criterion_L1 = nn.L1Loss(reduction='sum')
    Lc_loss = None
    for s in sizes:
        pool = nn.AdaptiveAvgPool2d(s)
        est = pool(output)
        gt = pool(target)
        if Lc_loss:
            Lc_loss += criterion_L1(est, gt) / s**2
        else:
            Lc_loss = criterion_L1(est, gt) / s**2
    return Lc_loss



def get_loader(train_list, ratio, kernel_path='ground_truth_npy'):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_loader = torch.utils.data.DataLoader(
        RawDataset(train_list, transform, aug=True, ratio=ratio, kernel_path=kernel_path),
        shuffle=True, batch_size=1, num_workers=4)
    
    return train_loader


def train_dsnet(argv = sys.argv):
    dataset_type = argv[1]
    base_path = data_path[dataset_type]
    n_split = argv[2]  

    checkpoint_path = os.path.join('checkpoints', 'DSNet',dataset_type)
    os.makedirs(checkpoint_path, exist_ok=True)

    with open(f'{base_path}/train_splits/train_{n_split}.pkl',"rb") as fp:
        train_list = pickle.load(fp)
        train_list = [f'{base_path}/images/'+item for item in  train_list if item != '.DS_Store']
    train_list = train_list[:15]

    model = DSNet('').to(DEVICE)  
    lbda = 1000
    ratio = 8 # density map scaling
    train_loader = get_loader(train_list, ratio,)


    epochs = 100
    lr=1e-6
    weight_decay=5e-4
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, (img, target) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(DEVICE)
            target = target.unsqueeze(1).to(DEVICE)
            output = model(img)
            Le_Loss = criterion(output, target)
            Lc_Loss = cal_lc_loss(output, target)
            loss = Le_Loss + lbda * Lc_Loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 10 == 0:
                print(loss)        
        print("epoch:",epoch,"loss:",train_loss/len(train_loader))    
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))  
        torch.save(model.state_dict(),'./checkpoints/DSNet/'+dataset_type+'/split_'+str(n_split)+".param")


if __name__ == '__main__':
    train_dsnet()
    
