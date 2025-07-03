import torch
import torch.nn as nn
from torchvision import  transforms
import os
import numpy as np
import sys
import time
import pickle
from .can_model import CANNet
from .can_dataloader import listDataset
from data.data_settings import data_path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_can(argv = sys.argv):
    dataset_type = argv[1]
    base_path = data_path[dataset_type]
    n_split = argv[2]

    with open(f'{base_path}/train_splits/train_{n_split}.pkl',"rb") as fp:
        train_list = pickle.load(fp)
        train_list = [f'{base_path}/images/'+item for item in  train_list if item != '.DS_Store']
    
    torch.backends.cudnn.enabled=False

    model = CANNet()
    model = model.to(DEVICE)
    criterion = nn.MSELoss(size_average=False).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,
                                    weight_decay=5*1e-4)
    checkpoint_path = os.path.join('checkpoints', 'CAN',dataset_type)
    os.makedirs(checkpoint_path, exist_ok=True)

    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(0, 100):
        train_loader = torch.utils.data.DataLoader(
            listDataset(train_list,dataset_type=dataset_type,
                       shuffle=True,transform=transforms.Compose([transforms.ToTensor(),]),
                       train=True,num_workers=2),batch_size=1
                       )

        model.train()
        epoch_loss=0
        for i,(img, target) in enumerate(train_loader):
            img = img.to(DEVICE)
            output = model(img)[:,0,:,:]
            target = target.to(DEVICE)
            loss = criterion(output, target)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if i % 10 == 0:
                print(loss)
        print("epoch:",epoch,"loss:",epoch_loss/len(train_loader))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        torch.save(model.state_dict(),'./checkpoints/CAN/'+dataset_type+'/split_'+str(n_split)+".param")

if __name__ == '__main__':
    train_can()

