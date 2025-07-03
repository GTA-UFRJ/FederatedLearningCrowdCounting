import os
import torch
import torch.nn as nn
import sys
import time
from .mcnn_model import MCNN
from .mcnn_dataloader import CrowdDataset
import numpy as np
from glob import glob
import pickle
from data.data_settings import data_path


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    return tuple(zip(*batch))


def train_mcnn(argv = sys.argv):
    dataset_type = argv[1]
    base_path = data_path[argv[1]]
    n_split = argv[2]
    with open(f'{base_path}/train_splits/train_{n_split}.pkl',"rb") as fp:
        list_images = pickle.load(fp)

    torch.backends.cudnn.enabled=False
    


    mcnn=MCNN().to(DEVICE)
    criterion=nn.MSELoss(size_average=False).to(DEVICE)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-8,
                                momentum=0.95
    )


    img_root = os.path.join(base_path,'images')

    gt_dmap_root = os.path.join(base_path, 'ground_truth_npy')
    dataset=CrowdDataset(img_root,list_images,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,)


    #training phase
    checkpoint_path = os.path.join('checkpoints','MCNN', dataset_type)
    os.makedirs(checkpoint_path, exist_ok=True)

    train_loss_list=[]
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(0,1):
        mcnn.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(DEVICE)
            gt_dmap=gt_dmap.to(DEVICE)

            # forward propagation
            et_dmap=mcnn(img)

            # calculate loss
            loss=criterion(et_dmap,gt_dmap)
            if i % 10 == 0:
                print(loss)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        train_loss_list.append(epoch_loss/len(dataloader))
        torch.save(mcnn.state_dict(),'./checkpoints/MCNN/'+dataset_type+'/split_'+str(n_split)+".param")
 
if __name__=="__main__":
    train_mcnn()
