import torch
import torchvision
import torch.nn as nn
import os
import glob
from dsnet_model import DenseScaleNet as DSNet
import torchvision.transforms as transforms
from dsnet_dataloader import RawDataset
import torch.nn.functional as F
import warnings
import pickle
import sys
import time
import numpy as np
import random

from dsnet_dataloader import RawDataset

torch.set_num_threads(4)

warnings.filterwarnings("ignore")

paths_model = {
    'ucf_50':'../data/UCF/data/',
    'shang':'../data/ShanghaiTech/data/',
    'ucsd':'../data/UCSD/data/',
    'mall':'../data/mall/data/',
    'drone':'../data/VisDrone2020-CC/data/',
    'rio': '../data/rio/data/'
}


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


def train(model, train_loader, n_split, name):

    device=torch.device("cuda")
    model.to(device)
    lbda = 1000

    epochs = 1
    lr=1e-6
    weight_decay=5e-4
    if not os.path.exists('./checkpoints/'+name):
        os.mkdir('./checkpoints/'+name)
    torch.save(model.state_dict(),'./checkpoints/'+name+'/split_'+str(n_split)+".param")
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, (img, target) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.unsqueeze(1).to(device)
            output = model(img)
            Le_Loss = criterion(output, target)
            Lc_Loss = cal_lc_loss(output, target)
            loss = Le_Loss + lbda * Lc_Loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 200 == 0:
                print(loss)        
        print("epoch:",epoch,"loss:",train_loss/len(train_loader))    
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))  
        #torch.save(model.state_dict(),'./checkpoints/'+name+'/split_'+str(n_split)+".param")
    print()
    #print(torch.cuda.memory_allocated()/1024**2)
    #print(torch.cuda.memory_cached()/1024**2)
    with torch.no_grad():
        del img, target, model, output, loss, train_loss, optimizer, Le_Loss, Lc_Loss
        torch.cuda.empty_cache()
    #print(torch.cuda.memory_allocated()/1024**2)
    #print(torch.cuda.memory_cached()/1024**2)

def val(model, test_loader,model_param_path,model_name='mall'):
    device = 'cuda'
    model.to(device)
    model.load_state_dict(torch.load(model_param_path))

    model.eval()
    mae = 0.0
    mre = 0.0
    gen_mae = 0.0
    len_gen = 0
    #model_name = (model_param_path.split('/')[3])
    out_path = paths_model[model_name]+'outliers/'+model_name+'_outlier.npy'
    out_bins = np.load(out_path,allow_pickle=True)

    with torch.no_grad():
        for img, target in test_loader:
            img = img.to(device)
            output = model(img)
            est_count = output.sum().item()
            mae += abs(est_count - target.sum())
            mre += abs((est_count - target.sum())/target.sum())
            if random.random() > 0.9:
                print(est_count,' ' ,target.sum())
            if any(a <= target.sum() <= b for a, b in out_bins):
                gen_mae += abs(est_count - target.sum())/target.sum()
                len_gen += 1
        del output, img, model
        if len_gen ==0:
            len_gen = 1
        torch.cuda.empty_cache()
    mae /= len(test_loader)
    mre /= len(test_loader)
    print("model_param_path:"+model_param_path+" MAE:"+str(mae))
    if len_gen >0:
        print('Gen MAE:',str(gen_mae/len_gen))
    time.sleep(5)
    return 0, float(mae), float(mre), float(gen_mae/len_gen)


def test(model, test_loader, n_split, name, model_name='mall'):
    torch.backends.cudnn.enabled=False

    model_param_path='./checkpoints/'+name+'/split_'+n_split+'.param'
    return val(model, test_loader, model_param_path,model_name)

if __name__=="__main__":
    import sys
    #from mcnn_model import MCNN
    name = sys.argv[1]
    target_path = paths_model[sys.argv[2]]
    n_splits = sys.argv[3]
    result = [[],[],[]]

    for n_split in range(int(n_splits)):
        img_root= f'{target_path}images'
        gt_dmap_root=f"{target_path}ground_truth_npy"
        with open(f'{target_path}test_splits/test_{n_split}.pkl','rb') as fp:
            test_list = pickle.load(fp)
            test_list = [f'{target_path}images/'+item for item in test_list]
        transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        testloader = torch.utils.data.DataLoader(
            RawDataset(test_list, transform, ratio=1, aug=False, ), 
            shuffle=False, batch_size=1, num_workers=0)

        model = DSNet() #.to("cuda")
        r = test(model,testloader,str(n_split),name,sys.argv[2])
        result[0].append(r[1])
        result[1].append(r[2])
        result[2].append(r[3])
    result = np.array(result)
    print("MAE:",result[0].mean(),result[0].std())
    print("MRE:",result[1].mean(),result[1].std())
    print("Out:",result[2].mean(),result[2].std())
    with open(f'./results/{name}_{sys.argv[2]}.pkl','wb') as fp:
        pickle.dump(result, fp)





