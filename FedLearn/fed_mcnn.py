import os
import torch
import torch.nn as nn
import sys
import time
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
import numpy as np
from glob import glob
import pickle



def collate_fn(batch):
    return tuple(zip(*batch))

paths_model = {
    'ucf_50':'../data/UCF/data/',
    'shang':'../data/ShanghaiTech/data/',
    'ucsd':'../data/UCSD/data/',
    'mall':'../data/mall/data/',
    'rio':'../data/rio/data/',
    'drone':'../data/VisDrone2020-CC/data/'
}

def train(model, trainloader, n_split, name):
    # type_model = argv[1]
    # base_path = paths_model[argv[1]]
    # n_split = argv[2]

    torch.backends.cudnn.enabled=False
    device=torch.device("cuda")

    mcnn=model
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-8,
                                momentum=0.95
    )
    # img_root= f'{base_path}/images'
    # gt_dmap_root=f"{base_path}/ground_truth_npy"
    dataloader=trainloader
    print(trainloader)
    
    #training phase
    if not os.path.exists('./checkpoints/'+name):
        os.mkdir('./checkpoints/'+name)
    #min_mae=10000
    #min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    #test_error_list=[]
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(0,20):
        mcnn.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(dataloader):
        # for i, (img_b) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)

            # forward propagation
            et_dmap=mcnn(img)
            # calculate loss
            loss=criterion(et_dmap,gt_dmap)
            if i % 100 == 0:
                print(loss)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
        torch.save(mcnn.state_dict(),'./checkpoints/'+name+'/split_'+str(n_split)+".param")

def cal_mae(model,testloader,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cuda")
    mcnn=model
    mcnn.load_state_dict(torch.load(model_param_path))
    # dataset=CrowdDataset(img_root,list_image,gt_dmap_root,4)
    dataloader=testloader
    mcnn.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
           
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            
            # forward propagation
            et_dmap=mcnn(img)
            #print(et_dmap.data.sum())
            #print(gt_dmap.data.sum())
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            # print(mae)
            if i % 20 == 0:
                print(et_dmap.data.sum(),' ', gt_dmap.data.sum())
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))
    return 0, mae/len(dataloader)



def test(model,testloader,n_split,name):
    torch.backends.cudnn.enabled=False
    # type_model = args[1]
    # type_eval = args[2]
    # n_split = args[3]
    # base_path = paths_model[type_eval]
    # with open(f'{base_path}test_splits/test_{n_split}.pkl','rb') as fp:
    #     img_list = pickle.load(fp)

    # img_root = f'{base_path}images'
    # gt_dmap_root = f"{base_path}ground_truth_npy"
    model_param_path='./checkpoints/'+name+'/split_'+str(n_split)+".param"
    return cal_mae(model, testloader,model_param_path)


if __name__=="__main__":
    import sys
    from mcnn_model import MCNN
    name = sys.argv[1]
    target_path = paths_model[sys.argv[2]]
    n_splits = sys.argv[3]
    result = []

    for n_split in range(int(n_splits)):
        img_root= f'{target_path}images'
        gt_dmap_root=f"{target_path}ground_truth_npy"
        with open(f'{target_path}test_splits/test_{n_split}.pkl','rb') as fp:
            test_list = pickle.load(fp)

        testloader=CrowdDataset(img_root,test_list,gt_dmap_root,4)
        testloader=torch.utils.data.DataLoader(testloader,batch_size=1,shuffle=False)
        model = MCNN().to("cuda")
        result.append(test(model,testloader,n_split,name)[1])
    result = np.array(result)
    print("MAE:",result.mean(),result.std())
    with open(f'./results/{name}_{sys.argv[2]}.pkl','wb') as fp:
        pickle.dump(result, fp)
        

