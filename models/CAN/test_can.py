import torch
import sys
import pickle
from .can_model import CANNet
from .can_dataloader import listDataset
from torchvision import  transforms
import cv2
from data.data_settings import data_path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_mae(dataset_type,list_image,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    model=CANNet().to(DEVICE)
    model.load_state_dict(torch.load(model_param_path))
    dataloader=torch.utils.data.DataLoader(listDataset(list_image,dataset_type,
                       shuffle=False,transform=transforms.Compose([transforms.ToTensor(),]),
                       train=False,num_workers=4),batch_size=1
                       )
    model.eval()
    mae=0
    mre=0
    #print(model)
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(DEVICE)
            et_dmap=model(img).cpu().numpy()
            if et_dmap.sum() < 0:
                print(cv2.threshold(et_dmap,0,0,cv2.THRESH_TOZERO)[1].sum())
                print(gt_dmap.sum())
                print(et_dmap.shape)
                print(gt_dmap.size())
                print(100*'-')

            mae+=abs(abs(et_dmap.sum())-gt_dmap.data.sum()).item()
            mre+=(abs(abs(et_dmap.sum())-gt_dmap.data.sum())/(gt_dmap.data.sum())).item()

            del img,gt_dmap,et_dmap
    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))
    return mae/len(dataloader), mre/len(dataloader)



def test_can(args = sys.argv):
    torch.backends.cudnn.enabled=False
    
    dataset_type = args[1]
    type_eval = args[2]
    n_split = args[3]
    base_path = data_path[type_eval]
    #print(base_path)
    with open(f'{base_path}test_splits/test_{n_split}.pkl','rb') as fp:
        train_list = pickle.load(fp)
        train_list = [f'{base_path}images/'+item for item in  train_list if item != '.DS_Store']
    #print(train_list)

    model_param_path='./checkpoints/CAN/'+dataset_type+'/split_'+n_split+'.param'
    return cal_mae(dataset_type,train_list,model_param_path)
#    np.save('./checkpoints/'+type_model+'_'+type_eval+'.npy',np.array(result))

if __name__ == '__main__':
    test_can()

