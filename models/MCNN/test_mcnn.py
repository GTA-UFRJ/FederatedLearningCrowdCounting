import torch
import sys
import pickle
from models.MCNN.mcnn_model import MCNN
from models.MCNN.mcnn_dataloader import CrowdDataset
from data.data_settings import data_path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_mae(img_root,list_image,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    mcnn=MCNN().to(DEVICE)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,list_image,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    mre=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(DEVICE)
            gt_dmap=gt_dmap.to(DEVICE)
            et_dmap=mcnn(img)

            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            mre+=(abs(et_dmap.data.sum()-gt_dmap.data.sum())/gt_dmap.data.sum()).item()

            del img,gt_dmap,et_dmap
    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))
    print('MRE:'+str(mre/len(dataloader)))
    return mae/len(dataloader), mre/len(dataloader)



def test_mcnn(args = sys.argv):
    torch.backends.cudnn.enabled=False
    dataset_type = args[1]
    type_eval = args[2]
    n_split = args[3]
    base_path = data_path[type_eval]
    with open(f'{base_path}test_splits/test_{n_split}.pkl','rb') as fp:
        img_list = pickle.load(fp)

    img_root = f'{base_path}images'
    gt_dmap_root = f"{base_path}ground_truth_npy"

    model_param_path='./checkpoints/MCNN/'+dataset_type+'/split_'+n_split+'.param'
    return cal_mae(img_root,img_list,gt_dmap_root,model_param_path)

if __name__ == '__main__':
    test_mcnn()

