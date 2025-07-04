from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,img_list,gt_dmap_root,gt_downsample=1):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_downsample=gt_downsample
        #names = [os.path.basename(x) for x in glob.glob(gt_dmap_root)] #new

        self.img_names=[filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root,filename))]
        self.img_names = [filename for filename in self.img_names if filename in img_list]
        #self.img_names=[filename for filename in self.img_names \
         #                  if filename.replace('jpg','npy') in names))]
        self.n_samples=len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)
        try:
            gt_dmap = np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg', '.npy')))

        except:
            gt_dmap = np.load(os.path.join(self.gt_dmap_root, img_name.replace('.png', '.npy')))
        if self.gt_downsample>1: # to downsample image and density-map to match deep-model.
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
#            img = cv2.resize(img,(ds_cols,ds_rows))
            img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
            gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
            gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

            img_tensor=torch.tensor(img,dtype=torch.float)
            gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        return img_tensor,gt_dmap_tensor


# test code
if __name__=="__main__":
    img_root=r"/Users/lucascostafavaro/PycharmProjects/CrowdCounting/ShanghaiTech/part_A/train_data/images"
    gt_dmap_root=r"/Users/lucascostafavaro/PycharmProjects/CrowdCounting/ShanghaiTech/part_A/train_data/ground_truth_npy"
    dataset=CrowdDataset(img_root,gt_dmap_root,gt_downsample=4)
    for i,(img,gt_dmap) in enumerate(dataset):
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(gt_dmap)
        # plt.figure()
        # if i>5:
        #     break
        print(img.shape,gt_dmap.shape)

