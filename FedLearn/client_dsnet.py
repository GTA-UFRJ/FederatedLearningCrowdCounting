"""Flower client example using PyTorch for CIFAR-10 image classification."""

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple


import fed_dsnet as fed_train
import flwr as fl
import numpy as np
import torch

from torch.utils.data import DataLoader
from my_dataloader import CrowdDataset
import pickle
import time

import torchvision.transforms as transforms
from dsnet_dataloader import RawDataset
from dsnet_model import DenseScaleNet as DSNet


def chunks(xs, n):
    n = max(1, n)
    return [xs[i:i+n] for i in range(0, len(xs), n)]

USE_FEDBN: bool = False
torch.set_num_threads(1)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
data_path = {
    'mall':'../data/mall/data/',
    'ucf_50':'../data/UCF/data/',
    'ucsd':'../data/UCSD/data/',
    'shang':'../data/ShanghaiTech/data/'
}

# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        model: DSNet,
        trainloader: DataLoader,
        testloader: DataLoader,
        n_split: int,
        name: str,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.n_split = n_split
        self.name = name

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        print("Here")
        self.model.train()
        print("After model train")
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding
            # parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        # cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        fed_train.train(self.model, self.trainloader, self.n_split, self.name)

        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        # loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        loss, accuracy = fed_train.test(self.model, self.testloader, self.n_split, self.name)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def main() -> None:
    """Load data, start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, 10))
    parser.add_argument("--n-split", type=int, required=True, choices=range(0, 10))
    parser.add_argument("--number-clients", type=int, required=True, choices=range(0, 11))
    parser.add_argument("--name", type=str, required=True,)
    args = parser.parse_args()
    
    
    base_path = data_path[args.dataset]
    number_clients = args.number_clients

    with open(f'{base_path}/train_splits/train_{args.n_split}.pkl',"rb") as fp:
        list_images = pickle.load(fp)
        list_images = [f'{base_path}/images/'+item for item in list_images]
    with open(f'{base_path}test_splits/test_{args.n_split}.pkl','rb') as fp:
        test_list = pickle.load(fp)
        test_list = [f'{base_path}/images/'+item for item in test_list]
    list_images = chunks(list_images,number_clients)
    test_list = chunks(test_list,number_clients)
    # print(test_list)
    img_root= f'{base_path}images'
    gt_dmap_root=f"{base_path}ground_truth_npy"
    # trainloader=CrowdDataset(img_root,list_images[int(args.partition_id)],gt_dmap_root,4)
    # trainloader=torch.utils.data.DataLoader(trainloader,batch_size=1,shuffle=True,)

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainloader = torch.utils.data.DataLoader(
        RawDataset(list_images[args.partition_id], transform, aug=True, ratio=8, kernel_path='ground_truth_npy'),
        shuffle=True, batch_size=1, num_workers=0)


    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testloader = torch.utils.data.DataLoader(
        RawDataset(test_list[args.partition_id], transform, ratio=1, aug=False, ), 
        shuffle=False, batch_size=1, num_workers=0)

    model = DSNet() #.to(DEVICE)

    # Start client
    client = CifarClient(model, trainloader, testloader, str(args.n_split), args.name).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
    #fed_train.train(client.model,client.testloader,client.n_split)


if __name__ == "__main__":
    main()
