#from train_script import train
from models.CAN.train_can import train_can
from models.DSNet.train_dsnet import train_dsnet
from models.MCNN.train_mcnn import train_mcnn
import sys

train_model = {
    'MCNN':train_mcnn,
    'CAN':train_can,
    'DSNet':train_dsnet
}

if __name__ == '__main__':
    model = sys.argv[1] 
    dataset_type = sys.argv[2]
    n_splits = sys.argv[3]
    train = train_model[model]
    for n in range(int(n_splits)):
        train([sys.argv[0], dataset_type,n])