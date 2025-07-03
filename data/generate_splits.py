import pickle
import glob
import sys
import os
from sklearn.model_selection import train_test_split
from data_settings import data_path

def generate_splits(args = sys.argv):
    base_path = data_path[args[1]]
    n_splits = args[2]

    # Create folders if necessary
    os.makedirs(os.path.join(base_path, 'train_splits'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'test_splits'), exist_ok=True)


    # Generate n train/test splits     
    for n in range(int(n_splits)):
        x = os.listdir(f'{base_path}images')
        x_train, x_test = train_test_split(x,shuffle=True)
        with open(f'{base_path}train_splits/train_{n}.pkl','wb') as fp:
            pickle.dump(x_train, fp)
        with open(f'{base_path}test_splits/test_{n}.pkl','wb') as fp:
            pickle.dump(x_test, fp)


if __name__ == '__main__':
    generate_splits()
