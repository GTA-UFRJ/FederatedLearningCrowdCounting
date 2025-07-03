import pickle
from scipy.stats import ttest_ind
import sys

if __name__ == '__main__':
    base_path = './results/'
    #source = sys.argv[1]
    #target = sys.argv[2]
    #with open(f'{base_path}{source}/{source}_{source}.pkl','rb') as fp:
    #    source_dist = pickle.load(fp)
    #with open(f'{base_path}{source}/{source}_{target}.pkl','rb') as fp:
    #    target_dist = pickle.load(fp)

    print(ttest_ind(source_dist, target_dist).pvalue)
