import pickle
from scipy.stats import ttest_ind
import sys

if __name__ == '__main__':
    base_path = './results/'
    #test1 = sys.argv[1]
    #test2 = sys.argv[2]
    #with open(f'{base_path}{test1}.pkl','rb') as fp:
    #    source_dist = pickle.load(fp)[0]
    #    print(source_dist)
    #with open(f'{base_path}{test2}.pkl','rb') as fp:
    #    target_dist = pickle.load(fp)[0]
    #    print(target_dist)
    target_dist = [0.19941776990890503, 0.14693480730056763, 0.18210358917713165, 0.13291674852371216, 0.14175167679786682]
    source_dist = [ 0.25657105,  0.23140805,  0.17075509,  0.29906878,  0.21505594]
    print(ttest_ind(source_dist, target_dist).pvalue)
