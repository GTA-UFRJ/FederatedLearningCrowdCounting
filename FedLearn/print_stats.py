import pickle
import numpy as np
import sys
import scipy.stats as stats

if __name__ == '__main__':
    model = sys.argv[1]
    target = sys.argv[2]
    base_path = './results/'
    with open(f'{base_path}{model}_{target}.pkl','rb') as fp:
        results = pickle.load(fp)[1]
    mean, std = results.mean(),stats.sem(results)
    print(list(results))
    print(mean,'+-',results.std())
    print(round(mean,2))
    inf, sup = stats.t.interval(0.95, 9, loc=mean, scale=std)
    print(round(mean-inf,2))
