import pickle
import numpy as np
import sys
import scipy.stats as stats

if __name__ == '__main__':
    model = sys.argv[1]
    source = sys.argv[2]
    target = sys.argv[3]
    base_path = './results/'
    with open(f'{base_path}{model}/{source}/{source}_{target}.pkl','rb') as fp:
        results = pickle.load(fp)
    print(results)
    #mean, std = results[0].mean(),stats.sem(results)
    #print(results)
    print(results[0].mean(),'+-',results[0].std())
    print('('+str(round(100*results[1].mean(),1))+'$\pm$'+str(round(100*results[1].std(),1))+')\%')

    print(results[1].mean(),'+-',results[1].std())
    print('('+str(round(100*results[2].mean(),1))+'$\pm$'+str(round(100*results[2].std(),1))+')\%')
    #print(round(mean,2))
    #inf, sup = stats.t.interval(0.95, 9, loc=mean, scale=std)
    #print(round(mean-inf,2))
