# from test_script import test
import sys
import numpy as np
import pickle
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.CAN.test_can import test_can
from models.DSNet.test_dsnet import test_dsnet
from models.MCNN.test_mcnn import test_mcnn

test_model = {
    'MCNN':test_mcnn,
    'CAN':test_can,
    'DSNet':test_dsnet,
}


if __name__ == '__main__':
    
    model = sys.argv[1]
    dataset_type_train = sys.argv[2]
    dataset_type_test = sys.argv[3]
    n_splits = sys.argv[4]
    print(os.getcwd())
    results_path = os.path.join('results', model,dataset_type_train)
    os.makedirs(results_path, exist_ok=True)

    result = [[],[]]
    test = test_model[model]
    for n in range(int(n_splits)):
        result_test = test(['test',dataset_type_train,dataset_type_test,str(n)])
        result[0].append(result_test[0])
        result[1].append(result_test[1])
    result = np.array(result)
    print('MAE: ' ,result[0].mean(),' +- ', result[0].std())
    print('MRE:  ' ,result[1].mean(),' +- ', result[1].std())

    # print(round(100*result[1].mean(),1),'$\pm$', round(100*result[1].std(),1))

    with open(f'./results/{model}/{dataset_type_train}/{dataset_type_train}_{dataset_type_test}.pkl','wb') as fp:
        pickle.dump(result, fp)

