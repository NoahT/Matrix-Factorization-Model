import numpy as np
import pickle

'''
@author Noah Teshima
'''
class Model:
    def __init__(self, data, train_prop=.4, validation_prop=.4, test_prop=.2) -> None:
        self.data = data
        self.split(train_prop, validation_prop, test_prop)
    
    '''
    K-fold cross validation.
    '''
    def k_foldCV(self, k=10) -> object:
        S_perm = np.copy(self.data)
        np.random.shuffle(S_perm)
        N = S_perm.shape[0]
        size = int(N / k)
        r = N % k
        folds = []
        index = 0
        for i in range(k):
            if r > 0:
                folds.append(S_perm[index:(index+size + 1)])
                r = r - 1
                index = index + size + 1
            else:
                folds.append(S_perm[index:(index+size)])
                index = index + size
        return np.array(folds)

    def data_split(self, S, split=.5) -> tuple:
        S_perm = np.copy(S)
        np.random.shuffle(S_perm)
        l = S_perm.shape[0]
        train_index = int(l * split)
        train = S_perm[:train_index]
        test = S_perm[train_index:]
        return(train, test)

    '''
    Train/Validation/Test split for data given by S.
    Default split is 40/40/20
    '''
    def train_validation_test_split(self, train_prop=.4, validation_prop=.4, test_prop=.2) -> tuple:
        if (train_prop + validation_prop + test_prop != 1):
            raise Exception
        (pseudo_train, test) = self.data_split(self.data, split=(train_prop+validation_prop))
        (train, validation) = self.data_split(pseudo_train, split=(train_prop / (train_prop + validation_prop)))
        return (train, validation, test)

    def split(self, train_prop=.4, val_prop=.4, test_prop=.2) -> None:
        (train_data, val_data, test_data) = self.train_validation_test_split(train_prop, val_prop, test_prop)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train(self) -> None:
        pass

    def test(self) -> object:
        pass

    def get_data(self) -> object:
        return self.data

    def get_training_data(self) -> object:
        return self.train_data

    def get_validation_data(self) -> object:
        return self.val_data

    def get_test_data(self) -> object:
        return self.test_data
    
    def get_model(self) -> object:
        return self.model
    
    def pickle(self, name) -> None:
        pickle.dump(self.get_model(), open(name, 'wb'))
    
    def depickle(self, name) -> None:
        self.model = pickle.load(open(name, 'rb'))
