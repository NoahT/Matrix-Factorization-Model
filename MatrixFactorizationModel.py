from Model import Model
import numpy as np
import pandas as pd

class MatrixFactorizationModel(Model):
    def __init__(self, relpath):
        self.data = pd.read_csv(relpath)

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
    def train_validation_test_split(self, S, train_prop=.4, validation_prop=.4, test_prop=.2) -> tuple:
        if (train_prop + validation_prop + test_prop != 1):
            raise Exception
        (pseudo_train, test) = self.data_split(S, split=(train_prop+validation_prop))
        (train, validation) = self.data_split(pseudo_train, split=(train_prop / (train_prop + validation_prop)))
        return (train, validation, test)

    '''
    K-fold cross validation.
    '''
    def k_foldCV(self, S, k=10) -> object:
        S_perm = np.copy(S)
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
    
    def calc_loss(self, E):
        return np.linalg.norm(E, ord='fro')**2

    '''
    Batch gradient descent algorithm.
    (Runtime currently not optimized for sparse matrices.)
    Our update equations are defined as
    U <- U + alpha* EV
    V <- V + alpha E'U
    where E = F - UV' is the residual matrix
    '''
    def gradient_descent(self, F, alpha=0.0005, K=50, d=2) -> tuple:
        m = F.shape[0] # Number of users in feedback matrix
        n = F.shape[1] # Number of items in feedback matrix
        U = np.random.rand(m, d)
        V = np.random.rand(n, d)
        # Sanity check: we keep Frobenius norm at each step to measure our
        # loss wrt the objective function
        cost = []
        # Stop condition is currently not relative to convergence,
        # but a fixed number of iterations
        for i in np.arange(1, K):
            E = F - (U @ V.T) # Residual matrix
            iter_cost = self.calc_loss(E)
            cost.append(iter_cost)
            # Intermediate U and V to insure simultaneous step
            U_new = U + (alpha * (E @ V))
            V_new = V + (alpha * (E.T @ U))
            U = U_new
            V = V_new
        
        return (U, V, cost)

    '''
    Calculate the MSE of F_hat with respect to F
    S is a list of 3-tuples (i, j, rating)
    We calculate the MSE over cells (i, j) found in S.
    '''
    def MSE(self, S, F, F_hat) -> float:
        # size of MSE calculation
        N = S.shape[0]
        mse = 0
        for (row, col, r) in S:
            residual = F_hat[row, col] - F[row, col]
            mse = mse + (residual**2)
        mse = mse / N
        return mse

    def split(self, train, val, test) -> None:
        self.

    def train(self) -> None:
        pass

    def test(self) -> object:
        pass

    def getModel(self) -> object:
        pass


