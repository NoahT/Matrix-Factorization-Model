from Model import Model
import numpy as np

'''
@author Noah Teshima
'''
class MatrixFactorizationModel(Model):
    def __init__(self, data, train_prop=.4, validation_prop=.4, test_prop=.2):
        super().__init__(data, train_prop, validation_prop, test_prop)

    
    def calc_loss(self, E):
        return np.linalg.norm(E, ord='fro')**2
    
    '''
    Input:
        S is a collection of 3-tuples (user, item, rating)
            user is the index of the user in list_user
            item is the index of the item in list_items
            rating is the corresponding rating
        Output: mxn feedback matrix
            m is the length of list_user
            n is the length of list_item
            Entry (i, j) is nonzero if list_user[i] reviewed list_item[j]
    '''
    def build_feedback_matrix(self, S, list_users, list_items):
        m = list_users.shape[0]
        n = list_items.shape[0]
        F = np.zeros((m, n))
        for (user_index, item_index, rating) in S:
            F[user_index, item_index] = rating
        return F

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
            print('Frobenius norm: ', iter_cost)
            cost.append(iter_cost)
            # Intermediate U and V to insure simultaneous step
            U_new = U + (alpha * (E @ V))
            V_new = V + (alpha * (E.T @ U))
            U = U_new
            V = V_new
        
        return (U, V, cost)
    
    def train(self, list_users, list_items, alpha, K, d) -> None:
        F_train = self.build_feedback_matrix(super().get_training_data(), list_users, list_items)
        self.F_train = F_train
        self.model = self.gradient_descent(F_train, alpha, K, d)

    def test(self) -> object:
        return self.model[2]


