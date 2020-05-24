import functools

import numpy as np
import numpy.linalg
import itertools

class Matrix():
    
    def __init__(self, matrix : np.ndarray):
        '''
        :param matrix: ndarray matrix (C)
        '''
        self.matrix = matrix #the matrix: 2d np.ndarray
        self.size = matrix.shape[0] # size of axis: int
        self.eig_val,_ = np.linalg.eig(self.matrix) # eigen values with duplications : np.ndarray
        self.charPoly = self.getCharacteristicPolynomial() #the characteristic Polynom in our presentation : 2d np.ndarray
        self.minPoly = self.getMinimalPolynomial() #the minimal Polynom in our presentation : 2d np.ndarray
        self.isDiagonal = self.isDiagonalizableMatrix() # boolean is diagonalizable matrix

    def __call__(self, *args, **kwargs):
        '''

        :param args:
        :param kwargs:
        :return:
        '''
        J = []
        P = []


        return J, P

    def getCharacteristicPolynomial(self):
        '''
        :return:
        '''
        unique_elements, counts_elements = np.unique(self.eig_val, return_counts=True)
        charPoly = np.array([unique_elements,counts_elements])
        return charPoly.transpose()

    def getMinimalPolynomial(self):
        '''

        :return:
        '''
        charP=self.charPoly
        linear_factors = [(self.matrix - charP[i][0] * np.identity(self.size),charP[i][0]) for i in
                          range(charP.shape[0])]  # list of all (A-lambda*I) factors in CharP
        #print(linear_factors)
        # loop from minimum num of linear factors in MinP to Size of matrix
        for cur_num_of_lin_factors in range (self.charPoly.shape[0],self.size+1):
            # list of combinations for cur_num_of_lin_factors componnents in poly
            poly_lin_factors=itertools.combinations_with_replacement(linear_factors,cur_num_of_lin_factors)
            poly_lin_factors=list(poly_lin_factors)
            poly_results=[]
            #calculate results of all polynoms (while removing the tuples with the eigen_values) and append into list
            for cur_lin_factors_tups in poly_lin_factors:
                cur_list=[cur_lin_factor[0] for cur_lin_factor in cur_lin_factors_tups]
                poly_results.append(functools.reduce((lambda x, y: np.dot(x,y)),cur_list))

            zero_index=[i for i in range(len(poly_results)) if numpy.count_nonzero(poly_results[i])==0]
            # if there is a solution zero, get the result
            if len(zero_index)>0:
                min_lin_factors=[lin_factor[1] for lin_factor in poly_lin_factors[zero_index[0]]]
                #np_arr=numpy.array(min_lin_factors)
                unique_elements, counts_elements = np.unique(min_lin_factors, return_counts=True)
                minPoly = np.array([unique_elements, counts_elements])
                return minPoly.transpose()


        return np.array() # error
        
    def isDiagonalizableMatrix(self):
        '''

        returns boolean : True- diagonalizable . else- False
        '''

        is_diag= (numpy.array_equal(numpy.sort(self.minPoly[:,0]),numpy.sort(self.charPoly[:,0]))) and (np.array_equal(self.minPoly[:,1],np.ones_like(self.minPoly[:,1])))
        return is_diag

    def getEigenvectors(self):
        '''

        :return:
        '''
        pass

    def getPmejardent(self):
        '''

        :return:
        '''
        pass

    def getGeoMul(self):
        '''

        :return:
        '''

    def getJordanForm(self):
        '''

        :return:
        '''

    def getGCD(self):
        '''

        :return:
        '''
        pass

    def getSmatrix(self):
        '''

        :return:
        '''
        pass

    def getNmatrix(self):
        '''

        :return:
        '''
        pass

if __name__ == '__main__':
    '''
    Can do here some plays
    '''
    arr =np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,2,0,0],[0,0,0,1,0],[0,0,0,0,2]])
    print(arr,"\n")
    mat = Matrix(arr)
    print(mat.getCharacteristicPolynomial(),"\n")
    print(mat.getMinimalPolynomial())
    print(mat.isDiagonal)

    #
    # print("\n",mat.eig_val)
    # print(np.linalg.eig(arr))
    # #J, P = mat() # will call "__call__"
    # print(arr)
    pass


