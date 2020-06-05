import numpy as np
import numpy.linalg
from numpy.linalg import matrix_rank
import scipy.linalg as la
import itertools
import functools


class Matrix():
    def __init__(self, matrix: np.ndarray):
        '''
        :param matrix: ndarray matrix (C)
        '''
        self.matrix = matrix
        self.matrix = matrix  # the matrix: 2d np.ndarray
        self.size = matrix.shape[0]  # size of axis: int
        # self.eig_val,_ = np.linalg.eig(self.matrix) # eigen values with duplications : np.ndarray
        self.eig_val = self.getEigenValues()  # eigen values with duplications : np.ndarray
        self.charPoly = self.getCharacteristicPolynomial()  # the characteristic Polynom in our presentation : 2d np.ndarray
        self.minPoly = self.getMinimalPolynomial()  # the minimal Polynom in our presentation : 2d np.ndarray
        self.isDiagonal = self.isDiagonalizableMatrix()  # boolean is diagonalizable matrix
        self.eigan_vectors = self.getEigenvectors()  # dictionary- eigenvals as keys,eigenvectors of each eigenval as ndarray *columns*
        self.S = None
        self.N = None
        self.P = self.getPmejardent()
        self.J = self.getJordanForm()
        self.S = self.getSmatrix()
        self.N = self.getNmatrix()

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
        charPoly = np.array([unique_elements, counts_elements])
        return charPoly.transpose()

    def getMinimalPolynomial(self):
        '''
        :return:
        '''
        charP = self.charPoly
        linear_factors = [(self.matrix - charP[i][0] * np.identity(self.size), charP[i][0]) for i in
                          range(charP.shape[0])]  # list of all (A-lambda*I) factors in CharP
        # print(linear_factors)
        # loop from minimum num of linear factors in MinP to Size of matrix
        for cur_num_of_lin_factors in range(self.charPoly.shape[0], self.size + 1):
            # list of combinations for cur_num_of_lin_factors componnents in poly
            poly_lin_factors = itertools.combinations_with_replacement(linear_factors, cur_num_of_lin_factors)
            poly_lin_factors = list(poly_lin_factors)
            poly_results = []
            # calculate results of all polynoms (while removing the tuples with the eigen_values) and append into list
            for cur_lin_factors_tups in poly_lin_factors:
                cur_list = [cur_lin_factor[0] for cur_lin_factor in cur_lin_factors_tups]
                poly_results.append(functools.reduce((lambda x, y: np.dot(x, y)), cur_list))

            zero_index = [i for i in range(len(poly_results)) if numpy.all(abs(poly_results[i]) <= 0.001)]
            # if there is a solution zero, get the result
            if len(zero_index) > 0:
                min_lin_factors = [lin_factor[1] for lin_factor in poly_lin_factors[zero_index[0]]]
                # np_arr=numpy.array(min_lin_factors)
                unique_elements, counts_elements = np.unique(min_lin_factors, return_counts=True)
                minPoly = np.array([unique_elements, counts_elements])
                return minPoly.transpose()
        print("error in min poly with matrix =\n", self.matrix, "\n\n")
        return np.array()  # error

    def isDiagonalizableMatrix(self):
        '''
        returns boolean : True- diagonalizable . else- False
        '''
        # check if all in factors are included in minP and 1-time only
        is_diag = (numpy.array_equal(numpy.sort(self.minPoly[:, 0]), numpy.sort(self.charPoly[:, 0]))) and (
            np.array_equal(self.minPoly[:, 1], np.ones_like(self.minPoly[:, 1])))
        return is_diag

    def getEigenValues(self):
        eig_val, _ = np.linalg.eig(self.matrix)
        return (eig_val.round(decimals=3))

    def getEigenvectors(self):
        '''
        :return: dict of eigen-value : eigen-vectors. {float, ndarray} ndarray=(mat_size,num_of_eigen_vectors)
        '''
        eigenlist = self.charPoly[:, 0]
        dic = {eig: (la.null_space(self.matrix - (eig * np.eye(self.matrix.shape[0])))).round(decimals=5) for eig in
               eigenlist}
        return dic

    def getPmejardent(self):
        '''
        Algorith:
            1. get Jordan form of the matrix
            2. for each eig value:
                2.1 for each block size find v in ker(A-xI)^(block_size) so that
                    v not in ker(A-xI)^(block_size-1) and v is linear independence with
                    all the other vectors we found
                    2.1.1 find jordan chain for v in block_size length
            3. stack_col all chains found
        '''
        J = self.getJordanForm()
        jordan_blocks = self.getJordanBlocks(J)
        p_vector_list = []
        for eig_value, blocks in sorted(jordan_blocks.items(), reverse=True):
            # A = (matrix - eig_value*I)
            A = (self.matrix - (eig_value * np.eye(self.matrix.shape[0])))
            for block_size in blocks:
                # A_pow = A^block_size
                A_pow = np.linalg.matrix_power(A, block_size)
                null_space = la.null_space(A_pow)
                A_small_pow = np.linalg.matrix_power(A, block_size - 1)
                for vector in null_space.T:
                    if (A_small_pow @ vector).any() != 0 and self.isLinearIndependence(p_vector_list + [vector]):
                        p_vector_list += self.findJordanChain(A, vector, block_size)
                        break
        P = np.stack(p_vector_list, axis=1)
        return P

    # find jordan chains and add them to base_list
    def findJordanChain(self, A, vector, block_size):
        jordan_chain = [vector]
        for tmp_pow in range(1, block_size, 1):
            A_pow_tmp = np.linalg.matrix_power(A, tmp_pow)
            result = A_pow_tmp @ vector
            if result.any() != 0:
                jordan_chain.append(result)
            else:
                break
        jordan_chain.reverse()
        return jordan_chain

    def isLinearIndependence(self, vector_list):
        A = np.stack(vector_list, axis=0)
        rank = np.linalg.matrix_rank(A)
        if rank == len(vector_list):
            return True
        else:
            return False

    def getJordanBlocks(self, J):
        jordan_blocks = {}
        i = 0
        j = 0
        size = J.shape[0]
        while i < size and j < size:
            eig_value = J[i][j]
            counter = 1
            blocks = []
            while i < size and j < size:
                if J[i][j] != eig_value:
                    break
                if j + 1 < size and J[i][j + 1] == 1:
                    counter += 1
                else:
                    blocks.append(counter)
                    counter = 1
                i += 1
                j += 1
            blocks.sort(reverse=True)
            jordan_blocks[eig_value] = blocks
        return jordan_blocks

    @staticmethod
    def combine_jordan_blocks(blocks, size):
        res = np.zeros((size, size), dtype=complex)
        i = 0
        for block in blocks:
            for j in range(block[1] - 1):
                res[i + j][i + j] = block[0]
                res[i + j][i + j + 1] = 1

            res[i + block[1] - 1][i + block[1] - 1] = block[0]

            i += block[1]

        return res

    def getJordanForm(self):
        """

        :return: List of tuples
        """

        # ALGORITHM:
        # 1. Get eigenvalue from minimal polynom
        # 2.
        #           Loop1:  for each ev:
        #           Loop2:      for power in range(1 to power_of_ev_in_min_pol):
        #
        #                           # Calculating number of blocks of size power for ev
        #                           num = 2*dimKer(A-lambda*I)^block_size - dimKer(A-lambda*I)^(block_size+1) -
        #                                                                            - dimKer(A-lambda*I)^(block_size-1)
        #                           # Adding blocks to output as tuple of (ev and block size)
        # 3. Construct the output matrix out of Jordan blocks

        result = []
        minimal_pol = self.getMinimalPolynomial()

        for row in minimal_pol:
            ev = row[0]
            n = self.matrix[0].size

            identity_m = np.identity(n)
            tmp_matrix = self.matrix - ev * identity_m  # A - lambda*I

            mat_1 = np.identity(n)
            mat_2 = tmp_matrix
            mat_3 = tmp_matrix @ tmp_matrix

            dim_ker_1 = n - matrix_rank(mat_1)
            dim_ker_2 = n - matrix_rank(mat_2)
            dim_ker_3 = n - matrix_rank(mat_3)

            for block_size in range(1, int(np.real(row[1])) + 1):
                number_of_blocks = 2 * dim_ker_2 - dim_ker_1 - dim_ker_3

                # Updates the arguments for the next iteration
                mat_3 = mat_3 @ tmp_matrix
                dim_ker_1 = dim_ker_2
                dim_ker_2 = dim_ker_3
                dim_ker_3 = n - matrix_rank(mat_3)

                [result.append((ev, block_size)) for _ in range(number_of_blocks)]
        #here add a sort first by block size and then by eig size
        result.sort(reverse=True)
        return Matrix.combine_jordan_blocks(result, n)

    def getSmatrix(self):
        '''

        :return: returns only the S matrix
        '''
        if self.S is None:
            self.getSNMatrices()
        return self.S

    def getNmatrix(self):
        '''

        :return: returns only the S matrix
        '''
        if self.N is None:
            self.getSNMatrices()
        return self.N

    def getSNMatrices(self):

        '''
        the jordan form A = P*J*(P^-1)
        the jordan form can be decomposed to the sum of two matrices:
        a diagonal matrix, and a nilpotent matrix (J - diag(J))
        A = P*(diagonal + nilpotent)*(P^-1)
        A = P*diagonal*(P^-1) + P*nilpotent*(P^-1)
        S = P*diagonal*(P^-1)
        N = P*nilpotent*(P^-1)

        :return: returns S,N of self.matrix
        '''

        J, P = self.getJordanForm(), self.getPmejardent()
        diag_index = np.array([[i, i] for i in range(J.shape[0])])
        diag_matrix = np.zeros_like(J, dtype=complex)
        diag_matrix[diag_index[:, 0], diag_index[:, 1]] = J[diag_index[:, 0], diag_index[:, 1]]
        second_diagonal = np.array([[i, i + 1] for i in range(J.shape[0] - 1)])
        nil_matrix = np.zeros_like(J, dtype=complex)
        nil_matrix[second_diagonal[:, 0], second_diagonal[:, 1]] = J[second_diagonal[:, 0], second_diagonal[:, 1]]
        P_inv = np.linalg.inv(P)
        self.S = P.dot(diag_matrix).dot(P_inv)
        self.N = P.dot(nil_matrix).dot(P_inv)
        return self.S, self.N


if __name__ == '__main__':

    #arr = np.array([[1,0,-4,4],[0,2,0,0],[0,1,1,0],[0,1,0,1]])
    #arr = np.array([[2, 0, 0], [0, 2, 0], [-1, 1, 2]])
    #arr = np.array([[7,1,2,2],[1,4,-1,-1],[-2,1,5,-1],[1,1,2,8]])
    #arr = np.array([[0,-1],[1,0]])
    #arr = np.array([[1,1],[0,1]])
    arr = np.array([[46,93],[71,89]])
    print(arr, "\n")
    mat = Matrix(arr)
    #print("char_poly -\n ", mat.getCharacteristicPolynomial(), "\n")
    print("char_poly -\n ", mat.charPoly, "\n")
    #print("min_poly -\n ", mat.getMinimalPolynomial())
    print("min_poly -\n ", mat.minPoly)
    print("isDiagonal -\n ", mat.isDiagonal)
    #print("eig_vectors -\n ", mat.getEigenvectors())
    print("eig_vectors -\n ", mat.eigan_vectors)
    #P = mat.getPmejardent()
    P = mat.P
    P.round(decimals=10)
    print("P =\n ", P)
    #J = mat.getJordanForm()
    J = mat.J
    print("J =\n ", J)
    invP = np.linalg.inv(P)
    print("new J = \n", (invP @ mat.matrix @ P).round(decimals=5))
    #N = mat.getNmatrix()
    N = mat.N
    #S = mat.getSmatrix()
    S = mat.S
    print("S = \n ", S)
    print("N = \n ", N)
    print("new A = \n", S + N)
