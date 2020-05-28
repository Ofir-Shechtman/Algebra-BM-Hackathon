import numpy as np
import numpy.linalg
from numpy.linalg import matrix_rank


class Matrix():
    def __init__(self, matrix : np.ndarray):
        '''
        :param matrix: ndarray matrix (C)
        '''
        self.matrix = matrix
        self.S = None
        self.N = None

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
        pass

    def getMinimalPolynomial(self):
        '''

        :return:
        '''
        pass

    def isDiagonalizableMatrix(self):
        '''

        :return:
        '''
        pass

    def getEigenvectors(self):
        '''

        :return:
        '''
        pass

    def getPmejardent(self):
        '''
        Algorith:
            1.check if A is Diagonalizable if so, P = stack_col(eig_vectors)
            2.find eig vectors muhlalim and jordan chains:
                2.1 for each eig value a:
                    2.1.1 find Base B for ker(A-aI)^k so that,
                            when k is the pow of (x-a) in the min_poly
                    2.1.2 for each v in B find jordan chain long as possible:
                        2.1.2.1 add v to the jordan chain
                        2.1.2.2 for i=1, if ((A-aI)^i)v !=0 then add ((A-aI)^i)v
                                to the jordan chain
                        2.1.2.3 continue check for i+1 while i<k and ((A-aI)^i)v !=0
                    2.1.3 for each chain check if their is vectors in other chains
                          that linear dependence in the chains vectors
                          2.1.3.1 remove the chains that are linear dependence
                2.2 P = stack_col(all_jordan_chains)
        '''

        char_poly = self.getCharacteristicPolynomial()
        min_poly = self.getMinimalPolynomial()

        # list to store all the jordan chains
        eig_values = char_poly[:,0]
        p_vector_list = []
        for eig_value in eig_values:
            jordan_chain_list = []
            # the pow of (x-eig_value*I) in the min_poly
            min_poly_pow = -1
            r_a = -1
            for row in min_poly:
                if row[0] == eig_value:
                    min_poly_pow = row[1]
                    break
            for row in char_poly:
                if row[0] == eig_value:
                    r_a = row[1]
                    break
            if min_poly_pow < 0 or r_a <= 0:
                print("problem in min_poly or char_poly, ", "(x -", eig_value, "I) power < 0")
                exit()

            # finding eig muhlalim
            # A = (matrix - eig_value*I)
            A = (self.matrix - (eig_value * np.eye(self.matrix.shape[0])))
            # A_pow = A^min_poly_pow
            A_pow = np.linalg.matrix_power(A, min_poly_pow)

            # compute A_pow null_space (no pre made function in numpy)
            # null_space = all eig_vectors of eig_value 0
            tmp, eig_muhlalim = np.linalg.eig(A_pow)
            null_space = [vector for i, vector in enumerate(eig_muhlalim.T)
                          if tmp[i] < 1e-12 and np.linalg.norm(vector) > 1e-12]

            # find jordan chain long as possible for each vector in null_space
            for vector in null_space:
                jordan_chain_list.append(self.findJordanChain(A, vector, min_poly_pow))

            # sort jordan chains by size
            jordan_chain_list.sort(key=len, reverse=True)
            num_of_vectors = r_a

            #check linear dependence between chains
            for i, chain in enumerate(jordan_chain_list):
                j = 1
                while j < len(jordan_chain_list):
                    if j >= len(jordan_chain_list):
                        break
                    next_chain = jordan_chain_list[j]
                    for vector in next_chain:
                        # if one of the vectors in the chains is linear
                        # dependence, remove it
                        if not self.isLinearIndependence(chain+[vector]):
                            jordan_chain_list.pop(j)
                            j -= 1
                            break
                    j += 1

                p_vector_list += reversed(chain)
                num_of_vectors -= len(chain)
                if num_of_vectors <= 0:
                    break

        # col_stack all jordan chains
        P = np.stack(p_vector_list, axis=1)
        return P

    # find jordan chains and add them to base_list
    def findJordanChain(self, A, vector, min_poly_pow):
        jordan_chain = [vector]
        for tmp_pow in range(1, min_poly_pow, 1):
            A_pow_tmp = np.linalg.matrix_power(A, tmp_pow)
            result = A_pow_tmp @ vector
            # debug("jordan chain", [["A_pow_tmp",A_pow_tmp],["tmp_pow",tmp_pow],["vector",vector],["result",result]])
            if result.any() != 0:
                jordan_chain.append(result)
            else:
                break
        return jordan_chain

    def isLinearIndependence(self, vector_list):
        A = np.stack(vector_list, axis=0)
        rank = np.linalg.matrix_rank(A)
        if rank == len(vector_list):
            return True
        else:
            return False

    def getGeoMul(self):
        '''

        :return:
        '''

    @staticmethod
    def combine_jordan_blocks(blocks, size):
        res = np.zeros((size, size))
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

            for block_size in range(1, row[1] + 1):
                number_of_blocks = 2 * dim_ker_2 - dim_ker_1 - dim_ker_3

                # Updates the arguments for the next iteration
                mat_3 = mat_3 @ tmp_matrix
                dim_ker_1 = dim_ker_2
                dim_ker_2 = dim_ker_3
                dim_ker_3 = n - matrix_rank(mat_3)

                [result.append((ev, block_size)) for _ in range(number_of_blocks)]

        return Matrix.combine_jordan_blocks(result, n)

    def getGCD(self):
        '''

        :return:
        '''
        pass

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

        J, P = self.getJordanForm() , self.getPmejardent()
        diag_index = np.array([[i, i] for i in range(J.shape[0])])
        diag_matrix = np.zeros_like(J)
        diag_matrix[diag_index[:, 0], diag_index[:, 1]] = J[diag_index[:, 0], diag_index[:, 1]]
        second_diagonal = np.array([[i, i + 1] for i in range(J.shape[0] - 1)])
        nil_matrix = np.zeros_like(J)
        nil_matrix[second_diagonal[:, 0], second_diagonal[:, 1]] = J[second_diagonal[:, 0], second_diagonal[:, 1]]
        P_inv = np.linalg.inv(P)
        self.S = P.dot(diag_matrix).dot(P_inv)
        self.N = P.dot(nil_matrix).dot(P_inv)
        return self.S, self.N


if __name__ == '__main__':
    '''
    Can do here some plays
    '''
    numpy_matrix = np.ndarray([1,2,3])

    mat = Matrix(numpy_matrix)
    J, P = mat() # will call "__call__"
    pass


