#file for mejardenet
#
# import numpy as np

#
#
# def getPmejardent(matrix, char_poly, min_poly):
#     # # if matrix is Diagonalizable, no need to find eig vectors muhlalim
#     # if self.isDiagonalizableMatrix:
#     #     P = np.zeros_like(self.matrix)
#     #     for i, vector in enumerate(eig_vectors_map):
#     #         P[:, i] = eig_vectors_map[i]
#     #     return P
#
#     # list to store all the jordan chains
#     base_list = []
#     eig_values = char_poly[:, 0]
#     for eig_value in eig_values:
#         # the pow of (x-eig_value*I) in the min_poly
#         min_poly_pow = -1
#         for row in min_poly:
#             if row[0] == eig_value:
#                 min_poly_pow = row[1]
#                 break
#         if min_poly_pow<0:
#             print("problem in min_poly, ", "(x -", eig_value, "I) power < 0")
#             exit()
#         #finding eig muhlalim
#         # A = (matrix - eig_value*I)
#         A = (matrix - (eig_value * np.eye(matrix.shape[0])))
#         # A_pow = A^min_poly_pow
#         A_pow = np.linalg.matrix_power(A, min_poly_pow)
#
#         # compute A_pow null_space (no pre made function in numpy)
#         # null_space = all eig_vectors of eig_value 0
#         tmp, eig_muhlalim = np.linalg.eig(A_pow)
#         # TODO: add check for eig_vectors (can be 1e-200)
#         null_space = [vector for i, vector in enumerate(eig_muhlalim) if tmp[i] == 0]
#         for vector in null_space:
#             base_list += findJordanChain(A, vector, min_poly_pow)
#
#     # col_stack all jordan chains
#     P = np.stack(base_list, axis=1)
#     return P
#
#
#     # find jordan chains and add them to base_list
# def findJordanChain(A, vector, min_poly_pow):
#     jordan_chain = []
#     for tmp_pow in range(min_poly_pow-1, 0, -1):
#         A_pow_tmp = np.linalg.matrix_power(A, tmp_pow)
#         result = A_pow_tmp @ vector
#         if result != 0:
#             jordan_chain.append(result)
#         else:
#             break
#     #
#     jordan_chain.append(vector)
#     jordan_chain.reverse()
#     return jordan_chain
#
