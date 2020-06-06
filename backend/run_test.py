#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import numpy.polynomial.polynomial as pol
import scipy.linalg as la

from matrix import Matrix 

EPSILON = 0.001


def create_jordan_matrix(eigenValues, sizeOfBlocks):
    """
    The function create_jordan_matrix get a list of eigen values and a list of the size of blocks.
    the function returns a pair of (J,Size) when J is jordan matrix of type numpy.ndarray and Size is the dimension of the matrix
    
    param: eigenValues is a list of eigen values
    param: sizeOfBlocks is the size of the blocks

    example of use:
    print(create_jordan_matrix([5,4] ,[1,3]))
    expected output:
    (array([[5., 0., 0., 0.],
       [0., 4., 1., 0.],
       [0., 0., 4., 1.],
       [0., 0., 0., 4.]]), 4)

    """
    n=sum(sizeOfBlocks)
    k = 0
    l = 0
    vector1 = np.zeros(n, dtype=complex)
    vector2 = np.zeros(n-1)
    
    for i in  range (len(sizeOfBlocks)):
        for r in range (sizeOfBlocks[i]):
            vector1[l] = eigenValues[k]
            l = l + 1
        k = k + 1

    k = 0
    l = 0
    for i in range (len(sizeOfBlocks)):
        for r in range (sizeOfBlocks[i]-1):
            vector2[l] = 1
            l = l + 1
        if l < n-2: 
            vector2[l] = 0
            l = l + 1

    jordan = np.zeros((n,n), dtype=complex)
    k = 0
    l = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                jordan[i][j] = vector1[k]
                k = k + 1
            if (i + 1) == j:
                jordan[i][j] = vector2[l]
                l = l + 1

    return (jordan,n)        


def create_testable_matrix(p, j):
    a = la.inv(p) @ j @ p
    return a


def create_mejarden_matrix(matrix_size):
    flag = 0
    while flag == 0 :
        a = np.random.randint(-100, 100, size=(matrix_size, matrix_size))
        if np.linalg.matrix_rank(a) == matrix_size:
            return a
    return


#not working
#calc the poly with the matrix as x
def mat_pol_solve (arr_coef, matrix):
    poly_of_matirces = []
    for i in len(arr_coef):
        temp = matrix
        for j in range(0,i):
            temp = temp@matrix
        poly_of_matirces[i] = temp*arr_coef[i]
    return poly_of_matirces.sum()


#checks if the poly is the right minimal poly of the matrix
def test_returned_min_polynom(matrix, poly):
    arr_coef = poly.coef #arr of the coef of the matrix
    
    #checks if the leading coef is 1
    if(arr_coef[sizeof(arr_coef)-1] != 1):
        return false
    
    #if the poly does not become 0 with the matrix as x than it isnt the minimal poly
    if(mat_pol_solve(arr_coef, matrix) != 0):
        return false
    
    arr_mat_eigenvals = la.eig(matrix)
    arr_poly_roots = pol.polyroots(poly)
    
    #checks if all the eigenvalues are roots of the poly and the poly has no other roots
    j=0
    for i in len(arr_poly_roots):
        if(j >= len(arr_mat_eigenvals)): return false #means that there are more roots than eigenvals
        while(j+1 < len(arr_mat_eigenvals) and arr_mat_eigenvals[j] == arr_mat_eigenvals[j+1]): #skips the same eigenvals
            j+=1
        if(arr_mat_eigenvals[j] != arr_poly_roots[i]): # means that there is a diff between eigenvals and roots
            return false
        j+=1
    if(j < len(arr_mat_eigenvals)-1): #means there are eigenvals that are not roots
        return false
    
    #checks if the poly is minimal
    for i in len(arr_mat_eigenvals):
        while(i+1 < len(arr_mat_eigenvals) and arr_mat_eigenvals[i] == arr_mat_eigenvals[i+1]): #skips the same eigenvals
            i+=1
        if(mat_pol_solve(pol.polydiv(arr_coef, (-arr_mat_eigenvals[i], 1)), matrix) == 0):
            return false
    
    return true        
        

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x


def test_returned_characteristic_polynom(polynom,matrix):
    eignValues = la.eigvals(matrix)
    coefArray = polynom.coef
    polyRoots = np.polyroots(polynom)
    
    if(len(polyRoots) != len(eignValues)) : return False
    
    arr1 = selection_sort(eignValus)
    arr2 = selection_sort(polyRoots)
    
    if(arr1 != arr2):
        return False
    
    return True

def is_diagnosiable(M):
    values, vectors = np.linalg.eig(np.array(M))
    if len(vectors) != len(M):
        print("S not diagoniasble")
        return False
    return True


def almost_equal(a, b, threshold=EPSILON):
    c = a - b
    if ((abs(c) <= threshold).all()):
        return True
    else:
        return False


def is_nilpotent(M, epsilon = EPSILON):
    values = np.linalg.eigvals(np.array(M))
    for i in values:
        if abs(i) > epsilon:
            return False
    return True


def test_returned_jordan_chevallier(T, S, N):
    shape_s = S.shape
    shape_n = N.shape
    shape_t = T.shape
    if shape_t != shape_s:
        print("Unmatching dimensions of T,S")
        return False
    if shape_s != shape_n:
        print("Unmatching dimensions of S,N")
        return False
    if shape_t != shape_n:
        print("Unmatching dimensions of T,N")
        return False
    if shape_s[0] != shape_s[1]:
        print("T not square matrix")
        return False
    X = S + N
    if not almost_equal(X, T):
        print("N+S != T")
        return False
    A = N @ S
    B = S @ N
    if not almost_equal(X, T):
        print("NS != SN")
        return False
    if not is_diagnosiable(S):
        print("S is not diagonal")
        return False
    if not is_nilpotent(N):
        print("N is not Nilpotent")
        return False
    return True


def has_jordan_form(j):
    m,n = j.shape
    if m != n:
        return False

    for x in range(m):
        for y in range(n):
            if x > y and j.item((x, y)) != 0:
                return False
            if x + 1 < y and j.item((x, y)) != 0:
                return False
    for x in range(m-1):
        if j.item((x, x)) != j.item((x+1, x+1)):
            if j.item((x, x+1)) != 0:
                return False
        elif j.item((x, x+1)) != 1 and j.item((x, x+1)) != 0:
            return False
    return True


def test_returned_P_Mejardenet_matrix(a, j, p):
    b = la.inv(p) @ a @ p
    if almost_equal(b, j) and has_jordan_form(j):
        return True
    return False


def test_matrix(A, printA):
    if printA:
        print("****   testing matrix  *****")
        print(A)
        print("--------------------------")
    try:
        M = Matrix(A)
    except:
        print("matrix analysis failed")
        return False
    if (test_returned_P_Mejardenet_matrix(A, M.getJordanForm(), M.getPmejardent())):
        print("P and J matricies are good")
    else:
        print("P and J matricies are bad")
        return False
    '''
    On hold - need to convert ndarray to polynomial in order to test it
    if (test_returned_min_polynom(A, M.getCharacteristicPolynomial())):
        print("min polynom is good")
    else:
        print("min polynom is bad")
        return False
    
    if (test_returned_characteristic_polynom(M.getMinimalPolynomial(), A)):
        print("characteristic polynom is good")
    else:
        print("characteristic polynom is bad")
        return False
    '''
    if (test_returned_jordan_chevallier(A, M.S, M.N)):
        print("jordan chevalier matricies are good")
    else:
        print("jordan chevalier matricies are bad")
        return False
        
    return True


def run_test(eigen, blocks):    
    J, size = create_jordan_matrix(eigen, blocks)
    P = create_mejarden_matrix(size)
    A = create_testable_matrix(P, J)
    print("#######   testing Jordan Specific matrix  #######")
    print(J)
    print("--------------------------")
    return test_matrix(A, printA=False)


def random_realNumbers_tests(num_tests):
    for i in range(num_tests):
        # set how many jordan-blocks
        numBlocks = random.randint(2,5)
        eigen = []
        blocks = []
        for b in range(numBlocks):
            # eigenvalue to be from range -10 to 10, and jordan block to be in size 1-5
            eigen.append(random.randint(-3, 3))
            blocks.append(random.randint(1,3))
        run_test(eigen, blocks)


def random_matricies_test(num_tests, matrix_size):
    for i in range(num_tests):
        A = np.random.rand(matrix_size, matrix_size)
        A = (100*A).round(0)
        print("running test on: ")
        print(A)
        try:
            M = Matrix(A)
            if (test_matrix(M)):
                print("test passed")
            else:
                print("test failed")
        except:
            print("Matrix analysis failed")
    
if __name__ == '__main__':
    eigen = [1, 2, 3]
    blocks = [1, 1, 1]
    # diag with 1,2,3 on the diag
    run_test(eigen, blocks)

    eigen = [1, 2, 3]
    blocks = [3, 2, 1]
    # jordan matrix that looks like this:
    '''
    1 1
    0 1
        0 1
        0 0
            3
    '''
    run_test(eigen, blocks)

    eigen = [1+1j, 1-1j, 3]
    blocks = [2, 2, 3]
    # jordan matrix that looks like this:
    '''
    1+i 1
    0 1+i
          1-i 1
          0   1-i
                  3 1
                  0 3 1
                  0 0 3
    '''
    run_test(eigen, blocks)

    eigen = [1, 1, 1, 1]
    blocks = [2, 3, 4, 1]
    # jordan matrix that looks like this:
    '''
    1 1
    0 1 0
        1 1
        0 1 1
        0 0 1 0
              1 1
              0 1 1
              0 0 1 1
              0 0 0 1 0
                      1
    '''
    run_test(eigen, blocks)

    # test for random matricies between sizes specified
    for i in range(2, 4):
        try:
            if(random_matricies_test(10,i)):
                print("test passed")
        except:
            pass
    
    # test for random Jordan matricies
    random_realNumbers_tests(20)
