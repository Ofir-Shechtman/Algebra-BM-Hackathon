#!/usr/bin/env python
# coding: utf-8

import numpy as np

# eigenValues is a list of eigen values
# size of block is the size of the blocks

#All rights reserved to Aviad Shiber & Tal Gelbard inc
"""
The function create_jordan_matrix get a list of eigen values and a list of the size of blocks.
the function returns a pair of (J,Size) when J is jordan matrix of type numpy.ndarray and Size is the dimension of the matrix
"""
def create_jordan_matrix(eigenValues, sizeOfBlocks):
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


"""
example of use:
    print(create_jordan_matrix([5,4] ,[1,3]))
expected output:
    (array([[5., 0., 0., 0.],
       [0., 4., 1., 0.],
       [0., 0., 4., 1.],
       [0., 0., 0., 4.]]), 4)
"""
        
        

        