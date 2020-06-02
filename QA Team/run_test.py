#!/usr/bin/env python
# coding: utf-8

import random

from create_testable_matrix 			  import create_testable_matrix
from create_mejarden_matrix				  import create_mejarden_matrix
from create_jordan_matrix				  import create_jordan_matrix
from test_returned_min_polynom 			  import test_returned_min_polynom
from test_char_polynom 					  import test_returned_characteristic_polynom
from test_returned_jordan_chevallier 	  import test_returned_jordan_chevallier
from test_returned_P_Mejardenet_matrix    import test_returned_P_Mejardenet_matrix

from matrix import # Mathmatics Matrix - put files in the right location

def run_test(eigen, blocks):	
	J, size = create_jordan_matrix(eigen, blocks)
	P = create_mejarden_matrix(size)
	A = create_testable_matrix(P, J)
	M = Matrix(A)
	
	if (test_returned_P_Mejardenet_matrix(A, M.getJordanForm(), M.getPmejardent())):
		print("P and J matricies are good")
	else:
		print("P and J matricies are bad")
		return False
	
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
	
	if (test_returned_jordan_chevalier(A, M.getSmatrix(), M.getNmatrix())):
		print("jordan chevalier matricies are good")
	else:
		print("jordan chevalier matricies are bad")
		return False
		
	return True

def random_realNumbers_tests(num_tests):
	for i in range(num_tests):
		# set how many jordan-blocks
		blocks = random.randint(1,5)
		eigen = []
		blocks = []
		for b in range(blocks):
			# eigenvalue to be from range -10 to 10, and jordan block to be in size 1-5
			eigen.append(random.randint(-10,10)
			blocks.append(random.randint(1,5))
		run_test(eigen, blocks)


eigen = [1, 2, 3]
blocks = [1, 1, 1]
# diag with 1,2,3 on the diag
run_test(eigen, blocks)

eigen = [1, 0, 3]
blocks = [1, 2, 1]
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


random_realNumbers_tests(50)


