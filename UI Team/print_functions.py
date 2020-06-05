import numpy as np
from IPython.display import Latex, display
import array_to_latex as a2l
import ipywidgets as ipw
import pandas as pd
from numpy.polynomial import Polynomial as P
from IPython.display import Markdown as md


def print_matrix(np_array_matrix):
    """
    Parameters:
    np_array_matrix - 2 dimensional ndarray.

    Prints the input matrix.
    """

    assert len(
        np_array_matrix.shape) == 2, "Are you rying to print multi dimmensional matrix?"
    latex_code = a2l.to_ltx(
        np_array_matrix, frmt='{}', arraytype='pmatrix', print_out=False)
    return latex_code


def polynomial_to_latex(p: P):
    """Small function to print nicely the polynomial p as we write it in maths,
    in LaTeX code."""
    coefs = p.coef  # List of coefficient, sorted by increasing degrees
    res = ""  # The resulting string
    for i, a in reversed(list(enumerate(coefs))):
        if int(a) == a:  # Remove the trailing .0
            a = int(a)
        if i == 0:  # First coefficient, no need for X
            if a > 0:
                res += "{a} + ".format(a=a)
            elif a < 0:  # Negative a is printed like (a)
                res += "({a}) + ".format(a=a)
            # a = 0 is not displayed
        elif i == 1:  # Second coefficient, only X and not X**i
            if a == 1:  # a = 1 does not need to be displayed
                res += "x + "
            elif a > 0:
                res += "{a} \;x + ".format(a=a)
            elif a < 0:
                res += "({a}) \;x + ".format(a=a)
        else:
            if a == 1:
                # A special care needs to be addressed to put the exponent in {..} in LaTeX
                res += "x^{i} + ".format(i="{%d}" % i)
            elif a > 0:
                res += "{a} \;x^{i} + ".format(a=a, i="{%d}" % i)
            elif a < 0:
                res += "({a}) \;x^{i} + ".format(a=a, i="{%d}" % i)
    return "$" + res[:-3] + "$" if res else ""


def get_extended_polynomial_latex(a):
    """
    Input: ndarray from size (2,n).
    each row will contain: [eigenvalues, power].
    Example: A = [[1,2],
                  [2,4]]
    when A is characteristic polynomial is: (X-1)^2  *  (X-2)^4
    """
    Q = P([0, 0])
    X = P([0, 1])
    for row in a:
        Q += (X-row[0]) ** row[1]
    return(polynomial_to_latex(Q))


def get_packed_polynomial_latex(a):
    str = ""
    for row in a:
        # str+= F"(X-{row[0]})^\{ {int(row[1])} \}  "
        str += "(x - {0})^{1}".format(row[0], "{%d}" % int(row[1]))
    return ("$"+str+"$")
