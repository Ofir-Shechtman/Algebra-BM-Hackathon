import numpy as np
from numpy.polynomial.polynomial import Polynomial
import functools
from backend import Matrix, print_matrix, polynomial_to_latex
from dataclasses import dataclass


def assign_matrix_to_polynom(p: Polynomial, A: np.ndarray):
    res = np.zeros_like(A)
    for i, v in enumerate(p.coef):
        t = np.eye(A.shape[0])
        for _ in range(i):
            t = t @ A
        res += t * v
    return res


def gcd(a: Polynomial, b: Polynomial):
    if a == Polynomial(0):
        x, y = 0, 1
        return b, x, y
    g, x, y = gcd(b % a, a)
    x, y = y - (b // a) * x, x
    return g, x, y


def multy_gcd(*vars: Polynomial):
    if len(vars) < 2:
        raise Exception("at least 2 vars needed")
    g, x, y = gcd(vars[0], vars[1])
    rets = [x, y]
    if len(vars) == 2:
        return g, rets
    for var in vars[:2]:
        g, x, y = gcd(g, var)
        rets = list(map(lambda v: v * y, rets))
        rets.append(x)
    return g, rets


@dataclass
class Projection:
    eigenvalue: np.float64
    q: Polynomial
    h: Polynomial

    @property
    def P(self):
        return self.q * self.h


def primary_decomposition(A: np.ndarray):
    m = Matrix(A)
    f = list()
    for root, dup in m.minPoly:
        poly = np.poly([root] * int(dup))
        f.append(Polynomial(np.flip(poly)))
    mx = functools.reduce(lambda a, b: a * b, f)
    q_list = list(map(lambda fi: mx // fi, f))
    g, h_list = multy_gcd(*q_list)
    h_list = list(map(lambda x: x // g, h_list))
    projections = [Projection(root[0], q, h) for root, q, h in zip(m.minPoly, q_list, h_list)]
    return projections


if __name__ == '__main__':
    A = np.array([[-2, 1, 0, 0], [0, -2, 0, 0], [0, 0, -3, 1], [0, 0, 0, -3]], np.float64)
    projections = primary_decomposition(A)
    for p in projections:
        print('q(x)= ', polynomial_to_latex(p.q))
        print('h(x)= ', polynomial_to_latex(p.h))
        print('q(A)h(A)= \n', print_matrix(assign_matrix_to_polynom(p.P, A)), end='\n\n')
