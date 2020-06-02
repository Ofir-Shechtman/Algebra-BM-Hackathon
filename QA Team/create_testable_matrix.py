#!/usr/bin/env python
# coding: utf-8

# %%
import numpy as np
from numpy import matrix


# %%
def create_testable_matrix(p, j):
    a = matrix(p).I @ j @ p
    return a

