import math
import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.05

def mH(N): return N**10
def f(N): return 4 * mH(2*N)*np.exp(-1/8*(epsilon)**2*N)
# def f(N): return 2 * mH(N)*np.exp(-2*(epsilon)**2*N)


minn = 440000
maxn = 460000
ns = np.linspace(minn,maxn,100)
Ps = f(ns)

print(np.column_stack((ns,Ps)))

# solution 1: (d)




