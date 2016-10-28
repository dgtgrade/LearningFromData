import math
import numpy as np

N=5
dVC = 50
delta = 0.05

def mH(N): return N**dVC
def log_mH(N): return dVC*np.log(N)

a = math.sqrt(8/N*math.log(4*mH(2*N))/ delta)
print("a:", a)

b = math.sqrt(2*math.log(2*N*mH(N))/N)+math.sqrt(2/N*math.log(1/delta))+1/N
print("b:", a)

es = np.linspace(4,6.0,100)

cs = np.sqrt(1/N*(2*es+np.log(6*mH(2*N)/delta)))
print("c:")
print(es[es<cs])

ds = np.sqrt(1/(2*N)*((4*es*(1+es))+np.log(4)+log_mH(N**2)-np.log(delta)))
print("d:")
print(es[es<ds])

