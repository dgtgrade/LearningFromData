import numpy as np
import time

# Constants
L=1000 # sufficiently Large number
N=10 # Number of points
R=1000 # Runs

# pick random L points in 1.0(bias)*[-1,1]*[-1,1]
X=np.empty((L,3))
X[:,0]=1.0 # bias
X[:,1:3]=np.random.uniform(-1.0,1.0,(L,2))

# run
iters = np.zeros(R)
disagree = np.zeros(R)

for r in range(1,R+1):

    # pick random two points for f0
    (xi1,xi2)=np.random.permutation(L)[0:2]
    (x1,x2)=(X[xi1],X[xi2])
    # calculate m,b for f0(x_1)=m*x_1+b
    m=(x1[2]-x2[2])/(x1[1]-x2[1])
    b=x1[2]-m*x1[1]
    # function f
    f = lambda x: np.sign(np.sign(x[2]-(m*x[1]+b))+0.5)

    # N random points from X
    XN = X[np.random.permutation(len(X))[0:N]]

    ########
    # PLA
    ########
    # Initialize W
    W = np.zeros(3)
    g = lambda x: np.sign(W[0]+W[1]*x[1]+W[2]*x[2])
    i = 0

    while True:

        i += 1
        #time.sleep(0.1)

        # index of misclassified points
        y = f(XN.T) # target
        y_h = g(XN.T) # output
        mpi = np.where(y != y_h)[0]

        if (len(mpi)>0):
            # perceptron rule
            k = np.random.randint(len(mpi))
            W += y[mpi[k]]*XN[mpi[k]]
        else:
            iters[r-1] = i
            y = f(X.T)  # target
            y_h = g(X.T)  # output
            disagree[r-1] = np.sum(y != y_h)

            break

    print("Run:",r,"Iteration",i)

print(np.mean(iters))
print(np.mean(disagree))
