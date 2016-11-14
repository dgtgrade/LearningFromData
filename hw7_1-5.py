import numpy as np

def PHI(x1, x2):
    return np.array([
        1,
        x1,
        x2,
        x1**2,
        x2**2,
        x1*x2,
        np.abs(x1 - x2),
        np.abs(x1 + x2)
        ]
    )


def read_data(fp):
    Zl = []
    Yl = []
    with open(fp) as fh:
        for line in fh:
            [x1, x2, y] = list(map(float, line.strip("\n").split()))
            Zl.append(PHI(x1, x2))
            Yl.append([y])
        Z = np.array(Zl)
        Y = np.array(Yl)
    return Z, Y

inZ, inY = read_data("data/hw6_2_train.txt")
testZ, testY = read_data("data/hw6_2_test.txt")

def error(W, Z, Y):
    return np.sum((Z.dot(W) > 0) != (Y > 0)) / len(Z)

# split
valZ, valY = inZ[25:], inY[25:]
trainZ, trainY = inZ[:25], inY[:25]

# problem 1, 2
def prob():
    for k in range(3, 8, 1):
        trainZk = trainZ[:, :k+1]
        valZk = valZ[:, :k+1]
        testZk = testZ[:, :k + 1]
        # one-shot learning
        W2 = np.linalg.inv((trainZk.T.dot(trainZk))).dot(trainZk.T).dot(trainY)

        print("k=", k)
        print(error(W2, trainZk, trainY))
        print(error(W2, valZk, valY))
        print(error(W2, testZk, testY))

# problem 1, 2
prob()

valZ, valY = inZ[:25], inY[:25]
trainZ, trainY = inZ[25:], inY[25:]

# problem 2, 3
prob()
