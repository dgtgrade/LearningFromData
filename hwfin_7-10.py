import numpy as np

def load(fp):
    sets = np.loadtxt(fp, dtype=np.float)
    X = sets[:, 1:]
    Y = sets[:, 0]
    return X, Y

trainX, trainY = load("data/hw8_features.train")


def Y_ova_true(Y, i):
    return np.array(Y == i) * 2.0 - 1.0


def Y_ova(Y, i):
    return np.array(Y > 0) * 2.0 - 1.0


def W_reg(Z, Y, l):
    return np.linalg.inv((np.dot(Z.T, Z) + l*np.eye(len(Z.T)))).dot(Z.T).dot(Y)


# problem 7
print("problem 7")
def Z_plain(X):
    return np.column_stack((np.ones(len(X)), X))

trainZ = Z_plain(trainX)
for i in [5, 6, 7, 8, 9]:
    trainY_ova_true = Y_ova_true(trainY, i)
    trainY_ova = Y_ova(trainZ.dot(W_reg(trainZ, trainY_ova_true, 1.0)), i)
    E_in = 1.0 - np.sum(trainY_ova == trainY_ova_true)/len(trainY_ova)
    print(i, "vs all: E_in:", E_in)


# problem 8
print("problem 8")
def Z_transformed(X):
    return np.column_stack((np.ones(len(X)), X, X[:, 0]*X[:, 1], X[:, 0]**2, X[:, 1]**2))

testX, testY = load("data/hw8_features.test")

trainZ = Z_transformed(trainX)
testZ = Z_transformed(testX)

for i in [0, 1, 2, 3, 4]:
    trainY_ova_true = Y_ova_true(trainY, i)
    W = W_reg(trainZ, trainY_ova_true, 1.0)
    testY_ova_true = Y_ova_true(testY, i)
    testY_ova = Y_ova(testZ.dot(W), i)
    E_out = 1.0 - np.sum(testY_ova == testY_ova_true)/len(testY_ova)
    print(i, "vs all: E_out:", E_out)

# problem 9
print("problem 9")

for Z_f in [Z_plain, Z_transformed]:
    trainZ = Z_f(trainX)
    testZ = Z_f(testX)
    print("Z_f", Z_f)
    for i in range(10):
        trainY_ova_true = Y_ova_true(trainY, i)
        W = W_reg(trainZ, trainY_ova_true, 1.0)
        testY_ova_true = Y_ova_true(testY, i)
        trainY_ova = Y_ova(trainZ.dot(W), i)
        testY_ova = Y_ova(testZ.dot(W), i)
        E_in = 1.0 - np.sum(trainY_ova == trainY_ova_true)/len(trainY_ova)
        E_out = 1.0 - np.sum(testY_ova == testY_ova_true)/len(testY_ova)
        print(i, "vs all: E_in:", E_in, "E_out:", E_out)

# problem 9
print("problem 9")
trainZ = Z_transformed(trainX[np.in1d(trainY, [1, 5])])
testZ = Z_transformed(testX[np.in1d(testY, [1, 5])])
trainY = trainY[np.in1d(trainY, [1, 5])]
testY = testY[np.in1d(testY, [1, 5])]

for l in [0.01, 1.0]:
    trainY_ova_true = Y_ova_true(trainY, 5)
    W = W_reg(trainZ, trainY_ova_true, l)
    testY_ova_true = Y_ova_true(testY, 5)
    trainY_ova = Y_ova(trainZ.dot(W), 5)
    testY_ova = Y_ova(testZ.dot(W), 5)
    E_in = 1.0 - np.sum(trainY_ova == trainY_ova_true) / len(trainY_ova)
    E_out = 1.0 - np.sum(testY_ova == testY_ova_true) / len(testY_ova)
    print(i, "vs all: lambda", l, "E_in:", E_in, "E_out:", E_out)

