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

trainZ, trainY = read_data("data/hw6_2_train.txt")
testZ, testY = read_data("data/hw6_2_test.txt")

def error(W, Z, Y):
    return np.sum((Z.dot(W) > 0) != (Y > 0)) / len(Z)

# problem 2
# one-shot learning
W2 = np.linalg.inv((trainZ.T.dot(trainZ))).dot(trainZ.T).dot(trainY)

print(error(W2, trainZ, trainY))
print(error(W2, testZ, testY))

def prob(k):
    l = 10 ** k
    IM = np.eye(trainZ.shape[1])
    W3 = np.linalg.inv((trainZ.T.dot(trainZ)) + l*IM).dot(trainZ.T).dot(trainY)
    print(error(W3, trainZ, trainY))
    print(error(W3, testZ, testY))

# p3
prob(-3)
# p4
prob(3)

# p5
for k in [2, 1, 0, -1, -2]:
    print("k=", k)
    prob(k)


# p6
for k in range(-10, 10):
    print("k=", k)
    prob(k)
