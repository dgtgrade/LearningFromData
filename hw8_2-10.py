import numpy as np
from sklearn import svm

def load_data(fp):
    data_set = np.loadtxt(fp, dtype=np.float64)
    return data_set[:, 0], data_set[:, 1:]

trainY, trainX = load_data("data/hw8_features.train")
testY, testX = load_data("data/hw8_features.test")

def kernel_pQ(Q):
    return lambda x_n, x_m: (1 + x_n.dot(x_m.T))**Q

def fit(clf, X, Y):
    clf.fit(X, Y)
    return np.sum(clf.predict(X) == Y) / len(X)

# problem 2-4
print("Problem 2-4")
clf = svm.SVC(kernel=kernel_pQ(2), C=0.01)
for i in range(10):
    trainY_ova = np.array([1 if y == i else -1 for y in trainY])
    E_in = 1.0 - fit(clf, trainX, trainY_ova)
    print(i, "vs. all", "E_in:", E_in, "n_support:", np.sum(clf.n_support_))

def ovo(X, Y, o1, o2):
    X_ = X[np.in1d(Y, [o1, o2])]
    Y_ = Y[np.in1d(Y, [o1, o2])]
    Y_ovo = np.array([1 if y == o1 else -1 for y in Y_])
    return X_, Y_, Y_ovo

train_1or5_X, train_1or5_Y, trainY_1v5 = ovo(trainX, trainY, 1, 5)
test_1or5_X, test_1or5_Y, testY_1v5 = ovo(testX, testY, 1, 5)

# problem 5
print("Problem 5")
for c in [0.001, 0.01, 0.1, 1]:
    clf = svm.SVC(kernel=kernel_pQ(2), C=c)
    E_in = 1.0 - fit(clf, train_1or5_X, trainY_1v5)
    E_out = 1.0 - np.sum(clf.predict(test_1or5_X) == testY_1v5) / len(test_1or5_X)
    print("C:", c, "E_in:", E_in, "E_out:", E_out, "n_support:", np.sum(clf.n_support_))

# problem 6
print("Problem 6")
for c in [0.0001, 0.001, 0.01, 1]:
    for q in [2, 5]:
        clf = svm.SVC(kernel=kernel_pQ(q), C=c)
        E_in = 1.0 - fit(clf, train_1or5_X, trainY_1v5)
        E_out = 1.0 - np.sum(clf.predict(test_1or5_X) == testY_1v5) / len(test_1or5_X)
        print("C:", c, "Q:", q, "E_in:", E_in, "E_out:", E_out, "n_support:", np.sum(clf.n_support_))

# problem 7-8
print("Problem 7-8")
selected = {
    "0.0001": 0,
    "0.001": 0,
    "0.01": 0,
    "0.1": 0,
    "1": 0
}
RUN = 100
FOLD = 10
m = len(train_1or5_X)
ss = [int(m/FOLD * i) for i in range(FOLD+1)]
E_v_total = selected.copy()
for r in range(RUN):
    s = np.random.randint(FOLD)
    E_v_min = 1.0
    c_min = 0.0001
    t_X = train_1or5_X[np.r_[0:ss[s], ss[s+1]:m]]
    t_Y = trainY_1v5[np.r_[0:ss[s], ss[s+1]:m]]
    v_X = train_1or5_X[ss[s]:ss[s+1]]
    v_Y = trainY_1v5[ss[s]:ss[s+1]]
    for c in [0.0001, 0.001, 0.01, 0.1, 1]:
        clf = svm.SVC(kernel=kernel_pQ(2), C=c)
        E_t = 1.0 - fit(clf, train_1or5_X, trainY_1v5)
        E_v = 1.0 - np.sum(clf.predict(v_X) == v_Y) / len(v_X)
        E_v_total[str(c)] += E_v
        if E_v < E_v_min:
            E_v_min = E_v
            c_min = c
    selected[str(c_min)] += 1

for c in sorted(selected):
    print("c:", c, "selected:", selected[c], "E_v_mean:", E_v_total[c]/RUN)

# problem 9-10
print("Problem 9-10")
for c in [0.01, 1, 100, 10**4, 10**6]:
    clf = svm.SVC(kernel="rbf", C=c, gamma=1.0)
    E_in = 1.0 - fit(clf, train_1or5_X, trainY_1v5)
    E_out = 1.0 - np.sum(clf.predict(test_1or5_X) == testY_1v5) / len(test_1or5_X)
    print("C:", c, "E_in:", E_in, "E_out:", E_out, "n_support:", np.sum(clf.n_support_))
