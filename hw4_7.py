import numpy as np

def f(x): return np.sin(np.pi*x)

def find_ab(h, X, y):

    if 0 == h:
        return 0, y.mean()
    elif 1 == h:
        X_pinv = np.linalg.inv((X.T.dot(X))).dot(X.T)
        return X_pinv.dot(y), 0
    elif 2 == h:
        X = np.column_stack(([1,1],X))
        X_pinv = np.linalg.inv((X.T.dot(X))).dot(X.T)
        return X_pinv.dot(y)
    elif 3 == h:
        X = X**2
        X_pinv = np.linalg.inv((X.T.dot(X))).dot(X.T)
        return X_pinv.dot(y), 0
    elif 4 == h:
        X = np.column_stack(([1,1],X ** 2))
        X_pinv = np.linalg.inv((X.T.dot(X))).dot(X.T)
        return X_pinv.dot(y)

n_h = 10000

for i in range(5):

    a = np.zeros(n_h)
    b = np.zeros(n_h)

    for j in range(n_h):
        X = (np.random.random(2)*2.0-1.0).reshape(2, 1)
        y = (f(X)).reshape(2, 1)
        a[j], b[j]= find_ab(i, X, y)

    print("===")
    print("h%d:" % i)

    a_hat = a.mean()
    b_hat = b.mean()

    print("p4: g_tilde(x) = %.2f*z + %.2f" % (a_hat.round(2) , b_hat.round(2)))

    def g_tilde(x): return a_hat*x + b_hat
    x_ = np.linspace(-1, 1, 1000)
    bias = ((g_tilde(x_)-f(x_))**2).mean()
    print("p5: bias =", bias)

    vars = np.zeros(n_h)

    for j in range(n_h):
        def g_D(x): return a[j]*x + b[j]
        vars[j] = ((g_D(x_) - g_tilde(x_))**2).mean()
    var = vars.mean()
    print("p6: variance =", var)

    print("p7: bias+variance =", bias + var)
