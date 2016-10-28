import numpy as np

def f(x): return np.sin(np.pi*x)

n_h = 10000

a = np.zeros(n_h)

for i in range(n_h):

    X = (np.random.random(2)*2.0-1.0).reshape(2, 1)
    y = (f(X)).reshape(2, 1)

    X_pinv = np.linalg.inv((X.T.dot(X))).dot(X.T)
    a[i] = X_pinv.dot(y)

a_hat = a.mean()
print("p4: g_tilde(x) = %.2f*x" % a_hat.round(2))

def g_tilde(x): return a_hat*x
x_ = np.linspace(-1, 1, 1000)
bias = ((g_tilde(x_)-f(x_))**2).mean()
print("p5: bias =", bias)

vars = np.zeros(n_h)
for i in range(n_h):
    def g_D(x): return a[i]*x
    vars[i] = ((g_D(x_) - g_tilde(x_))**2).mean()
var = vars.mean()
print("p6: variance =", var)

