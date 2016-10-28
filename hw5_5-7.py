import numpy as np
import time


def E(u, v):
    return (u*np.exp(v) - 2*v*np.exp(-u))**2


def d_E_d_u(u, v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(np.exp(v)+2*v*np.exp(-u))


def d_E_d_v(u, v):
    return 2*(u*np.exp(v)-2*v*np.exp(-u))*(u*np.exp(v)-2*np.exp(-u))

lr = 0.1


# problem 5-6
i = 0
[u, v] = np.array((1.0, 1.0), dtype=np.float64)
while True:
    e = E(u, v)
    print ("i:%d, u:%e, v:%e, e:%e" % (i, u, v, e))
    time.sleep(0.1)

    if e<1e-14:
        break

    d_u = lr*d_E_d_u(u, v)
    d_v = lr*d_E_d_v(u, v)

    u -= d_u
    v -= d_v

    i += 1

print("---")
[u, v] = np.array((1.0, 1.0), dtype=np.float64)
# problem 7
for i in range(15):

    d_u = lr*d_E_d_u(u, v)
    u -= d_u

    d_v = lr*d_E_d_v(u, v)
    v -= d_v

    e = E(u, v)
    print("i:%d, u:%e, v:%e, e:%e" % (i, u, v, e))
    time.sleep(0.1)

