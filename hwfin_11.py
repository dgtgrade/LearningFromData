import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
Y = np.array([-1, -1, -1, 1, 1, 1, 1])

def XtoZ(X):
    return np.column_stack((
        X[:, 1]**2 - 2 * X[:, 0] - 1,
        X[:, 0]**2 - 2 * X[:, 1] + 1))

Ex = np.array(
    [[-1, 1, -0.5], [1, -1, -0.5], [1, 0, -0.5], [0, 1, -0.5]])

# problem 11
plt.ion()
plt.scatter(X[:, 0], X[:, 1], s=400, c=Y, cmap='bwr', vmin=-1, vmax=1)
plt.show()
plt.waitforbuttonpress()
plt.clf()
Z = XtoZ(X)
plt.scatter(Z[:, 0], Z[:, 1], s=400, c=Y, cmap='bwr', vmin=-1, vmax=1)
plt.show()
plt.waitforbuttonpress()

z1_min = np.min(Z[:, 0])
z1_max = np.max(Z[:, 0])
z2_min = np.min(Z[:, 1])
z2_max = np.max(Z[:, 1])

def draw_line(W, b):
    # w1*z1 + w2*z2 + b = 0
    # w2 = - 1/w2 * (w1*z1 + b)
    if W[1] != 0:
        z2_1 = - 1 / W[1] * (W[0] * z1_min + b)
        z2_2 = - 1 / W[1] * (W[0] * z1_max + b)
        l, = plt.plot([z1_min, z1_max], [z2_1, z2_2],
                     marker="", linewidth=3, color='black')
    else:
        z1 = -b / W[0]
        l, = plt.plot([z1, z1], [z2_min, z2_max],
                     marker="", linewidth=3, color='black')
    return l

for ex in Ex:
    l = draw_line(ex[0:2], ex[2])
    plt.show()
    plt.waitforbuttonpress()
    l.remove()
