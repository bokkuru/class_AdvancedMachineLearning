import numpy as np
import matplotlib.pyplot as plt

def twoValuePloter(x,y):
    fig = plt.figure()
    plt.scatter(x[:,0],x[:,1], c = y)
    plt.legend()
    plt.grid(True)
    fig.show()

def dataset1():
    n = 100
    x = 3 * (np.random.rand(n,2) -0.5)
    radius = x[:,0]*x[:,0] + x[:,1]*x[:,1]
    y = (radius > 0.7 + 0.1 * np.random.randn(n)) & (radius < 2.2 + 0.1*np.random.randn(n))
    y = 2 * y - 1
    return x, y

def dataset2():
    n = 40
    omega = np.random.randn(1)
    noise = 0.8 * np.random.randn(n)
    x = np.random.randn(n,2)
    y = 2 *(omega * x[:,0] + x[:,1] + noise > 0) - 1
    return x, y

def dataset3():
    m = 20
    n = 40
    r = 2
    A = np.dot(np.random.rand(m,r),np.random.rand(r,n)).flatten()
    ninc = 100
    Q = np.random.permutation(m * n)[:ninc]
    A[Q] = None
    A = A.reshape(m, n)
    return A

def dataset4():
    n = 200
    x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
    y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
    y_d4 = 2 * y_d4 -1
    return x_d4, y_d4

def dataset5():
    n = 200
    x_d5 = 3 * (np.random.rand(n, 4) - 0.5)
    W = np.array([[ 2,  -1, 0.5,],
                [-3,   2,   1,],
                [ 1,   2,   3]])
    y_d5 = np.argmax(np.dot(np.hstack([x_d5[:,:2], np.ones((n, 1))]), W.T)
                            + 0.5 * np.random.randn(n, 3), axis=1)
    return x_d5,y_d5