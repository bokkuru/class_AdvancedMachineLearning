import numpy as np
import matplotlib.pyplot as plt

def f(w,A,u):
    return np.dot(np.dot((w-u).T , A), (w-u))

def grad(w,A,u):
    return np.dot((A + A.T),(w-u))

def soft_threashold(u, lam):
    return np.sign(u) * np.maximum(np.abs(u) - lam, 0.0)

w0 = np.array([3,-1])
A = np.array([[3,0.5],[0.5,1]])
u = np.array([1,2])
step =  1 / max(np.linalg.eig(2*A)[0])
epochNum = 20

resultW = [[w0] for i in range(3)]
for i,lam in enumerate([2,4,6]):
    print("lamda : {}".format(lam))
    w = np.copy(w0)
    print(f(w,A,u))
    for epoch in range(epochNum):
        w = soft_threashold(w - np.dot(step,grad(w,A,u)) , step*lam)
        print(w.shape)
        resultW[i].append(w)
        print(f(w,A,u))
for i in range(3):
    print(resultW[i][-1])
    for j in range(len(resultW[i])):
        resultW[i][j] = np.linalg.norm(resultW[i][j]-resultW[i][-1],ord=2)


fig = plt.figure()
epoch = [ i for i in range(epochNum+1)]
print((resultW[2]-resultW[2][-1]).shape)
plt.plot(epoch, resultW[0] , label = "lamda : 2" )
plt.plot(epoch, resultW[1] , label = "lamda : 4" )
plt.plot(epoch, resultW[2] , label = "lamda : 6" )
ax = plt.gca()
ax.set_yscale('log')
plt.legend()
plt.grid()
fig.savefig("problem2.png")