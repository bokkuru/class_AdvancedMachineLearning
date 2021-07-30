import re
import numpy as np
from numpy import random
import dataset
import matplotlib.pyplot as plt

'''
def J(x,y,w, lam):
    tmp = -y*np.dot(w.T,x).reshape(200,1)
    return sum(np.log(1 + np.exp(tmp))) + lam * np.dot(w.T,w)

def grad(x,y,w, lam):
    tmp = np.exp(-y* np.dot(w.T,x).reshape(200,1))
    ys = np.stack([y for _ in range(x.shape[1])],1)
    tmps = np.stack([tmp / (1 + tmp) for _ in range(x.shape[1])],1)
    return sum(tmps * (-ys * x) ) + 2 * lam * w

def Hessian(x,y,w,lam):
    tmp = np.exp(-y* np.dot(w.T,x).reshape(200,1))
    tmps = np.stack([tmp / ((1 + tmp) *(1 + tmp))  for _ in range(x.shape[1]*x.shape[1])],1)
    tmps = tmps.reshape(-1,x.shape[1],x.shape[1])
    return sum(tmps * np.dot(x,x.reshape(x.shape[0],-1,x.shape[1]))) + 2 * lam * np.eye(x.shape[1])
'''

def J(xs,ys,w,lam):
    ans = 0
    for x,y in zip(xs,ys):
        tmp = -y*np.dot(w.T,x)
        ans += np.log(1 + np.exp(tmp))
    ans += lam * np.dot(w.T,w)
    return ans[:,0] / xs.shape[0]

def grad(xs,ys,w,lam):
    ans = np.zeros(w.shape)
    for x,y in zip(xs,ys):
        tmp = np.exp(-y* np.dot(w.T,x))
        ans += tmp / (1 + tmp) * (-y * x)
    ans += 2 * lam * w
    return ans

def Hessian(xs,ys,w,lam):
    ans = np.zeros((w.shape[0],w.shape[0]))
    for x,y in zip(xs,ys):
        tmp = np.exp(-y* np.dot(w.T,x))
        ans += tmp / ((1+tmp)*(1+tmp)) * np.dot(x,x.T)
    ans += 2 * lam * np.eye(w.shape[0])
    return ans



x, y = dataset.dataset4()
lam = 0.001
firstAlpha = 1
c = np.random.rand()
p = np.random.rand()
epochNum = 60
firstW = ( np.random.rand(4,1) - 0.5 ) * 2
y = y.reshape(200,1)
x = x.reshape(200,4,1)
print("Training")
print("step")
print("c:{}".format(c))
print("p:{}".format(p))

#batch stepest gradient method
print("batch stepest gradient method")
stepestJs = []
w = np.copy(firstW)
alpha = firstAlpha
stepestJs.append(J(x,y,w,lam))
for i in range(epochNum):
    g = grad(x,y,w,lam)
    if J(x,y,w - alpha * g,lam) <= J(x,y,w,lam) + c*p*sum(g*g):
        alpha = p*alpha
    w -= alpha * g
    stepestJs.append(J(x,y,w,lam))



# newton method
print("newton method")
alpha = firstAlpha
newtonJs = []
w = np.copy(firstW)
newtonJs.append(J(x,y,w,lam))
for i in range(epochNum):
    g = grad(x,y,w,lam)
    d = np.dot(np.linalg.inv(Hessian(x,y,w,lam)), g)
    if J(x,y,w - alpha * d,lam) <= J(x,y,w,lam) + c*p*sum(g*d):
        alpha = p*alpha
    w -= alpha * d
    newtonJs.append(J(x,y,w,lam))

# result print
print("batch stepest gradient method result")
print("first : {}".format(stepestJs[0]))
print("finish : {}".format(stepestJs[-1]))
print("newton method result")
print("first : {}".format(newtonJs[0]))
print("finish : {}".format(newtonJs[-1]))

# create figure
best = min(stepestJs[-1],newtonJs[-1])

fig = plt.figure()
epoch = [ i for i in range(epochNum + 1)]
plt.plot(epoch, stepestJs - best, label = "batch stepest gradient method")
plt.plot(epoch, newtonJs - best, label = "newton method")
plt.legend()
plt.grid()
fig.show()
fig.savefig("problem1-1.png")