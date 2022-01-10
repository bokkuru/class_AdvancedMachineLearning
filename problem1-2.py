import re
import numpy as np
from numpy import random
import dataset
import matplotlib.pyplot as plt


def J(xs,ys,w,lam):
    xs = xs.reshape(200,4)
    ans = 0
    for x,y in zip(xs,ys):
        ans += sum(1+np.exp(-y*np.dot(w,x)))
    for i in w:
        ans += lam * np.dot(i.T,i)
    return ans / ys.size

def grad(xs,ys,w,lam):
    ys = ys.reshape(200,1)
    w = w.reshape(4,1)
    ans = np.zeros(w.shape)
    for x,y in zip(xs,ys):
        tmp = np.exp(-y* np.dot(w.T,x))
        ans += tmp / (1 + tmp) * (-y * x)
    ans += 2 * lam * w
    return ans / ys.size

def Hessian(xs,ys,w,lam):
    ans = np.zeros((w.shape[0],w.shape[0]))
    for x,y in zip(xs,ys):
        tmp = np.exp(-y* np.dot(w.T,x))
        ans += tmp / ((1+tmp)*(1+tmp)) * np.dot(x,x.T)
    ans += 2 * lam * np.eye(w.shape[0])
    return ans



x, y = dataset.dataset4()
lam = 0.001
firstAlpha = 1.0
c = np.random.rand()
p = np.random.rand()
epochNum = 60
firstWs = ( np.random.rand(3,4) - 0.5 ) * 2
ys = np.identity(3)[y] * 2 - 1
x = x.reshape(200,4,1)
print("Training")
print("step")
print("c:{}".format(c))
print("p:{}".format(p))

#batch stepest gradient method
print("batch stepest gradient method")
stepestJs = []
ws = np.copy(firstWs)
alpha = firstAlpha
stepestJs.append(J(x,ys,ws,lam))
for i in range(epochNum):
    for k in range(3):
        y = ys[:,k]
        w = ws[k]
        g = grad(x,y,w,lam)
        #if J(x,y,w - alpha * g,lam) <= J(x,y,w,lam) + c*p*sum(g*g):
        #    alpha = p*alpha
        ws[k] -= (alpha / np.sqrt(i+1) * g).reshape(4)
    stepestJs.append(J(x,ys,ws,lam))
print(stepestJs)



# newton method
print("newton method")
alpha = firstAlpha
newtonJs = []
ws = np.copy(firstWs)
newtonJs.append(J(x,ys,ws,lam))
for i in range(epochNum):
    for k in range(3):
        y = ys[:,k]
        w = ws[k]
        g = grad(x,y,w,lam)
        d = np.dot(np.linalg.inv(Hessian(x,y,w,lam)), g)
        #if J(x,y,w - alpha * d,lam) <= J(x,y,w,lam) + c*p*sum(g*d):
        #    alpha = p*alpha
        ws[k] -= (alpha / np.sqrt(i+1) * d).reshape(4)
    newtonJs.append(J(x,ys,ws,lam))
print(newtonJs)

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
fig.savefig("problem1-2.png")