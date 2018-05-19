# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.linspace(0,20,10000,dtype=np.double)
# y1 = np.power(1+1/x,x)
# y2 = np.ones(10000)*np.e
# plt.plot(x,y1,'blue',lw=2,label='y=(1+1/x)^x')
# plt.plot(x,y2,'red',lw=2,label='y=')
#
# plt.legend(loc='best')
# plt.plot
# plt.show()

import torch
import os
import time
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x
# data = np.sin(np.arange(20)).reshape(4, 5)
data = np.random.rand(100,5)*10
data = np.array(data,dtype=np.float32)
x = torch.from_numpy(data)

b = np.argsort(data,axis=1)
# print(b)
index = b[:,0]
# print(index)
y = np.array(index,dtype=np.int64)
# for i in range(y.shape[0]):
#     y[i][index[i]] = 1
y = torch.from_numpy(y)
net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 5),

    )
print(x.size())
print(x.size())
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

plt.ion()   # something about plotting

optimizer = torch.optim.SGD(net.parameters(),lr=0.05)
loss_func = torch.nn.CrossEntropyLoss()
for eoch in range(300):
    pred = net(x)

    loss = loss_func(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if eoch % 30 ==0:
        _,classify = torch.max(pred,1)
        print("eoch:",sum(classify==y))
# time.sleep(30)
# x = np.array(np.random.random((100,5))*30,dtype=np.float32)
x = np.array([[2,1,5,3,5],
              [7,9,8,2,5],
              [3,3,5,7,9],
              [9,2,8,4,6],
              [6,2,7,9,3],
              [3,0,0,1,8]],dtype=np.float32)
b = np.argsort(x,axis=1)
index = b[:,0]
y = torch.from_numpy(np.array(index,dtype=np.int64))
print("y",y.size())
x = Variable(torch.from_numpy(x))

pred = net(x)
_,test = torch.max(pred,1)
print(test)
print(y)
print(1111)
print(sum(test.data==y))






