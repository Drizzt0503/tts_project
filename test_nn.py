import torch
from torch import nn
import numpy as np
def get_lid( lid):
    llist = lid.strip().split(' ')
    llist = torch.LongTensor([int(x) for x in llist])
    return llist
aaa='1 2 1 3 4'

bbb=get_lid(aaa)
print(bbb,bbb.type())


'''
#embedding = nn.Embedding(10, 3)
#input = torch.LongTensor([[1, 2, 4, 5], [4, 0, 3, 9]])
m1 = nn.Conv1d(3, 5, 1)
m2 = nn.Linear(4, 5)
in1 = torch.rand(2, 1, 3)
print(in1)
in2= in1.expand(-1,4,-1)
print(in2)
#print(input)
#out2 = m1(input)
#print(out2)
out =m2(input)
print(out)
a=input.numpy()[0]
print(a)
b=m2.weight.detach().numpy()
print(b)
c=m2.bias.detach().numpy()
print(c)

for any in a:
    d=np.matmul(b,any)+c
    print(d)
'''
