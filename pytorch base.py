import torch 
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
print (torch.__version__)

scalar = torch.tensor(7)
print(scalar)
print (scalar.ndim)

#vector 
vector = torch.tensor([7,7])
print (vector)
print(vector.ndim )
print(vector.shape)

#matrix
matrix = torch.tensor([[7,8,3],
                       [8,9,3]])
print(matrix )#matrix[0]
print(matrix.shape)

#tensor
tensor = torch.tensor([[[1,2,3],
                        [4,5,6],
                        [7,88,99]]])

#rand
rand = torch.rand(2,2,2)
print (rand)

shapetensor = torch.rand(size=(224,224,3))
print(shapetensor)
#sum matrix
tensor = torch.tensor([3,5,6])
print(tensor+10)
print(torch.matmul (tensor,tensor))
print(tensor @ tensor)

#rule of matrix
tensora = [[1,2],
           [2,3],
           [3,3]]
tensorb = [[1,2],
           [2,3],
           [3,3]]
#print(torch.matmul (tensora , tensorb)) اینو ارور میده
#print(torch.matmul (tensora , tensorb.T))

#max , min , sum in list
x = torch.arange (0,100,10)
print(x.min())
print(x.max())
print(torch.mean(x.type(torch.float32 )))# اینجا ار 32 استفادع میکنیم چون ایکس ما 64 و این طولانیه برای میانگین
print(x.sum())

#finding postional min and max
print(x.argmin())
print(x.argmax())

#Reahape , view , staking
p = torch.arange(1,11)
print(f"uns {p.unsqueeze(dim=1)}")
print(p.reshape(2,5))
print(p.reshape(5,2))
print(p.reshape(1,10))
print(p.reshape(10,1))

z = p.view(2,5) # p رو هم مثل زد میکنه

stakk = torch.stack([p,p,p] , dim = 0)
print(stakk)
stakkk = torch.stack([p,p,p] , dim = 1)
print(stakkk)
 
 
#random seed
torch.manual_seed(0)
rac=torch.rand(3,4)
print(rac)