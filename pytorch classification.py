from sklearn.datasets import make_circles 
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


nsamples = 1000
x , y = make_circles(nsamples , noise = 0.03 , random_state = 42)

print( x[:5] , y[:5] )
circles = pd.DataFrame({"X" : x[: , 0] ,
                        "X2" : x[:,1],
                        "label" : y })
print(circles.head(10))
plt.scatter (x=x[: , 0],
            y=x[: , 1],
            c=y,
            cmap = plt.cm.RdYlBu)

x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

xtrain , xtest , ytrain , ytest = train_test_split(x,y, test_size=0.2 , random_state=42)

class circlemodel (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features = 2,
                                out_features = 5)
        self.layer = nn.Linear(in_features = 5,
                                out_features = 1)
    def forward(self, x):
        return self.layer2 (self.layer1(X))
    
model = circlemodel()    

# nn.sequential when  we dont use forward in linear just linear we used the sequential is esaier than 

model = nn.Sequential(
    nn.Linear(in_features=2, out_features=5 ),
    # nn.ReLU(), we can add activation function also
    nn.Linear(in_features=5, out_features=1 )
)

lossfn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def accuracy(ytrue , ypred ) :
    correct = torch.eq(ytrue ,ypred).sum().item()
    acc = (correct/len(ypred))*100
    return acc

epochs = 100
for epoch in range(epochs):
    model.train()
    y_logits = model(xtrain).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = lossfn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   ytrain) 
    acc = accuracy(ytrue=ytrain, 
                      ypred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(xtest).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))    
        test_loss = lossfn(test_logits,
                            ytest)
        test_acc = accuracy(ytrue=ytest,
                               ypred=test_pred)    
        
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")     

#improving our models 
#first increase layer and units 

class circlemodel2 (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features = 2,
                                out_features = 10)
        self.layer = nn.Linear(in_features = 10,
                                out_features = 10)
        self.layer3 = nn.Linear(in_features = 10,
                                out_features = 1)
        
    def forward (self,x):
        return self.layer3(self.layer2(self.layer1(x)))

model_1 = circlemodel2()
#train again new module 
# in the end we need non linear  see you im non linear reggression code    