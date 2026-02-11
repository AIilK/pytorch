import matplotlib.pyplot as plt 
from sklearn.datasets import make_circles
import torch
from torch import nn
from sklearn.model_selection import train_test_split

nsamples = 1000
x ,y = make_circles (nsamples,
                     noise= 0.03,
                     random_state= 1)

plt.scatter(x[: , 0] , x[: , 1] , c=y , cmap=plt.cm.RdBu) 
plt.show()

x = torch.from_numpy(x)
y = torch.from_numpy(y)

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.2,random_state=1)

class circle(nn.Module) :
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

model = circle()    
lossfn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)    

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
            
            
            
# relu and sigmoid 
a = torch.arange(-10,10,1,dtype=torch.float32)
plt.plot(a)

def relu(x):
  return torch.maximum(torch.tensor(0), x) # inputs must be tensors

# Pass toy tensor through ReLU function
plt.plot(relu(a));
# Create a custom sigmoid function
def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

plt.plot(sigmoid(a));

# Make predictions
model.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model(xtest))).squeeze()
y_preds[:10], y[:10] # want preds in same format as truth labels