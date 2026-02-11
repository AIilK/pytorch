import torch 
from torch import nn
import matplotlib
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device : {device}")
fig = plt.figure()

weight = 0.7
bias = 0.3

X = torch.arange(0,1,0.02).unsqueeze(dim = 1)
y = weight * X + bias

print(f"X{X}")
print (f"y{y}")

trains = int (0.8 * len(X))
x_tarin , y_train = X[:trains] , y[:trains]
x_test , y_test = X[trains:] , y[trains:]

def plot(traind = x_tarin,
          trainl = y_train,
          testd = x_test,
          testl = y_test,
          predict = None) :
    plt.figure(figsize=(10,7))
    plt.scatter(traind , trainl , c="b" , s=5 , label ="training data" )
    plt.scatter(testd , testl , c="g" , s=5 , label ="testining data" )
    if predict is not None:
        plt.scatter(testd , testl , c="r" , s=4 , label ="testining data" )
    plt.legend(prop = {"size" : 14})
    plt.show()
plot (x_tarin,y_train,x_test,y_test, None)    

class linearregressionmodel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.weaights = nn.Parameter(torch.randn (1, requires_grad = True , dtype = torch.float))
        #self.bias = nn.Parameter(torch.randn (1, requires_grad = True , dtype = torch.float))
        self.linear_layer = nn.Linear(in_features = 1,
                                      out_features = 1) # its specially for linear regression model parameter 
    def forward(self, x: torch.tensor ) -> torch.tensor :
        #return self.weaights * x + self.bias 
        return self.linear_layer(x)
# make model
torch.manual_seed(1)
model0 = linearregressionmodel() 
print(list(model0.parameters()))  
print(model0.state_dict()) 
# make predictions
with torch.inference_mode():
    ypred = model0(x_test)
    print(ypred)
plot(predict=ypred)    

lossfn = nn.L1Loss()
optimizer = torch.optim.SGD(model0.parameters(), lr=0.01 )
#training model 

epochs = 19
for epoch in range(epochs):
    model0.train()
    #forward pass
    y_pred = model0(x_tarin)
    loss = lossfn(y_pred , y_train)
    print(f"loss : {loss}")
    optimizer.zero_grad()
    # autograd calclute the gradiant and loss
    loss.backward()
    #update own weight 
    optimizer.step()
    
    #testing
    model0.eval()
    with torch.inference_mode():
        y_preds = model0(x_test)    
        testloss = lossfn(y_preds , y_test)
    if epoch % 10 == 0:
        print(f"epoch : {epoch} | loss : {loss} | testloss : {testloss} ")     
    #print out model after train
    print(model0.state_dict())    

#perdict 
