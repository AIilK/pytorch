import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

xblob , yblob = make_blobs(n_samples=1000,
                           n_features=2,
                           centers=4,
                           cluster_std=1.5,
                           random_state=42)

x_blob = torch.from_numpy(xblob).type(torch.float)
y_blob = torch.from_numpy(yblob).type(torch.LongTensor)

x_blob_train , x_blob_test , y_blob_train , y_blob_test = train_test_split(x_blob,y_blob,test_size=0.2,random_state=42)

plt.figure(figsize=(10,7))
plt.scatter(x=x_blob[:, 0], y=x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

class blobmodel (nn.Module) :
    def __init__ (self, input_features , output_features , hidden_units = 8) :
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    def forward(self,x):
        return self.linear_layer_stack(x)
    
model = blobmodel(input_features=2,
                  output_features=4,
                  hidden_units=8)    
        
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)        

#y_logit = model(x_blob_test)
#y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
#print(y_pred)


def accuracy(ytrue , ypred ) :
    correct = torch.eq(ytrue ,ypred).sum().item()
    acc = (correct/len(ypred))*100
    return acc


epochs = 100
for epoch in range(epochs):
    model.train()
    y_logits = model(x_blob_train).squeeze()
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_fn(y_logits, # Using nn.BCEWithLogitsLoss works with raw logits
                   y_blob_train) 
    acc = accuracy(ytrue=y_blob_train, 
                      ypred=y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_blob_test).squeeze() 
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits,
                            y_blob_test)
        test_acc = accuracy(ytrue=y_blob_test,
                               ypred=test_pred)    
        
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")            
#predictions         
y_pred_probs = torch.softmax(test_logits, dim=1).argmax(dim=1)
print(f"Predictions: {y_pred_probs[:10]}\nLabels: {y_blob_test[:10]}")
print(f"Test accuracy: {accuracy( ytrue =y_blob_test, ypred = y_pred_probs)}%")
        
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.title("Train")
#plot_decision_boundary (model, x_blob_train, y_blob_train)
#plt.subplot(1, 2, 2)
#plt.title("Test")
#plot_decision_boundary(model, x_blob_test, y_blob_test)        
from torchmetrics import Accuracy
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4)
# Calculate accuracy
torchmetrics_accuracy(y_pred_probs , y_blob_test) 