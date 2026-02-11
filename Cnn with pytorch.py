import torch 
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import dataset
from torchvision.transforms import ToTensor 
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import requests
from pathlib import Path 
from tqdm.auto import tqdm
from helper_functions import accuracy_fn



traindata = datasets.FashionMNIST(
    root = "Desktop\pytorch",
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor(),
    target_transform = None
)

testdata = datasets.FashionMNIST(
    root = "Desktop\pytorch",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = None
)
class_names = traindata.classes



trainloader = DataLoader(traindata, batch_size=32, shuffle=True)
testloader = DataLoader(testdata, batch_size=32, shuffle=False)
train_features_batch, train_labels_batch = next(iter(trainloader))
# flatten image into a 1D vector
flattenmodel = nn.Flatten()
x = train_features_batch[0]
output = flattenmodel(x)

print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")

# Create a flatten layer
flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)

print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    url = "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py"
    request = requests.get(url)
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)
        
        

from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()
        
from torch import nn
class FashionMNISTMOdelV2(nn.Module):
    def __init__(self, input_shape : int , hidden_units: int , output_shape: int ):
        super().__init__()
        self.Con_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape , 
                      out_channels= hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units , 
                      out_channels= hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        self.Con_block_2 = nn.Sequential (
        nn.Conv2d(in_channels= hidden_units,
                    out_channels= hidden_units,
                    kernel_size=3,
                    stride= 1 ,
                    padding= 1,),
        nn.ReLU(),
        nn.Conv2d(in_channels= hidden_units,
                    out_channels= hidden_units,
                    kernel_size=3,
                    stride= 1 ,
                    padding= 1,),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2)    
        )
        self.classifire = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= hidden_units *7*7,  # beacuse we have 10*7*7 in conv2 then flatten nead one lyear 
                      out_features= output_shape)
        )
    
    def forward(self, x):
        x=self.Con_block_1(x)
        #print(x.shape)
        x = self.Con_block_2(x)
        #print(x.shape)this are just use for undrestand of shapes of layer 
        x = self.classifire(x) 
         #print(x.shape)
        return x

torch.manual_seed(42)
model2 = FashionMNISTMOdelV2(input_shape=1,
                             hidden_units=10,
                             output_shape=len(class_names))


from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model2.parameters(), 
                            lr=0.1)
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    train_loss, train_acc = 0, 0
    model.train()
    for batch , (X, y) in enumerate(data_loader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode(): 
        for X, y in data_loader:            
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        
torch.manual_seed(42)


from timeit import default_timer as timer
train_time_start_on_cpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=trainloader, 
        model=model2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(data_loader=testloader,
        model=model2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

train_time_end_on_cpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,)   
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}    
            
model_2_results= eval_model(
    model = model2,
    data_loader=testloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn,
)
print(model_2_results)



#make the perdictions 
def makeperdictions (model:torch.nn.Module,
                     data: list
                     ):
    pred_probs = []
    model.eval()
    with torch.inference_mode() :
        for sample in data :
            sample = torch.unsqueeze(sample , dim = 0)
            pred_logits = model(sample)
            pred_prob = torch.softmax(pred_logits.squeeze() , dim=0)
            pred_probs.append(pred_prob)
    
    return torch.stack(pred_probs)    
    
import random
random.seed(42)
test_samples = []
test_lables = []

for samples , label in random.sample(list(testdata), k= 9):
    test_samples.append(samples)
    test_lables.append(label)
    
pred_probs = makeperdictions(model=model2,
                             data=test_samples)     
print(pred_probs.argmax(dim=1))
        
        
from pathlib import path 
model_path = path("Desktop")
model_path.mkdir(parents= True,
                 exit_ok = True)
modelname = "cnn model"
modelsavepath = model_path / modelname

torch.save(obj=model2.state_dict(),
           f = modelsavepath)        