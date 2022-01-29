import torch
import os
from torch.utils.data import DataLoader,random_split
from ClassData import myDataset
from ClassModel import myModel
from DeviceData import DeviceDataLoader
import numpy as np

PATH=os.path.join("/content","drive","MyDrive")
PATHJ=os.path.join("/content","Major")

#PATH=os.curdir

learn_type=2
if learn_type==1:
    model_name="trained1"
    json_name="JSONa.json"
elif learn_type==2:
    model_name="trained2"
    json_name="JSONb.json"
else:
    model_name="trained3"
    json_name="JSONc.json"
print(model_name)
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
print(device)    
depth_dir=os.path.join(PATH,"Depth2","All")
of_dir=os.path.join(PATH,"NewOpticalFlow")
an_dir=os.path.join(PATH,"Annotations")
json_dir=os.path.join(PATHJ,json_name)

dataset=myDataset(json_dir,depth_dir,of_dir,an_dir)
dataset_size=len(dataset)
train_size=int(dataset_size*0.9)
train_ds, val_ds = random_split(dataset, [train_size,dataset_size-train_size])
train_dl=DataLoader(train_ds,batch_size=4,shuffle=True)
val_dl=DataLoader(val_ds,batch_size=4,shuffle=True)

train_dl=DeviceDataLoader(train_dl,device)
val_dl=DeviceDataLoader(val_dl,device)

model=myModel().to(device)
def evaluate(model, val_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_dl]
    return model.validation_epoch_end(outputs)
def fit(epochs,model,train_dl,val_dl,learning_rate,optim=torch.optim.SGD):
    optimizer=optim(model.parameters(),learning_rate)
    
    
    try:
        model.load_state_dict(torch.load(os.path.join(PATHJ,"State",model_name)))
        print("Loading Model ...")
    except:
        print("Cannot Load")

    for ep in range(epochs):
        print("epoch",ep)
        history=[]
        for idx,batch in enumerate(train_dl):
            print("idx",idx)
            optimizer.zero_grad()
            loss=model.training_step(batch)
            l=loss.detach().cpu().item()
            print("loss",l)
            loss.backward()
            optimizer.step() 
            history.append(l)
            print("average_Loss for last 50 batches",np.average(history[-50:]))
        torch.save(model.state_dict(),os.path.join(PATHJ,"State",model_name))
        print("saving model")
        print("evaluation  model ... wait ")
        result=evaluate(model,val_dl)
        print("validation loss",result['val_loss'])    
#comments added for branch2            
        
        

fit(10,model,train_dl,val_dl,0.00001,torch.optim.Adam)




