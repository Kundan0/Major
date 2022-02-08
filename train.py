import torch
import os
from torch.utils.data import DataLoader,random_split
from ClassData import myDataset
from ClassModel import myModel
from DeviceData import DeviceDataLoader
import numpy as np
import matplotlib.pyplot as plt

#colab
PATH=os.path.join("/content")
PATHJ=os.path.join("/content","Major")

#kaggle
# PATH=os.path.join("/kaggle","working")
# PATHJ=os.path.join(PATH,"Major")
# PATHS=os.path.join(PATH,"models")
#PATH=os.curdir

learn_type=1
if learn_type==1:
    model_name="trained1"
    json_name="JSONa1.json"
elif learn_type==2:
    model_name="trained2"
    json_name="JSONb.json"
elif learn_type==4:
    model_name="trained4"
    json_name="JSONd.json"
else:
    model_name="trained3"
    json_name="JSONc.json"
print(model_name)
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
print(device)    
depth_dir=os.path.join(PATH,"Depth2")
of_dir=os.path.join(PATH,"NewOpticalFlow")
an_dir=os.path.join(PATH,"Annotations")
json_dir=os.path.join(PATHJ,json_name)

dataset=myDataset(json_dir,depth_dir,of_dir,an_dir,delta=10)
dataset_size=len(dataset)
train_size=int(dataset_size*0.8)
train_ds, val_ds = random_split(dataset, [train_size,dataset_size-train_size])
train_dl=DataLoader(train_ds,batch_size=64,shuffle=True)
val_dl=DataLoader(val_ds,batch_size=64,shuffle=True)

train_dl=DeviceDataLoader(train_dl,device)
val_dl=DeviceDataLoader(val_dl,device)
lr_rate=0.0001
chkpt_file_pth=os.path.join(PATHJ,"State",model_name)
model=myModel('1',chkpt_file_pth).to(device)
train_loss=[]
validation_loss=[]
def plot_losses():
    
    plt.plot(train_loss, '-bx')
    plt.plot(validation_loss, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

def evaluate(model, val_dl):
    model.eval()
    outputs=[]
    for idx,batch in enumerate(val_dl):
      print("validating batch number ",idx+1)
      output=model.validation_step(batch)
      outputs.append(output)
      
      
    
    return model.validation_epoch_end(outputs)


def fit(epochs,optim,learning_rate,model,train_dl,val_dl):
    optimizer=optim(model.parameters(),learning_rate)
    
    
    try:
        print("Loading Model ...")
        optimizer,trained_epoch,last_tl,last_vl=model.load_model(optimizer)
        print(f"Training loss {last_tl} and validation loss {last_vl}for last epoch {trained_epoch} ")
        print("Successfully loaded the model")
        print("Starting from epoch ",trained_epoch+1)
        
    except:
        trained_epoch=-1
        print("Cannot Load Model")
    
    for ep in range(trained_epoch+1,epochs):
        print("epoch",ep)
        model.train()
        
        train_losses=[]
        for idx,batch in enumerate(train_dl):
            
            print("idx",idx)
            optimizer.zero_grad()
            loss=model.training_step(batch)
            l=loss.detach()
            print("loss",l)
            loss.backward()
            optimizer.step() 
            train_losses.append(l)
            #print("average_Loss for last 20 batches",np.average([x.item() for x in train_losses[-20:]]))
        mean_tl=torch.stack(train_losses).mean().item()
        print("Performing Model Evaluation   ... wait ")
        mean_vl=evaluate(model,val_dl)
        print("Saving model")
        model.save_model(ep,mean_tl,mean_vl,optimizer)
        print("Saved ")
        train_loss.append(mean_tl)
        validation_loss.append(mean_vl)

        print(f"mean validation loss for this epoch {ep}is {mean_vl} /n mean training loss is {mean_tl}")
        
            
            
        
        

fit(500,torch.optim.Adam,lr_rate,model,train_dl,val_dl)

plot_losses()


