import torch
import torch.nn as nn
class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,type_of,checkpoint_pth,size=(128,72),hl_dim1=32,hl_dim2=32,hl_dim3=32,hl_dim4=32,output_dims=4,loss=nn.MSELoss()):
        super().__init__()
        self.type_of=type_of
        self.output_dims=output_dims
        self.size=size
        
        self.hl_dim1=hl_dim1
        self.hl_dim2=hl_dim2
        self.hl_dim3=hl_dim3
        self.hl_dim4=hl_dim4
        self.checkpoint_path=checkpoint_pth
        
        self.loss=loss
        self.n1=nn.Sequential(
            
            nn.Linear(5*self.size[0]*self.size[1],self.hl_dim1),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim1,self.hl_dim2),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim2,self.hl_dim3),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim3,self.hl_dim4),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim4,self.output_dims),
            
            

        )
        self.n2=nn.Sequential(
            nn.Linear(5*self.size[0]*self.size[1],self.hl_dim1),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim1,self.hl_dim2),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim2,self.hl_dim3),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim3,self.hl_dim4),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim4,self.output_dims),
            

        )
        self.n3=nn.Sequential(
            nn.Linear(5*self.size[0]*self.size[1],self.hl_dim1),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim1,self.hl_dim2),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim2,self.hl_dim3),
            nn.Dropout(p=0.2),
            
            nn.Linear(self.hl_dim3,self.output_dims),
            

        )
        self.n4=nn.Sequential(
            nn.Linear(5*self.size[0]*self.size[1],self.hl_dim1),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim1,self.hl_dim2),
            nn.Dropout(p=0.2),
            nn.Linear(self.hl_dim2,self.hl_dim3),
            nn.Dropout(p=0.2),
            
            nn.Linear(self.hl_dim3,self.output_dims),
            

        )

    def forward(self,image):
        inte=torch.flatten(image,start_dim=1)
        if self.type_of=='1':
            result=self.n1(inte)
        if self.type_of=='2':
            result=self.n2(inte)
        if self.type_of=='3':
            result=self.n3(inte)
        return result
        


    def training_step(self,batch):
        image,label=batch
        result=self(image)
        loss=self.loss(result,label)
        
        return loss
    
    def validation_step(self,batch):
        
        image,label=batch
        result=self(image)
        loss=self.loss(result,label)
        print("val loss for this batch ",loss.detach().item())
        return loss.detach()
    
    def validation_epoch_end(self, outputs):
        print("calculating mean val loss")
        
        epoch_loss = torch.stack(outputs).mean()   # Combine losses
        
        return epoch_loss.item()

    def save_model(self,ep,train_loss,validation_loss,optimizer):
        checkpoint={
            'train_loss':train_loss,
            'validation_loss':validation_loss,
            'epoch':ep,
            'optimizer':optimizer.state_dict(),
            'model':self.state_dict(),
        }
        torch.save(checkpoint,self.checkpoint_path)
        
    def load_model(self,optimizer=None):
        checkpoint=torch.load(self.checkpoint_path)
        self.load_state_dict(checkpoint['model'])
        if optimizer==None:
            return
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer,checkpoint['epoch'],checkpoint['train_loss'],checkpoint['validation_loss']