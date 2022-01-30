import torch
import torch.nn as nn
class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,size=(128,72),hl_dim1=64,hl_dim2=64,hl_dim3=128,output_dims=4,loss=nn.MSELoss()):
        super().__init__()
        
        self.output_dims=output_dims
        self.size=size
        self.hl_dim1=hl_dim1
        self.hl_dim2=hl_dim2
        self.hl_dim3=hl_dim3

        
        self.loss=loss
        self.n1=nn.Sequential(
            
            nn.Linear(5*self.size[0]*self.size[1],self.hl_dim1),
            nn.Linear(self.hl_dim1,self.hl_dim2),
            nn.Linear(self.hl_dim2,self.output_dims)
            

        )
        

    def forward(self,image):
        inte=torch.flatten(image,start_dim=1)
        
        result=self.n1(inte)
        
        return result
        


    def training_step(self,batch):
        image,label=batch
        result=self(image)
        loss=self.loss(result,label)
        
        return loss
    
    def validation_step(self,batch):
        print('wait')
        image,label=batch
        result=self(image)
        loss=self.loss(result,label)
        print("val loss for this batch ",loss.detach())
        return loss.detach()
    
    def validation_epoch_end(self, outputs):
        print("calculating mean val loss")
        
        epoch_loss = torch.stack(outputs).mean()   # Combine losses
        print("calculated mean val loss",epoch_loss.item())
        return epoch_loss.item()