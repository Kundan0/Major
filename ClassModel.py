import torch
import torch.nn as nn
class myModel(nn.Module):
    #hl_dims are hidden layer dimensions
    def __init__(self,checkpoint_pth,size=(128,72),hl_dim1=32,hl_dim2=256,hl_dim3=128,output_dims=4,loss=nn.MSELoss()):
        super().__init__()
        
        self.output_dims=output_dims
        self.size=size
        self.hl_dim1=hl_dim1
        self.hl_dim2=hl_dim2
        self.hl_dim3=hl_dim3
        self.checkpoint_path=checkpoint_pth
        # self.learning_rate=learning_rate
        # self.optimizer=optim(self.parameters(),self.learning_rate)
        self.loss=loss
        self.n1=nn.Sequential(
            
            nn.Linear(5*self.size[0]*self.size[1],self.hl_dim1),
            nn.Linear(self.hl_dim1,self.output_dims)
            

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

    def save_model(self,ep,optimizer):
        checkpoint={
            'epoch':ep,
            'optimizer':optimizer.state_dict(),
            'model':self.state_dict(),
        }
        torch.save(checkpoint,self.checkpoint_path)
        
    def load_model(self,optimizer):
        checkpoint=torch.load(self.checkpoint_path)
        self.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer,checkpoint['epoch']