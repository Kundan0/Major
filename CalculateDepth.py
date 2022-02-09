from time import time
import sys
import os
sys.path.append(os.path.abspath('./AdaBins'))

from AdaBins import model_io
from AdaBins.models import UnetAdaptiveBins

from PIL import Image
import torchvision.transforms as transform
import torch
import numpy as np
import matplotlib.image as mpimg
import cv2
from torchvision.transforms import ToPILImage,ToTensor
unloader = ToPILImage()
loader = ToTensor()  

def image_loader(imgs):
    size=640, 480
    inter_tensor=None
    for i,img in enumerate(imgs):
        if (isinstance(img,np.ndarray)):
            
             
            img=cv2.resize(img,(640,480))
        
            img=torch.from_numpy((img/255)).to(torch.float32).permute(2,0,1)
            
        elif (isinstance(img,str)):
            img=loader(Image.open(img).convert('RGB').resize(size, Image.ANTIALIAS))
            
        if inter_tensor!=None:
            inter_tensor=torch.cat((inter_tensor,img.unsqueeze(0)),dim=0)
        else:
            inter_tensor=img.unsqueeze(0)
    return inter_tensor
    
    
def ret_depth(batch,model):
    imgs=image_loader(batch)            
    
    
    start=time()
    _,depth=model(imgs)
    print(f"took {time()-start}") 
    #print(depth.squeeze(0).squeeze(0).size())
    return depth.detach().squeeze(0).squeeze(0)

def load_ADA(pretrained,device):
    MIN_DEPTH = 1e-3
    #MAX_DEPTH_NYU = 10
    MAX_DEPTH_KITTI = 80
    N_BINS = 256 
    
    model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
    model, _, _ = model_io.load_checkpoint(pretrained, model)
    return model.to(device)


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained = "/home/kundan/Documents/Major/AdaBins/pretrained/AdaBins_kitti.pt"
    model=load_ADA(pretrained,device)
    imgs=['./sample2.png']

    depth=ret_depth(imgs,model) # now depth is a tuple 
    # print(depth.detach().size())
    # mpimg.imsave('./depth2.jpg',depth.detach(),cmap='gray')

    
    
    
