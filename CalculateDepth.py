from time import time
from AdaBins import model_io
from AdaBins.models import UnetAdaptiveBins
import os
from PIL import Image
import torchvision.transforms as transform
import torch
unloader = transform.ToPILImage()
loader = transform.ToTensor()  

def image_loader(imgs):
    size=640, 480
    inter_tensor=None
    for i,img in enumerate(imgs):
        print(i)
        image=loader(Image.open(img).convert('RGB').resize(size, Image.ANTIALIAS))
        if inter_tensor!=None:
            inter_tensor=torch.cat((inter_tensor,image.unsqueeze(0)),dim=0)
        else:
            inter_tensor=image.unsqueeze(0)
    print(inter_tensor.size())
    return inter_tensor
    
    
def ret_depth(batch,pretrained_path):
    imgs=image_loader(batch)
    MIN_DEPTH = 1e-3
    #MAX_DEPTH_NYU = 10
    MAX_DEPTH_KITTI = 80
    N_BINS = 256 
    model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
    model, _, _ = model_io.load_checkpoint(pretrained_path, model)
    start=time()
    _,depth=model(imgs)
    print(f"took {time()-start}") 
    #print(depth.squeeze(0).squeeze(0).size())
    print(depth.size())

if __name__=="__main__":
    pretrained = "/home/kundan/Documents/Major/AdaBins/pretrained/AdaBins_kitti.pt"
    
    imgs=['./sample.jpg','./002.jpg']
    ret_depth(imgs,pretrained)

    
    
    
