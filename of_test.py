import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from glob import glob
from PIL import Image

sys.path.append(os.path.abspath('./archive'))

from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
from raft.config import RAFTConfig

config = RAFTConfig(
    dropout=0,
    alternate_corr=False,
    small=False,
    mixed_precision=False
)

model = RAFT(config)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

weights_path = './archive/raft-sintel.pth'
#weights_path = '.archive/raft-things.pth'

ckpt = torch.load(weights_path, map_location=device)
model.to(device)
model.load_state_dict(ckpt)


def load_image(imfile, device):
    if (isinstance(imfile,np.ndarray)):
         img = imfile
    elif (isinstance(imfile,str)):
        img=np.array(Image.open(imfile)).astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def viz(img1, img2, flo):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    ax1.set_title('input image1')
    ax1.imshow(img1.astype(int))
    ax2.set_title('input image2')
    ax2.imshow(img2.astype(int))
    ax3.set_title('estimated optical flow')
    ax3.imshow(flo)
    plt.show()



if __name__=="__main__":

    odir="./Output"

    file1="./Input/001.jpg" #path1 here 
    file2="./Input/002.jpg" #path2 here

    try:
        os.mkdir(odir)
    except:
        pass
    try:
        odir1=os.path.join(odir,'test')
        os.mkdir(odir1)

    except:
        pass


    # file1=cv2.imread(file1)
    # file2=cv2.imread(file2)
    image1 = load_image(file1, device)
    image2 = load_image(file2, device)

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

    #viz(image1, image2, flow_up)
    flo=flow_up
    flo = flo[0].cpu().numpy()
    plt.axis("off")
    try:
        filename11=os.path.join(odir1,'a.png')
        plt.imsave(filename11,flo[0],cmap="gray")
    except:
        pass     
    try:
        filename22=os.path.join(odir1,'b.png')
        plt.imsave(filename22,flo[1],cmap="gray")
    except:
        pass
