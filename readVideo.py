import cv2
import numpy as np
from CalculateDepth import ret_depth,load_ADA
from CalculateOF import ret_of,load_RAFT
from tracker import ret_bbox

from AdaBins import model_io
from AdaBins.models import UnetAdaptiveBins
import matplotlib.image as mpimg
import torchvision.transforms as tr
import torch

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#depth
pretrained_depth= "/home/kundan/Documents/Major/AdaBins/pretrained/AdaBins_kitti.pt"
depth_model=load_ADA(pretrained_depth,device)

#optical_flow
weights_path = './archive/raft-sintel.pth'
of_model=load_RAFT(weights_path,device)

#tracker
tracker_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#Vehicle velocity and position estimation model
type1_model=

#video 
file_path='./download.mp4'
video=cv2.VideoCapture(file_path)
FPS=video.get(cv2.CAP_PROP_FPS)

#Checking if the video is loaded successfully

if (video.isOpened()):
    print("Successfully Opened Video :) ")
else:
    print("Error Opening Video")
    exit()

i=1
frames=[] # two store two frames for sending to depth ,tracker and of network

while(video.isOpened()):
    ret,frame=video.read()
    if ret==False:
        print("couldn't read the frame ")
        exit()
    if i%2 ==0 :
        depth=ret_depth(frames,depth_model) #it is tuple returned for each frame
        of=ret_of(frame[0],frame[1],of_model,device)
        bbox_track=ret_bbox([frames[0]],tracker_model)[0] # dictionary with {left,right,bottom,top}

        pass

        

    
    

    frames.append(frame)
    i+=1
    
    depth0=depth[0].squeeeze(0)
    depth1=depth[1].squeeze(0)
    
    depth=tr.functional.crop(depth,top=50,left=0,height=(240-50),width=320)
    
    
    print('saved')
    
video.release()
cv2.destroyAllWindows()

# import numpy as np
# import cv2
# image=cv2.imread('./sample.jpg')
# print(isinstance(image,np.ndarray))
#mpimg.imsave('./download_frames/fromreadvideo'+str(i)+'.png',depth,cmap='gray')
#depth=np.transpose(depth,(1,2,0))