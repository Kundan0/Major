from re import I
import cv2
import os
import numpy as np
from CalculateDepth import ret_depth,load_ADA
from CalculateOF import ret_of,load_RAFT
from tracker import ret_bbox
from ClassModel import myModel
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

type1_model=myModel('1',os.path.join(".","State","trained1"))
type1_model.load_model()
type2_model=myModel('2',os.path.join(".","State","trained2"))
type2_model.load_model()
type3_model=myModel('3',os.path.join(".","State","trained3"))
type3_model.load_model()
type4_model=myModel('4',os.path.join(".","State","trained4"))
type4_model.load_model()



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
JUMP=round(FPS*0.1-1)
DIFF=6
size=(128,72)
HEIGHT_RATIO=10
WIDTH_RATIO=10
while(video.isOpened()):
    success,frame=video.read()
    if success==False:
        print("Couldn't read the frame ")
        exit()
    if (i+DIFF-1) % DIFF==0 : # we want to skip calculating for each successive 
                                #frame, so DIFF defines the amount of frames after which we want to calculate again
                                
        frames.append(frame) 
        CUR_INDEX=i
    if (i==CUR_INDEX+JUMP): # JUMP is for managing frame rate (we have trained for 20 fps)
        frames.append(frame)
        #depth processing
        depth0,depth1=ret_depth(frames,depth_model) #it is tuple returned for each frame
        depth0,depth1=depth0.squeeeze(0),depth1.squeeze(0)# removing the channel layer as it is a single channel
        depth0,depth1=tr.functional.crop(depth0,top=50,left=0,height=190),tr.functional.crop(depth1,top=50,left=0,height=190)# cropping top portion
        depth0,depth1=torch.from_numpy(cv2.resize(depth0.numpy(),size)).to(torch.float32),torch.from_numpy(cv2.resize(depth0.numpy(),size)).to(torch.float32)

        #of processing 
        of0,of1=ret_of(frames[0],frames[1],of_model,device)
        of0,of1=of0[50:,:],of1[50:,:]
        of0,of1=cv2.resize(of0,(128,72)),cv2.resize(of1,(128,72))
        of0,of1=torch.from_numpy(of0),torch.from_numpy(of1)

        # vehicle identification
        bbox=ret_bbox([frames[0]],tracker_model,0.6)[0]
        left=int(bbox["left"]/WIDTH_RATIO)
        right=int(bbox["right"]/WIDTH_RATIO)
        top=int(bbox["top"]/HEIGHT_RATIO)
        bottom=int(bbox["bottom"]/HEIGHT_RATIO)

        DELTA=10
        HALF_DELTA=int(DELTA/2)
        
        bbox_mask=torch.zeros(size[::-1])
        
        
        
        try:
            bbox_size=(bottom-top+DELTA,right-left+DELTA)
            ones=torch.ones(bbox_size)
            bbox_mask[top-HALF_DELTA:bottom+HALF_DELTA,left-HALF_DELTA:right+HALF_DELTA]=ones
        except:
            bbox_size=(bottom-top,right-left)
            ones=torch.ones(bbox_size)
            bbox_mask[top:bottom,left:right]=ones
        
        inter_tensor=torch.cat((depth0.unsqueeze(0),of0.unsqueeze(0),of1.unsqueeze(0),depth1.unsqueeze(0),bbox_mask.unsqueeze(0)),dim=0)
        

        frames=[]

        

        

    
    

    frames.append(frame)
    i+=1
    
    
    
    # depth=tr.functional.crop(depth,top=50,left=0,height=(240-50),width=320)
    
    
    # print('saved')
    
video.release()
cv2.destroyAllWindows()

# import numpy as np
# import cv2
# image=cv2.imread('./sample.jpg')
# print(isinstance(image,np.ndarray))
#mpimg.imsave('./download_frames/fromreadvideo'+str(i)+'.png',depth,cmap='gray')
#depth=np.transpose(depth,(1,2,0))






# of=ret_of(frame[0],frame[1],of_model,device)
# bbox_track=ret_bbox([frames[0]],tracker_model)[0] # dictionary with {left,right,bottom,top}

# pass