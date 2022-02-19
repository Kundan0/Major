
import cv2
import os
import numpy as np
from CalculateDepth import ret_depth,load_ADA
from CalculateOF import ret_of,load_RAFT
from tracker import ret_bbox
from ClassModel import myModel
import matplotlib.image as mpimg
import torchvision.transforms as tr
import torch
from sys import exit
from time import time
import matplotlib.image as mpimg
import PIL.Image as Image

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#depth
pretrained_depth= "./AdaBins_kitti.pt"
depth_model=load_ADA(pretrained_depth,device)

#optical_flow
weights_path = './OpticalFlow/archive/raft-sintel.pth'
of_model=load_RAFT(weights_path,device)

#tracker
tracker_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#Vehicle velocity and position estimation model
print("building type1")
type1_model=myModel('1',os.path.join(".","State","trained1"))
#type1_model.load_model()
print("building type2")
type2_model=myModel('2',os.path.join(".","State","trained2"))
#type2_model.load_model()
print("building type3")
type3_model=myModel('3',os.path.join(".","State","trained3"))
#type3_model.load_model()
print("building type4")
type4_model=myModel('4',"./trained4").to(device)
type4_model.load_model()
print("reading video information")






#video 
file_path='./imgs_video.avi'
output_file_name='./output.avi'
video=cv2.VideoCapture(file_path)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
frame_size=(frame_width,frame_height)
FPS=video.get(cv2.CAP_PROP_FPS)
video_writer= cv2.VideoWriter(output_file_name, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         FPS, frame_size)
ROI=(frame_width/2,frame_width,frame_height/3,frame_height) # Region Of Interest (left,right,top,bottom)
#bounding box
RECT_COLOR_BBOX=(0,255,255) #yellow #BGR
RECT_COLOR_ROI=(0,255,0)
TEXT_COLOR=(0,255,255)



#Checking if the video is loaded successfully

if (video.isOpened()):
    print("Successfully Opened Video :) ")
else:
    print("Error Opening Video")
    exit()

i=1
frames=[] # to store two frames for sending to depth ,tracker and of network
JUMP=round(FPS*0.1-1)
DIFF=10
size=(128,72)
HEIGHT_RATIO=int(frame_width/size[0])
WIDTH_RATIO=int(frame_width/size[1])
results=[]

print("fps",FPS)
print("jump ",JUMP)
while(video.isOpened()):
    
    success,frame=video.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if success==False:
        print("Couldn't read the frame ")
        exit()
    if (i+DIFF-1) % DIFF==0 : # we want to skip calculating for each successive 
                                #frame, so DIFF defines the amount of frames after which we want to calculate again
                                
        frames.append(frame) 
        CUR_INDEX=i
        print("first calculated frame ",i)
    if (i==CUR_INDEX+JUMP): # JUMP is for managing frame rate (we have trained for 20 fps)
        print("cur index inside second",CUR_INDEX)
        print("second calculated frame ",i)
        frames.append(frame)
        #depth processing
        
        depth0=ret_depth(frames[0],depth_model,device) # torch.size([1,240,320])
        depth1=ret_depth(frames[1],depth_model,device)
        depth0=(depth0-torch.min(depth0))/(torch.max(depth0)-torch.min(depth0))
        depth1=(depth1-torch.min(depth1))/(torch.max(depth1)-torch.min(depth1))
       
        # print('.....depth..... ')
        # print("mean value of tensor before reading the image ",torch.mean((depth0)))
        # print("std ",torch.std((depth0)))
        # print("maximum value ",torch.max(depth0))
        # print("minimum value ",torch.min(depth0))
        # print("tensor ",depth0)
        

        # mpimg.imsave('./depth0.jpg',depth0.squeeze(0).detach().cpu(),cmap='gray')
        # mpimg.imsave('./of1.jpg',of1.cpu()/255.,cmap='gray')
        # depthimg=tr.ToTensor()(Image.open('./depth0.jpg'))
        # print("mean value of tensor after reading the image ",torch.mean(depthimg))
        # print("std ",torch.std(depthimg))
        # print("after reading from the image ",depthimg)
        depth0=tr.functional.crop(depth0,left=0,top=50,height=190,width=320)
        depth1=tr.functional.crop(depth1,left=0,top=50,height=190,width=320)
        
        depth0=tr.functional.resize(depth0,(72,128)).to(device=device) # size(1,72,128)
        depth1=tr.functional.resize(depth1,(72,128)).to(device=device)
        

        print("original depth 0",depth0)
        print("original depth 1",depth1)

        
        #of processing 
        start=time()
        of0,of1=ret_of(frames[0],frames[1],of_model,device)
        # of0=torch.tensor(of0)
        # of1=torch.tensor(of1)
        of0=(of0-torch.min(of0))/(torch.max(of0)-torch.min(of0))
        of1=(of1-torch.min(of1))/(torch.max(of1)-torch.min(of1))
        print("mean value of tensor before reading the image ",torch.mean(torch.tensor(of0)))
        print("std ",torch.std(torch.tensor(of0)))
        print("maximum value ",torch.max(of0))
        print("minimum value ",torch.min(of0))
        # mpimg.imsave('./of0.jpg',of0.cpu(),cmap='gray')
        # mpimg.imsave('./of1.jpg',of1.cpu(),cmap='gray')
        # ofimg=tr.ToTensor()(Image.open('./of0.jpg'))
        # print("mean value of tensor after reading the image ",torch.mean(ofimg))
        # print("std ",torch.std(ofimg))
        of0=of0.unsqueeze(0)
        of1=of1.unsqueeze(0)
        
        # print("read from image ",ofimg)
        print("of shape",of0.shape)
        print(f"for of took {time()-start}")

        of0=tr.functional.crop(of0,left=0,top=50,height=520,width=1280)
        of1=tr.functional.crop(of1,left=0,top=50,height=520,width=1280)
        
        of0=tr.functional.resize(of0,(72,128)).to(device=device) # size(1,72,128)
        of1=tr.functional.resize(of1,(72,128)).to(device=device)
        print("original of 0",of0)
        print("original of 1",of1)
        # ofimage0=np.transpose(of0.cpu().numpy(),(1,2,0))
        # ofimage1=np.transpose(of1.cpu().numpy(),(1,2,0))
        

        # vehicle identification
        
        bbox=ret_bbox([frames[0]],tracker_model,ROI,0.5)[0]
        num_vehicles=len(bbox)
        print(f"found {num_vehicles} no of vehicles")
        results=[]
        for vehicle in bbox:
            
            left=int(vehicle["left"])
            right=int(vehicle["right"])
            top=int(vehicle["top"])
            bottom=int(vehicle["bottom"])
            area=(right-left)*(bottom-top)
            
            left_bbox=int(left/WIDTH_RATIO)
            right_bbox=int(right/WIDTH_RATIO)
            top_bbox=int(top/HEIGHT_RATIO)
            bottom_bbox=int(bottom/HEIGHT_RATIO)
            DELTA=10
            HALF_DELTA=int(DELTA/2)
            
            bbox_mask=torch.zeros(size[::-1],device=device)
            
            
            
            try:
                bbox_size=(bottom_bbox-top_bbox+DELTA,right_bbox-left_bbox+DELTA)
                ones=torch.ones(bbox_size,device=device)
                bbox_mask[top_bbox-HALF_DELTA:bottom_bbox+HALF_DELTA,left_bbox-HALF_DELTA:right_bbox+HALF_DELTA]=ones
            except:
                bbox_size=(bottom_bbox-top_bbox,right_bbox-left_bbox)
                ones=torch.ones(bbox_size,device=device)
                bbox_mask[top_bbox:bottom_bbox,left_bbox:right_bbox]=ones
        
            
            #cv2.imwrite('bbox',bbox_mask.cpu().numpy())
            inter_tensor=torch.cat((depth0,of0,of1,depth1,bbox_mask.unsqueeze(0)),dim=0).permute(0,2,1).unsqueeze(0)
            depth0=None
            depth1=None
            of0=None
            of1=None
            bbox_mask=None
            print('in readvideo before sending to model',inter_tensor.shape)
            print("area of vehicle",area)
            if area<2500:
                result=type1_model(inter_tensor)
                
            elif area>=2500 and area<5000:
                result=type2_model(inter_tensor)
                
            elif area>=5000 and area < 7500:
                result=type3_model(inter_tensor)
                
            elif area >7500:
                result=type4_model(inter_tensor)
            
            
            inter_tensor=None
            
            
            result=result.squeeze(0)
            print("result",result)
            results.append((result,left,right,top,bottom))
            result=None
            frames=[]
            
        
    for value in results:
        
        result,left,right,top,bottom=value
        
        velocity_f,velocity_s,position_f,position_s=result[0],result[1],result[2],result[3]
        text_left_bottom=(left,bottom)
        cv2.rectangle(frame,(left,top),(right,bottom),color=RECT_COLOR_BBOX,thickness=2)
        cv2.putText(frame,"V "+str((round(velocity_f.item(),2),round(velocity_s.item(),2))),(left,top-40),cv2.FONT_HERSHEY_SIMPLEX,0.4,TEXT_COLOR,1,cv2.LINE_AA)
        cv2.putText(frame,"P "+str((round(position_f.item(),2),round(position_s.item(),2))),(left,top-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,TEXT_COLOR,1,cv2.LINE_AA)
    cv2.rectangle(frame,(600,360),(1270,715),color=RECT_COLOR_ROI,thickness=2)
    i+=1
    video_writer.write(frame)

        

        

        

    
    

    
    

    
video.release()
video_writer.release()
