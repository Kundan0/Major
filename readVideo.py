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
type4_model=myModel('4',os.path.join(".","State","trained4"))
#type4_model.load_model()
print("reading video information")


#video 
file_path='./download.mp4'
output_file_name='./output.avi'
video=cv2.VideoCapture(file_path)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
frame_size=(frame_width,frame_height)
FPS=video.get(cv2.CAP_PROP_FPS)
video_writer= cv2.VideoWriter(output_file_name, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         FPS, frame_size)

#bounding box
RECT_COLOR=(0,255,255) #yellow
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
DIFF=6
size=(128,72)
HEIGHT_RATIO=int(frame_width/size[0])
WIDTH_RATIO=int(frame_width/size[1])
results=[]
while(video.isOpened()):
    print(" idx ",i)
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
        print("calculating depth")
        print("size of input ",frame.shape)
        depth0=ret_depth([frame[0]],depth_model)

        print(depth0.size())
        
        #depth0,depth1=ret_depth(frames,depth_model) #it is tuple returned for each frame
        # depth0,depth1=depth0.squeeeze(0),depth1.squeeze(0)# removing the channel layer as it is a single channel
        # depth0,depth1=tr.functional.crop(depth0,top=50,left=0,height=190),tr.functional.crop(depth1,top=50,left=0,height=190)# cropping top portion
        # depth0,depth1=torch.from_numpy(cv2.resize(depth0.numpy(),size)).to(torch.float32),torch.from_numpy(cv2.resize(depth0.numpy(),size)).to(torch.float32)
        # 
        print("depth calculated")
        print("second depth")
        depth1=ret_depth([frame[1]],depth_model)
        print("calculated seconnd depth")
        #of processing 
        print("calculating of")
        of0,of1=ret_of(frames[0],frames[1],of_model,device)
        of0,of1=of0[50:,:],of1[50:,:]
        of0,of1=cv2.resize(of0,size),cv2.resize(of1,size)
        of0,of1=torch.from_numpy(of0),torch.from_numpy(of1)
        print("calculated of ")
        # vehicle identification
        print("tracking")
        bbox=ret_bbox([frames[0]],tracker_model,0.66)[0]
        print("tracked")
        num_vehicles=len(bbox)
        results=[]
        for vehicle in bbox:
            
            left=int(bbox["left"])
            right=int(bbox["right"])
            top=int(bbox["top"])
            bottom=int(bbox["bottom"])
            area=(right-left)*(bottom-top)
            
            left_bbox=int(left/WIDTH_RATIO)
            right_bbox=int(right/WIDTH_RATIO)
            top_bbox=int(top/HEIGHT_RATIO)
            bottom_bbox=int(bottom/HEIGHT_RATIO)
            DELTA=10
            HALF_DELTA=int(DELTA/2)
            
            bbox_mask=torch.zeros(size[::-1])
            
            
            
            try:
                bbox_size=(bottom_bbox-top_bbox+DELTA,right_bbox-left_bbox+DELTA)
                ones=torch.ones(bbox_size)
                bbox_mask[top_bbox-HALF_DELTA:bottom_bbox+HALF_DELTA,left_bbox-HALF_DELTA:right_bbox+HALF_DELTA]=ones
            except:
                bbox_size=(bottom_bbox-top_bbox,right_bbox-left_bbox)
                ones=torch.ones(bbox_size)
                bbox_mask[top_bbox:bottom_bbox,left_bbox:right_bbox]=ones
            
            inter_tensor=torch.cat((depth0.unsqueeze(0),of0.unsqueeze(0),of1.unsqueeze(0),depth1.unsqueeze(0),bbox_mask.unsqueeze(0)),dim=0).unsqueeze(0)
            if area<2500:
                result=type1_model(inter_tensor)
            elif area>=2500 and area<5000:
                result=type2_model(inter_tensor)
            elif area>=5000 and area < 7500:
                result=type3_model(inter_tensor)
            elif area >7500:
                result=type4_model(inter_tensor)
            
            #convert result to tuple using torch.split(result,1)

            results.append((result,left,right,top,bottom))
            frames=[]
    for value in results:
        result,left,right,top,bottom=value
        velocity_f,velocity_s,position_f,position_s=result
        text_left_bottom=(left,bottom)
        cv2.rectangle(frame,(left-5,top-5),(right+5,bottom+5),color=RECT_COLOR,thickness=2)
        cv2.putText(frame,"Velocity "+str((velocity_f,velocity_s)),(left,top-40),cv2.FONT_HERSHEY_SIMPLEX,0.4,TEXT_COLOR,1,cv2.LINE_AA)
        cv2.putText(frame,"Position "+str((velocity_f,velocity_s)),(left,top-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,TEXT_COLOR,1,cv2.LINE_AA)
        
    cv2.imwrite('./output.jpg',frame)
    video_writer.write(frame)

        

        

        

    
    

    
    i+=1

    
video.release()
video_writer.release()
