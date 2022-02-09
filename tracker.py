from matplotlib.transforms import Bbox
import torch
from time import time
import cv2
def ret_bbox(img_batch,model,CONFIDENCE_MINM=0.5):

    
    results = model(img_batch)
    vehicles=results.xyxy[0]
    valid_vehicle=[2,3,5,7]
    
    bbox=[]
    for image in img_batch:
        bbox_single_image=[]
        for vehicle in vehicles:
            if ( int(vehicle[5].item()) in valid_vehicle ) and vehicle[4].item()>CONFIDENCE_MINM and vehicle[0].item()>400:
                bbox_single_image.append({"left":int(vehicle[0].item()),"top":int(vehicle[1].item()),"right":int(vehicle[2].item()),"bottom":int(vehicle[3].item()),"confidence":vehicle[4].item()})
        bbox.append(bbox_single_image)
    return bbox

if __name__=="__main__":
    # Model

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Images
    imgs = ['./veh.jpg']  # batch of images
    
    #  opencv
    image=cv2.imread(imgs[0])
    box=ret_bbox(imgs,model,0.66)[0]
    for  i,vehicle in enumerate(box):
        left=vehicle["left"]
        right=vehicle["right"]
        bottom=vehicle["bottom"]
        top=vehicle["top"]
        color=(0,255,255)
        cv2.rectangle(image,(left,top),(right,bottom),color=color,thickness=2)
        
        if i==0:
            cv2.putText(image,"Velocity "+str((-2.114,-0.18)),(left,top-40),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
            cv2.putText(image,"Position "+str((7.114,5.11)),(left,top-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
        else:
            cv2.putText(image,"Velocity "+str((0.114,-0.01)),(left,top-40),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
            cv2.putText(image,"Position "+str((11.114,2.14)),(left,top-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1,cv2.LINE_AA)
        

        cv2.imwrite("./tracked.png",image)
    print(box)

# we can send multiple images inside imgs variable , box[0] returns bbox for first image , box[1] for bbox for second image and so on 
# we will pass two images frame1 and frame2 to get two bboxes
# the box[0] is a dictionary with keys left,top,right,bottom and confidence
 
