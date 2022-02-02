from matplotlib.transforms import Bbox
import torch
from time import time
def ret_bbox(img_batch,CONFIDENCE_MINM=0.5):

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Inference
    start=time()
    results = model(img_batch)
    
    vehicles=results.xyxy[0]
    #print(results.pandas().xyxy[0])
    # predicted_num=len(len(results.xyxy[0]))
    # print(predicted_num)  # img1 predictions (tensor)
    valid_vehicle=[2,3,5,7]
    CONFIDENCE_MINM=0.50
    bbox=[]
    for image in img_batch:
        bbox_single_image=[]
        for vehicle in vehicles:
            if int(vehicle[5].item()) in valid_vehicle and vehicle[4].item()>CONFIDENCE_MINM:
                bbox_single_image.append({"left":int(vehicle[0].item()),"top":int(vehicle[1].item()),"right":int(vehicle[2].item()),"bottom":int(vehicle[3].item()),"confidence":vehicle[4].item()})
        bbox.append(bbox_single_image)
    return bbox

if __name__=="__main__":
    # Model
    
    # Images
    imgs = ['./veh.jpeg','./001.jpg']  # batch of images

    box=ret_bbox(imgs)
    print(box)

# we can send multiple images inside imgs variable , box[0] returns bbox for first image , box[1] for bbox for second image and so on 
# we will pass two images frame1 and frame2 to get two bboxes
# the box[0] is a dictionary with keys left,top,right,bottom and confidence
 