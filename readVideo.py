import cv2
import numpy as np
from CalculateDepth import ret_depth
from AdaBins import model_io
from AdaBins.models import UnetAdaptiveBins
import matplotlib.image as mpimg
import torchvision.transforms as tr
#depth
MIN_DEPTH = 1e-3
MAX_DEPTH_KITTI = 80
N_BINS = 256 
pretrained = "/home/kundan/Documents/Major/AdaBins/pretrained/AdaBins_kitti.pt"
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
model, _, _ = model_io.load_checkpoint(pretrained, model)

video=cv2.VideoCapture('./download.mp4')

if (video.isOpened()):
    print("video opened ")
else:
    print("error")
    exit()
fps=video.get(cv2.CAP_PROP_FPS)
i=1
fps
while(video.isOpened()):
    if i%3 !=0 :
        

    ret,frame=video.read()
    if ret==False:
        print("couldn't read the frame ")
        exit()

    frames.append(frame)
    i+=1
    depth=ret_depth(frames,model)
    depth0=depth[0].squeeeze(0)
    depth1=depth[1].squeeze(0)
    
    depth=tr.functional.crop(depth,top=50,left=0,height=(240-50),width=320)
    
    mpimg.imsave('./download_frames/fromreadvideo'+str(i)+'.png',depth,cmap='gray')
    
    print('saved')
    #depth=np.transpose(depth,(1,2,0))
video.release()
cv2.destroyAllWindows()

# import numpy as np
# import cv2
# image=cv2.imread('./sample.jpg')
# print(isinstance(image,np.ndarray))
