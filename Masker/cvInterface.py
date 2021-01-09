import cv2
import torch
import numpy as np
import torch.nn.functional as F


# Does some back and forth between pytorch and opencv

# Assuming y is BGR numpy array,
# Shows it on screen 
def showFrame(y):
    cv2.imshow("Test", y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Converts video frame (assuming BGR array)
# into tensor for model input
def toTensor(frame, HEIGHT = 256, WIDTH = 256, HALF = False, CUDA = True):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(frame)
    x = x.unsqueeze(0)
    if CUDA: x = x.cuda() # GPU
    if HALF: x = x.half() # Half precision
    x = x/255
    x = x.permute(0,3,1,2) # NHWC to NCHW
    x = F.interpolate(x, [HEIGHT, WIDTH]) # Resize to (256, 256)
    return x

# Converts tensor into BGR video frame
def toFrame(x):
    frame = x.squeeze().permute(1,2,0).cpu().numpy()*255
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame