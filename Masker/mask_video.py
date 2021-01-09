import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np

from .model import BASNet
from .cvInterface import toTensor


# Normalizes array values to [0,1] interval
def normalize_prediction(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn
  
# Given a frame, runs said frame through masking model
# Returns mask as a boolean array
def mask_frame(model, frame, CUDA = True):
	frame = toTensor(frame, CUDA = CUDA)
	d1, _, _, _, _, _, _, _ = model(frame)
	pred = d1[:,0,:,:]
	pred = normalize_prediction(pred)
	pred = pred.squeeze().cpu().detach().numpy()
	return np.where(pred > 0.5, True, False).astype(np.bool)

def mask(video_path, result_path):
	# Open video and get some info on it
	# This assumes we will crop bottom right corner
	vid_in = cv2.VideoCapture(video_path)
	width = int(vid_in.get(3))
	height = int(vid_in.get(4))
	center_x = width // 2
	center_y = height // 2
	num_frames = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
	
	final_mask = np.zeros((num_frames, 256, 256), dtype = np.bool)
	frame_index = 0
	
	# Load model
	model_dir = "./basnet.pth"
	model = BASNet(3, 1)
	model.load_state_dict(torch.load(model_dir))
	
	model.cuda()
	#model.half() # This doesn't work (I think it's because of sigmoid)
	
	with torch.no_grad():
		while(vid_in.isOpened()):
			ret, frame = vid_in.read()
			if frame is None: break
			frame = frame[center_y:, center_x:] # This is the actual crop
			final_mask[frame_index] = mask_frame(model, frame)
			frame_index += 1
	
	np.save(result_path, final_mask)
        
if __name__ == "__main__":
        found = False
        try:
                file = sys.argv[1]
                found = True
        except:
                print("Invalid input")
        if found:
                name, ext = os.path.splitext(file)
                if(ext != ".mp4"):
                        print("Not mp4 files")
                else:
                        mask(file, name+" masked.mp4")
