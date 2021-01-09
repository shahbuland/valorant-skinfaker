from CycleGAN.models import resGenerator
from Masker.cvInterface import showFrame, toTensor, toFrame
from clock import Clock
from Masker.model import BASNet
from Masker.mask_video import mask_frame

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import random
import sys
    
def convert(src_path, dest_path, model_path, do_stitch = True, do_mask = True, use_cuda = True, use_half = True):
    # Verify cuda is available
    if use_cuda:
        if not torch.cuda.is_available():
            print("Error: Trying to use CUDA when it is not available")
            print("Either something's wrong with pytorch or you don't have CUDA toolkit installed")
            
    # Load main model 
    generator = resGenerator()
    generator.load_state_dict(torch.load(model_path))
    if use_cuda: generator.cuda() # To GPU
    if use_half: generator.half() # Use half precision
    
    # Load clock if we want to time
    times = []
    timer = Clock()
    timer.hit()
    
    # Load masker is needed
    if do_mask:
        masker = BASNet(3, 1)
        masker.load_state_dict(torch.load("./Masker/basnet.pth"))
        if use_cuda: masker.cuda()
       
    # Video reader
    vid_in = cv2.VideoCapture(src_path)
    if vid_in.isOpened(): # Get information on video
        width = int(vid_in.get(3))
        height = int(vid_in.get(4))
        center_x = width // 2
        center_y = height // 2
        corner_width = width - center_x
        corner_height = height - center_y
        fps = vid_in.get(cv2.CAP_PROP_FPS)
    else:
        print("ERROR: Failed to read video")
        return
        
    # Video writer
    out_dim = (width, height) if do_stitch else (corner_width, corner_height)
    vid_out = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, out_dim)
    

    print("Ready up time: " + str(timer.hit()) + "ms")
        
    with torch.no_grad():
        while(vid_in.isOpened()):
            timer.hit()
            ret, frame = vid_in.read()
            if frame is None: break
            
            if do_stitch or do_mask:
                orig_frame = frame
            
            frame = frame[center_y:, center_x:]
            
            if do_mask: mask = mask_frame(masker, frame, CUDA = use_cuda)
            
            frame = toTensor(frame, HALF = use_half, CUDA = use_cuda)
            frame = toFrame(generator(frame))
            frame = cv2.resize(frame, (corner_width, corner_height), interpolation = cv2.INTER_CUBIC)
            
            if do_mask:
                mask = np.squeeze(mask)
                mask = np.expand_dims(mask, 2)
                mask = np.concatenate((mask, mask, mask), axis = 2).astype(np.uint8)
                mask = cv2.resize(mask, (corner_width, corner_height), interpolation = cv2.INTER_CUBIC)
                frame = frame * mask + (1 - mask) * orig_frame[center_y:, center_x:]
                
            if do_stitch:
                orig_frame[center_y:, center_x:] = frame
                frame = orig_frame
            
            vid_out.write(frame)
            times.append(timer.hit())
            
    vid_out.release()
    vid_in.release()
    
    avg_time = sum(times)/len(times)
    print("Average Time Per Frame: " + str(round(avg_time,1)) + "ms")
    print("Total Time: " + str(round(sum(times)/1000,1)) + "s")
            
