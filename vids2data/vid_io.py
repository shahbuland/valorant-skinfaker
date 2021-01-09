import cv2
import gzip
import numpy as np

CROP = True
OUT_H = 256
OUT_W = 256


def vid2arr(sources, destination, frame_skip = 10):
    A = None
    
    A_index = 0
    for source in sources:
        print("Reading " + source + "...")
        vid_in = cv2.VideoCapture(source)
        # Get vid info
        width = int(vid_in.get(3))
        height = int(vid_in.get(4))
        center_x = width // 2
        center_y = height // 2
        corner_width = width - center_x
        corner_height = height - center_y
        total_frames = (int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT)) // frame_skip) + 1

        frame_index = 0
        
        if A is None:
            A = np.zeros((total_frames, OUT_H, OUT_W, 3), dtype = np.uint8)
        else:
            A = np.resize(A, (A.shape[0] + total_frames, OUT_H, OUT_W, 3))
       
        
        while(vid_in.isOpened()):
            ret, frame = vid_in.read()
            if frame is None: break
            
            if frame_index % frame_skip != 0:
                frame_index += 1
                continue
            
            if CROP: frame = frame[center_y:, center_x:]
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation = cv2.INTER_AREA)
            
            if frame_index % frame_skip == 0:
                A[A_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                A_index += 1
                
            frame_index += 1
        
        vid_in.release()
    
    np.save(destination, A)
    
