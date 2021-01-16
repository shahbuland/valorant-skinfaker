# Valorant SkinFaker

Both convert.py and trainer.py can be run and have a basic UI  
  
**Setup**  

- put model for masker ("basnet.pth") in "Masker" folder
- You can download all models I made in my video as well as the masking model from this [google drive](https://drive.google.com/file/d/1I2889hwmhZXWmfPaaW0L6ldmRU2LjUfN).

**Converting**


- Open the source video (.mp4 or .avi), and generator you want to use (.pt file)
- Note the 4 check boxes on the side
- If stitch is off, the resulting video will just be the bottom right of the screen. If it's on, the program will fit the resulting video (the corner) back into the original frame.
- If mask is on, the program will use another neural network to attempt to extract the area of the frame in which the gun is and only apply the generator to said area of the frame. This should often eliminate the box around the corner of the screen but has varying success.
- IMPORTANT: I STRONGLY suggest you enable half precision and try to get CUDA working, as it is literally 100s of times slower without it
- Timings on my system: 
  - No stitch, no mask, no half, no CUDA (i7 7700k): 676ms per frame = 1.5 FPS
  - No stitch, no mask, half, CUDA (RTX 3090):  17.4ms per frame = 58 FPS
  - Stitch, no mask, half, CUDA: 25.6ms = 40 FPS
  - Stitch, mask, half, CUDA: 61ms = 16 FPS
- Memory Impact:
  - No boxes checked: 200MB RAM/ 0MB VRAM
  - Just CUDA and half: 2.4GB RAM/ 1.4GB VRAM
  - Everything: 2.5GB RAM/ 2.1GB VRAM

**Training**

- Create experiment and title it
- Navigate to experiment folder 
- Place videos of weapon/skin A into folder "source_A", videos of weapon/skin B into folder "source_B"
- Press "Convert Videos" to turn videos into data model can train with (if data npy files already exist you can delete source videos, but ensure they are titled "A_data.npy" and "B_data.npy" respectively)
  - Set frame skip accordingly. If it was 10, every 10th frame would be taken from both videos. I'd recommend shooting for around 2000 frames, though more is always better.
- Parameters you can specify:
  - Log Interval: How often to display a progress report message in console
  - Sample Interval: How often to draw samples to "samples" folder
  - Checkpoint Interval: How often to save a checkpoint to "checkpoints" folder
- The checkpoint prompt has two purposes:
  1. If you want to continue training from a checkpoint, select "use checkpoint" and provide a valid checkpoint (.pt file)
  2. If you're done and satisfied with the final model, press "extract generator" to get two .pt files from the checkpoint. "toA.pt" will be the generator that converts from weapon/skin B to weapon/skinA and "toB.pt" will be the opposite. For example, if A was the base weapon, and B was the skin, you want "toB.pt" to deepfake the skin. This can then be used with the converting program.
- Ctrl-C on the python window to stop training
  - You should stop early based on what you see in samples. I never needed to do full 200 epochs. If you're keeping track of the samples folder, where the numbers represent the number of parameter updates, I generally only had to go to around 20,000 before getting results.

**Python Requirements**

- Python 3.6
- Pytorch
- OpenCV
- Numpy
- PyQt5

I used miniconda and I think that's probably the simplest way to get this working on windows.  You can google how to install each of the above packages in miniconda, each is a single line.
