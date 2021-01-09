import numpy as np
import cv2
import torch
from CycleGAN.models import CycleGAN
import sys

# Usage:
# python extract_gen [path to model checkpoint] [name for new generator checkpoint]
def extract_generators(model, dest = "./"):
    gen = model.models["G_AB"]
    torch.save(gen.state_dict(), dest + "toB.pt")
    gen = model.models["G_BA"]
    torch.save(gen.state_dict(), dest + "toA.pt")

def extract_from_path(model_path, dest = "./"):
    with torch.no_grad():
        model = CycleGAN()
        model.load_checkpoint_from_path(model_path)
        extract_generators(model, dest)
    
if __name__ == "__main__":
    extract_from_path(sys.argv[1])

