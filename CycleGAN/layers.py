import torch
from torch import nn
from torch.nn import functional as F

# ========= BASIC LAYERS ========

# Decoding Block (Deconv)
class DecodeBlock(nn.Module):
        def __init__(self,fi,fo,k=4,s=2,p=1, op = 0,useBN = True,Act=F.relu):
                super(DecodeBlock,self).__init__()
		
                self.conv = nn.ConvTranspose2d(fi,fo,k,s,p,op, bias = False)
                self.bn = nn.BatchNorm2d(fo) if useBN else None
                self.Act = Act
		
        def forward(self,x):
                #x = F.interpolate(x,scale_factor=(2,2))
                x = self.conv(x)
                if self.Act is not None: x = self.Act(x)
                if self.bn is not None: x = self.bn(x)
                return x


# Encoding Block (Conv)
class EncodeBlock(nn.Module):
        def __init__(self,fi,fo,k=4,s=2,p=1,useBN = True,Act = F.relu):
                super(EncodeBlock, self).__init__()
		
                self.conv = nn.Conv2d(fi,fo,k,s,p, bias = False)
                #self.pool = nn.MaxPool2d(2)
                self.bn = nn.BatchNorm2d(fo) if useBN else None
                self.Act = Act

        def forward(self,x):
                x = self.conv(x)
                #x = self.pool(x)
                if self.Act is not None: x = self.Act(x)
                if self.bn is not None: x = self.bn(x)
                return x

# ======== RESNET LAYERS ==========

class ResBlock(nn.Module):
        def __init__(self, fi, fo):
                super(ResBlock, self).__init__()

                self.pad = nn.ReflectionPad2d(1)
                self.conv1 = EncodeBlock(fi, fo, 3, 1, 0)
                self.conv2 = EncodeBlock(fo, fo, 3, 1, 0, Act = None)

        def forward(self, x):
                y = self.pad(x)
                y = self.conv1(y)
                y = self.pad(y)
                y = self.conv2(y)

                return x + y
