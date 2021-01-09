from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from .layers import EncodeBlock, DecodeBlock, ResBlock
from .constants import *
from .ops import get_labels, Tensor
from .loss_funcs import Discriminator_Loss, Generator_Loss

# Generator 
class Generator(nn.Module):
	def __init__(self):
		super(Generator,self).__init__()
		
		# Generator is an encoder-decoder
		# Assuming input is 64x64:
		ech = [CHANNELS,64,128,256,512,512,512]

		self.encode_blocks = nn.ModuleList()

		for i in range(len(ech)-1):
			self.encode_blocks.append(EncodeBlock(ech[i],ech[i+1],Act=nn.LeakyReLU(0.2)))

		# Now 512x4x4

		dch = [512,512,512,256,128,64,CHANNELS]

		self.decode_blocks = nn.ModuleList()

		for i in range(len(dch) - 2):
			self.decode_blocks.append(DecodeBlock(dch[i],dch[i+1]))
	
		self.decode_blocks.append(DecodeBlock(dch[-2],dch[-1],Act=torch.sigmoid,useBN=False))

		# Now CHANNELS x 64 x 64

	def forward(self, x):
		for block in self.encode_blocks:
			x = block(x)

		for block in self.decode_blocks:
			x = block(x)
		
		return x

# ====== G with residual layers

class resGenerator(nn.Module):
        def __init__(self):
                super(resGenerator, self).__init__()

                self.blocks = nn.ModuleList()

                # Conv
                self.blocks.append(nn.ReflectionPad2d(3))
                self.blocks.append(EncodeBlock(CHANNELS, 64, 7, 1, 0))
                self.blocks.append(EncodeBlock(64, 128, 3, 2, 1))
                self.blocks.append(EncodeBlock(128, 256, 3, 2, 1))

                # Res
                num_res_blocks = 9
                for i in range(num_res_blocks):
                        self.blocks.append(ResBlock(256, 256))

                # Deconv
                self.blocks.append(DecodeBlock(256, 128, 3, 2, 1, 1))
                self.blocks.append(DecodeBlock(128, 64, 3, 2, 1, 1))
                self.blocks.append(nn.ReflectionPad2d(3))
                self.blocks.append(EncodeBlock(64, CHANNELS, 7, 1, 0, useBN = False, Act = None))
                
        def forward(self, x):
                for block in self.blocks:
                        x = block(x)

                return torch.sigmoid(x)
# Discriminator
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		# Just encoder
		# Assume input is 64x64
		# Ends in 1 for 4x4 patch
		ch = [CHANNELS,64,128,256,512,1]

		self.convblocks = nn.ModuleList()

		self.convblocks.append(
                        EncodeBlock(CHANNELS, 64, useBN = False, Act = nn.LeakyReLU(0.2)))
		self.convblocks.append(
                        EncodeBlock(64, 128, Act = nn.LeakyReLU(0.2)))
		self.convblocks.append(
                        EncodeBlock(128, 256, Act = nn.LeakyReLU(0.2)))
		self.convblocks.append(
                        EncodeBlock(256, 512, 4, 1, (1,2), Act = nn.LeakyReLU(0.2)))
		self.convblocks.append(
                        EncodeBlock(512, 1, 4, 1, (2, 1), useBN = False, Act = None))

	def forward(self,x):
		for block in self.convblocks:
			x = block(x)

		return x

# CycleGAN model
class CycleGAN(nn.Module):
	def __init__(self):
		super(CycleGAN,self).__init__()

		# Models
		self.models = nn.ModuleDict()
		self.models["G_BA"] = resGenerator()
		self.models["G_AB"] = resGenerator()
		self.models["D_A"] = Discriminator()
		self.models["D_B"] = Discriminator()

		# Optimizers
		self.opts = {}
		for name, model in self.models.items():
			self.opts[name] = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE, eps = 1e-04 if HALF_PRECISION else 1e-08)

		# Init weights
		self.apply(self.init_weights)
                
	# Paper intializes from normal(0, sqrt(0.02))
	def init_weights(self, m):
		if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
			torch.nn.init.normal_(m.weight, 0, 0.02)

        # Convert all layers to half precision
	def layers_to_half(self):
		for layer in self.modules():
			if isinstance(layer, nn.BatchNorm2d):
				layer.float()

	# Set learning rate for all optimizers
	def set_lr(self, new_lr):
		for opt in self.opts.values():
			for g in opt.param_groups:
				g['lr'] = new_lr
        
	# Functions that simplify using all the models
	def A_to_B(self,A):
		return self.models["G_AB"](A)

	def B_to_A(self,B):
		return self.models["G_BA"](B)

	def judge_A(self,A):
		return self.models["D_A"](A)

	def judge_B(self,B):
		return self.models["D_B"](B)

	def load_checkpoint(self, path, label = ""):
		try:
			self.load_state_dict(torch.load(path + "./" + label + "params.pt"))
			print("Loaded checkpoint")
		except:
			print("Could not load checkpoint")
			print("Do you want to continue? [Y/N]")
			inp = str(input())
			if(inp == "y" or inp == "Y"):
				return
			else:
				exit(0)
				
	def load_checkpoint_from_path(self, path):
		try:
			self.load_state_dict(torch.load(path))
			print("Loaded checkpoint")
		except:
			print("Could not load checkpoint")
			print("Do you want to continue? [Y/N]")
			inp = str(input())
			if(inp == "y" or inp == "Y"):
				return
			else:
				exit(0)
	
	def save_checkpoint(self, path, label):
		torch.save(self.state_dict(), path + "./" + label + "params.pt")

	def forward(self, batch_A, batch_B):
		# Generate fake images 
		fake_B = self.A_to_B(batch_A)
		fake_A = self.B_to_A(batch_B)
		
		# Judge images
		real_score_A = self.judge_A(batch_A)
		fake_score_A = self.judge_A(fake_A)
		real_score_B = self.judge_B(batch_B)
		fake_score_B = self.judge_B(fake_B)

		return real_score_A, fake_score_A, real_score_B, fake_score_B

	def train_disc_on_batch(self,batch_A,batch_B):
		
		# Zero out all optimizers
		for opt in self.opts.values():
			opt.zero_grad()

		# Generate fake images 
		fake_B = self.A_to_B(batch_A)
		fake_A = self.B_to_A(batch_B)

		# Explanation of loss functions can be found within them
		D_A_loss = 0.5*(Discriminator_Loss(batch_A, self.models["D_A"], True) +
				Discriminator_Loss(fake_A, self.models["D_A"], False))
		
		D_B_loss = 0.5*(Discriminator_Loss(batch_B, self.models["D_B"], True) +
				Discriminator_Loss(fake_B, self.models["D_B"], False))

		D_A_loss.backward()
		D_B_loss.backward()

		self.opts["D_A"].step()
		self.opts["D_B"].step()

		return D_A_loss.item(), D_B_loss.item()

	def train_gen_on_batch(self,batch_A,batch_B):
		
		# Zero out all optimizers
		for opt in self.opts.values():
			opt.zero_grad()
		
		loss_A, loss_B = Generator_Loss(batch_A, batch_B,
						self.models["G_BA"], self.models["G_AB"],
						self.models["D_A"], self.models["D_B"])

		# Retain graph cause both models are used in both objectives
		loss_A.backward(retain_graph=True) 
		loss_B.backward()
		
		self.opts["G_AB"].step()
		self.opts["G_BA"].step()

		return loss_A.item(), loss_B.item()


