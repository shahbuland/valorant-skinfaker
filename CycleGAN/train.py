from .constants import *
from .ops import Tensor, npimage
from .util import imtile

import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
import os
import cv2

def train(model, A, B, epochs, log_interval = LOG_INTERVAL, sample_interval = SAMPLE_INTERVAL, checkpoint_interval = CHECKPOINT_INTERVAL, base_dir = "./"):
        if USE_CUDA:
                model.cuda()
        if HALF_PRECISION:
                model.half()
                model.layers_to_half()
        
        data_sizes = [A.shape[0],B.shape[0]]
        print("Dataset shapes:", data_sizes)

        # Ensure existence of folders for checkpoints and samples
        try:
                os.makedirs('./checkpoints')
        except:
                pass
        try:
                os.makedirs('./samples')
        except:
                pass
        
        # Returns batch in form of (A,B) from provided indices
        # Indices could be random or fixed
        def get_batch(batch_size, inds = None):
                if inds is None:
                        indsA = torch.randint(0,data_sizes[0],(batch_size,))
                        indsB = torch.randint(0,data_sizes[1],(batch_size,))
                else:
                        indsA, indsB = inds

                batch_A = A[indsA]
                batch_B = B[indsB]

                if batch_size == 1:
                        batch_A = np.expand_dims(batch_A, 0)
                        batch_B = np.expand_dims(batch_B, 0)

                batch_A = torch.from_numpy(batch_A).permute(0,3,1,2)/255
                batch_B = torch.from_numpy(batch_B).permute(0,3,1,2)/255

                # Make sure both are right size
                batch_A = F.interpolate(batch_A, size=[IMG_SIZE, IMG_SIZE])
                batch_B = F.interpolate(batch_B, size=[IMG_SIZE, IMG_SIZE])
                
                if USE_CUDA:
                        batch_A = batch_A.cuda()
                        batch_B = batch_B.cuda()

                if HALF_PRECISION:
                        batch_A = batch_A.half()
                        batch_B = batch_B.half()
                        
                return batch_A, batch_B

        # Draws samples in 4x4 window
        def save_samples(title):
                A_sample, B_sample = get_batch(4)
                B_fake, A_fake = model.A_to_B(A_sample), model.B_to_A(B_sample)
                
                _A_sample = npimage(A_sample.float())
                _B_sample = npimage(B_sample.float())
                A_fake = npimage(A_fake.float())
                B_fake = npimage(B_fake.float())

                res = imtile(np.concatenate((_A_sample, B_fake, _B_sample, A_fake)), 4, 4)
                cv2.imwrite(title + ".png", res)

        # Actual training loop
        TOTAL_ITER = 0
        pass_size = min(A.shape[0], B.shape[0])
        TOTAL_PASSES = pass_size * epochs
        for EPOCH in range(epochs):
                # Get random indices for entire epoch for both A and B
                epoch_inds = [torch.randint(0, A.shape[0], (pass_size,)),
                              torch.randint(0, B.shape[0], (pass_size,))]
                
                for ITER in range(pass_size//BATCH_SIZE):
                        inds_A = epoch_inds[0][ITER*BATCH_SIZE:(ITER+1)*BATCH_SIZE]
                        inds_B = epoch_inds[1][ITER*BATCH_SIZE:(ITER+1)*BATCH_SIZE]

                        # Train discriminators
                        A_batch,B_batch = get_batch(BATCH_SIZE, [inds_A, inds_B])
                        D_A_loss,D_B_loss = model.train_disc_on_batch(A_batch,B_batch)
                        
                        # Train generators
                        G_A_loss, G_B_loss = model.train_gen_on_batch(A_batch, B_batch)
                        
                        # Write things down
                        
                        if (TOTAL_ITER + 1) % log_interval == 0:
                                print("Epoch: [" + str(EPOCH) + "/" + str(epochs) + "] Total Progress: " + str(round(100 * TOTAL_ITER / TOTAL_PASSES, 2)) + "%")
                                #print("D A Loss:",D_A_loss)
                                #print("G A Loss:",G_A_loss)
                                #print("D B Loss:",D_B_loss)
                                #print("G B Loss:",G_B_loss)

                        if (TOTAL_ITER+1) % sample_interval == 0:
                                save_samples(base_dir + "samples/"+str(TOTAL_ITER))
                        if (TOTAL_ITER+1) % checkpoint_interval == 0:
                                model.save_checkpoint(base_dir + "checkpoints", str(TOTAL_ITER))

                        TOTAL_ITER += 1
