import torch
from torch import nn
from .constants import *

# Loss for discriminator
# Tensor, Model, Bool -> Tensor (Loss)
def Discriminator_Loss(inp, model, label):
    predicted_labels = model(inp)

    # Create label patch like 
    if label == True:
        labels = torch.ones_like(predicted_labels)
    else:
        labels = torch.zeros_like(predicted_labels)

    # MSE loss
    mse = nn.MSELoss()

    return mse(predicted_labels, labels)

# Generator loss
# Idt + Adversarial + Reconstruction
# Tensor, Tensor, Model, Model, Model, Model -> Float (Loss)
def Generator_Loss(A, B, G_BA, G_AB, D_A, D_B):
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    # Identity losses (i.e. we want G_AB(B) = B, things already in B shouldn't be changed)
    if IDT_WEIGHT != 0:
        idt_A = G_BA(A)
        idt_B = G_AB(B)
        idt_loss_A = l1(idt_A, A)
        idt_loss_B = l1(idt_B, B)
    else:
        idt_loss_A = 0
        idt_loss_B = 0

    # Adv losses (How well each network fools respective discrminators
    fake_A = G_BA(B)
    fake_B = G_AB(A)
    labels_A = D_A(fake_A)
    labels_B = D_B(fake_B)
    # Trick discriminators to output ones
    true_labels_A = torch.ones_like(labels_A)
    true_labels_B = torch.ones_like(labels_B)
    adv_loss_A = mse(labels_A, true_labels_A)
    adv_loss_B = mse(labels_B, true_labels_B)

    # Reconstruction loss (Cycle consistency)
    A_rec = G_BA(fake_B)
    B_rec = G_AB(fake_A)
    rec_loss_A = l1(A_rec, A)
    rec_loss_B = l1(B_rec, B)

    # Identity losses for both
    loss_A = IDT_WEIGHT * idt_loss_A + ADV_WEIGHT * adv_loss_A + CYCLE_WEIGHT * rec_loss_A
    loss_B = IDT_WEIGHT * idt_loss_B + ADV_WEIGHT * adv_loss_B + CYCLE_WEIGHT * rec_loss_B
    return loss_A, loss_B


    
    
