import argparse, os
import glob
import jittor as jt
from jittor import optim
from jittor.dataset import Dataset
from jittor import lr_scheduler
from model import L1_Charbonnier_loss, LapSRN
from dataset import LapSRNDataset
from utils import LRScheduler, PSNR
from test import eval
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description="Jittor LapSRN")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--scale", default=4, type=int, help="evalution scale factor, Default: 4")

global opt, model
opt = parser.parse_args()

if(opt.cuda):
    print("Using CUDA")
    jt.flags.use_cuda = 1

model = LapSRN()
dataset = LapSRNDataset("data/lap_pry_x4_small.h5", batch_size=opt.batchSize)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = LRScheduler(optimizer, opt.lr) # mutates learning rate of optimizer
lossFn = L1_Charbonnier_loss()

def saveModel(model, epoch, isBest=False):
    model_folder = "saves/"
    if isBest:
        path = model_folder + "model_best.pkl"
    else:
        path = model_folder + "lapsrn_model_epoch_{}.pkl".format(epoch)
    model.save(path)
    print("Model saved as: {}".format(path))

def train(model, optimizer, epoch, dataloader):
    print("Epoch:{}, Learning rate:{}".format(epoch, optimizer.lr))
    # set the mode to train
    model.train()
    for inputData, label_x2, label_x4 in dataloader:
        HR_2x, HR_4x = model(inputData)
        loss_x2 = lossFn(HR_2x, label_x2)
        loss_x4 = lossFn(HR_4x, label_x4)
        # accumulate loss
        optimizer.backward(loss=loss_x2)
        optimizer.step(loss_x4)

best_psnr = 0
for epoch in range(opt.epochs):
    scheduler.step(len(dataset) * opt.batchSize)
    train(model, optimizer, epoch, dataset)
    psnr_predicted = eval(model)
    best_psnr = max(best_psnr, psnr_predicted)
    if best_psnr == psnr_predicted:
        saveModel(model, epoch, isBest=True)
    print(f'Epoch:{epoch} PSNR={psnr_predicted:.4f}, Best={best_psnr:.4f}')