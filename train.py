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

def setup():
    global opt, model, dataset, optimizer, scheduler, lossFn, saved_models_folder # expose globally

    parser = argparse.ArgumentParser(description="Jittor LapSRN")
    parser.add_argument("--cuda", dest="cuda", action="store_true")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false")
    parser.add_argument("--batchSize", type=int, default=64, help="Batch size")
    parser.add_argument("--startFrom", type=int, default=0, help="Continue from")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
    parser.add_argument("--scale", default=4, type=int, help="Evalution scale factor, Default: 4")
    parser.set_defaults(cuda=False)
    opt = parser.parse_args()

    if opt.cuda:
        print("Using CUDA")
        jt.flags.use_cuda = 1

    saved_models_folder = "saves/"
    folder_exists = os.path.exists(saved_models_folder)
    if not folder_exists:
        os.makedirs(saved_models_folder) # Create a new directory because it does not exist 
        print("The saved models directory " + saved_models_folder + "is created!")

    model = LapSRN() # initialize
    
    if opt.startFrom > 0:
        model.load(saved_models_folder + "lapsrn_model_epoch_{}.pkl".format(opt.startFrom))
    
    dataset = LapSRNDataset("data/lap_pry_x4_small.h5", batch_size=opt.batchSize)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = LRScheduler(optimizer, opt.lr) # mutates learning rate of optimizer
    lossFn = L1_Charbonnier_loss()

def saveModel(model, epoch, isBest=False):
    path = saved_models_folder + "lapsrn_model_epoch_{}.pkl".format(epoch)
    model.save(path)
    # save an extra best model
    if isBest:
        model.save(saved_models_folder + "model_best.pkl")
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

if __name__ == "__main__":
    setup()
    best_psnr = 0
    epoch = opt.startFrom
    while(True):
        epoch += 1
        scheduler.step(epoch) # update learning rate
        train(model, optimizer, epoch, dataset) # train model with dataset
        psnr_predicted = eval(model, opt) # evaluate
        best_psnr = max(best_psnr, psnr_predicted) # store best result
        if best_psnr == psnr_predicted:
            saveModel(model, epoch, isBest=True) # save model on disk as pkl
        print(f'Epoch:{epoch} PSNR={psnr_predicted:.4f}, Best={best_psnr:.4f}')
        
