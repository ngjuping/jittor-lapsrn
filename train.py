import argparse, os
import glob
import jittor as jt
from jittor import optim
from jittor.dataset import Dataset
from jittor import lr_scheduler
from model import L1_Charbonnier_loss, LapSRN
from dataset import LapSRNDataset
from utils import LRScheduler
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
scheduler = LRScheduler(optimizer, opt.lr)
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

    print("Epoch={}, lr={}".format(epoch, optimizer.lr))
    
    # set the mode to train
    model.train()

    for inputData, label_x2, label_x4 in dataloader:

        HR_2x, HR_4x = model(inputData)

        loss_x2 = lossFn(HR_2x, label_x2)
        loss_x4 = lossFn(HR_4x, label_x4)

        # accumulate loss
        optimizer.backward(loss=loss_x2)
        optimizer.step(loss_x4)

def saveToFile(className, inputImage, outputImage, gt):
    plt.imshow(inputImage)
    plt.savefig(f'{className}_input.png')
    plt.imshow(outputImage)
    plt.savefig(f'{className}_output.png')
    plt.imshow(gt)
    plt.savefig(f'{className}_gt.png')

def eval(model, visualize=True):
    # set the mode to eval
    model.eval()

    image_list = glob.glob("Set5"+"/*.mat") 

    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0

    for image_name in image_list:
        im_gt_y = sio.loadmat(image_name)['im_gt_y']
        im_b_y = sio.loadmat(image_name)['im_b_y']
        im_l_y = sio.loadmat(image_name)['im_l_y']

        className = image_name.split('_')[0]
        
        im_gt_y = im_gt_y.astype(float)
        im_b_y = im_b_y.astype(float)
        im_l_y = im_l_y.astype(float)

        psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)
        avg_psnr_bicubic += psnr_bicubic

        im_input = im_l_y/255.
        im_input = jt.Var(im_input).float().view(1, -1, im_input.shape[0], im_input.shape[1])

        HR_2x, HR_4x = model(im_input)

        im_h_y = HR_4x.data[0].astype(np.float32)
        im_h_y = im_h_y*255.
        im_h_y[im_h_y<0] = 0
        im_h_y[im_h_y>255.] = 255.
        im_h_y = im_h_y[0,:,:]

        if visualize:
            saveToFile(className, im_l_y, im_h_y, im_gt_y)

        psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
        avg_psnr_predicted += psnr_predicted

    # return average predicted PSNR
    return avg_psnr_predicted/len(image_list)

best_psnr = 0
for epoch in range(opt.epochs):
    scheduler.step(len(dataset) * opt.batchSize)
    train(model, optimizer, epoch, dataset)
    psnr_predicted = eval(model)
    best_psnr = max(best_psnr, psnr_predicted)
    if best_psnr == psnr_predicted:
        saveModel(model, epoch)
    print(f'val PSNR={psnr_predicted:.4f}, best={best_psnr:.4f}')