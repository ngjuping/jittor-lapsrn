import argparse
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import jittor as jt
from model import LapSRN
from os import listdir
from os.path import isfile, join
from utils import PSNR

# jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description="Jittor LapSRN Test")
parser.add_argument("--cuda")
parser.add_argument("--model", default="saves/lapsrn_model_epoch_9.pkl", type=str, help="Load from which model")
parser.add_argument("--dataset", default="Set5", type=str, help="What dataset to use")
parser.add_argument("--scale", default=4, type=int, help="Scale factor, default to 4")

opt = parser.parse_args()

model = LapSRN()
model_names = [f for f in listdir("saves/") if isfile(join("saves/", f))]
model_names = sorted(model_names)
print(f"Using model: {model_names[-1]}")
model.load("saves/" + model_names[-1])

def saveImages(className, inputImage, outputImage, gt):
    plt.imshow(inputImage)
    plt.savefig(f'{className}_input.png')
    plt.imshow(outputImage)
    plt.savefig(f'{className}_output.png')
    plt.imshow(gt)
    plt.savefig(f'{className}_gt.png')

def eval(model, dataset="Set5", visualize=True):
    # set the mode to eval
    model.eval()
    image_list = glob.glob(dataset+"/*.mat") 
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
            saveImages(className, im_l_y, im_h_y, im_gt_y)

        psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
        avg_psnr_predicted += psnr_predicted

    print("PSNR_predicted=", avg_psnr_predicted/len(image_list))
    print("PSNR_bicubic=", avg_psnr_bicubic/len(image_list))

    # return average predicted PSNR
    return avg_psnr_predicted/len(image_list)

eval(model, dataset=opt.dataset)