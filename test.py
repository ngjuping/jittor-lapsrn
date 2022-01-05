import argparse
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import jittor as jt
from jittorLAPSRN import L1_Charbonnier_loss, LapSRN
import matplotlib.pyplot as plt

# jt.flags.use_cuda = 1

parser = argparse.ArgumentParser(description="Jittor LapSRN Test")
parser.add_argument("--cuda")
parser.add_argument("--model", default="saves/lapsrn_model_epoch_9.pkl", type=str, help="Load from which model")
parser.add_argument("--dataset", default="Set5", type=str, help="What dataset to use")
parser.add_argument("--scale", default=4, type=int, help="Scale factor, default to 4")

opt = parser.parse_args()

model = LapSRN()
model.load("saves/lapsrn_model_epoch_0.pkl")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def visualize(className, inputImage, outputImage, gt):
    plt.imshow(inputImage)
    plt.savefig(f'{className}_input.png')
    plt.imshow(outputImage)
    plt.savefig(f'{className}_output.png')
    plt.imshow(gt)
    plt.savefig(f'{className}_gt.png')

image_list = glob.glob(opt.dataset+"/*.*") 

avg_psnr_predicted = 0.0
avg_psnr_bicubic = 0.0
avg_elapsed_time = 0.0

for image_name in image_list:
    im_gt_y = sio.loadmat(image_name)['im_gt_y']
    im_b_y = sio.loadmat(image_name)['im_b_y']
    im_l_y = sio.loadmat(image_name)['im_l_y']

    className = image_name.split('_')[0]
    print(className)
    im_gt_y = im_gt_y.astype(float)
    im_b_y = im_b_y.astype(float)
    im_l_y = im_l_y.astype(float)

    psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)
    avg_psnr_bicubic += psnr_bicubic

    im_input = im_l_y/255.
    im_input = jt.Var(im_input).float().view(1, -1, im_input.shape[0], im_input.shape[1])

    start_time = time.time()
    HR_2x, HR_4x = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    im_h_y = HR_4x.data[0].astype(np.float32)
    im_h_y = im_h_y*255.
    im_h_y[im_h_y<0] = 0
    im_h_y[im_h_y>255.] = 255.
    im_h_y = im_h_y[0,:,:]

    visualize(className, im_l_y, im_h_y, im_gt_y)

    psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
    avg_psnr_predicted += psnr_predicted

print("Scale=", opt.scale)
print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted/len(image_list))
print("PSNR_bicubic=", avg_psnr_bicubic/len(image_list))
print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))
