import argparse, os
import jittor as jt
from jittor import optim
from jittor.dataset import Dataset
from jittorLAPSRN import L1_Charbonnier_loss, LapSRN
from jittordataset import LapSRNDataset
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description="Jittor LapSRN")
parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

global opt, model
opt = parser.parse_args()

model = LapSRN()
dataset = LapSRNDataset("data/lap_pry_x4_small.h5")
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
criterion = L1_Charbonnier_loss()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "lapsrn_model_epoch_{}.pkl".format(epoch)
    model.save(model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def train(model, optimizer, epoch, dataloader):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    
    model.train()

    for inputData, label_x2, label_x4 in dataloader:
        # print(inputData.shape)
        # print(label_x2.shape)
        # print(label_x4.shape)

        HR_2x, HR_4x = model(inputData)

        loss_x2 = criterion(HR_2x, label_x2)
        loss_x4 = criterion(HR_4x, label_x4)
        loss = loss_x2 + loss_x4

        optimizer.backward(loss=loss_x2)
        optimizer.step(loss_x4)

step = 0
best_acc = 0
for epoch in range(opt.nEpochs):

    train(model, optimizer, epoch, dataset)
    save_checkpoint(model, epoch)
    # best_acc = max(best_acc, acc)
    # print(f'val acc={acc:.4f}, best={best_acc:.4f}')