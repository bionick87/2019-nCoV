import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as dset
from   torchvision      import transforms
from   dataset          import Dataset
from   torch.utils.data import DataLoader
from   torch.autograd   import Variable
from   model            import SiameseNet
import time
import numpy            as np
import gflags 
import sys
import os
from   tqdm import tqdm
from   utils import *
import random
import math


if __name__ == '__main__':
    ############################################
    Flags = gflags.FLAGS
    gflags.DEFINE_bool   ("cuda", True, "use cuda")
    ############################################
    gflags.DEFINE_string ("train_path", "path-to-dataset/train", "training folder to be set")
    gflags.DEFINE_string ("test_path",  "path-to-dataset/test",   "path of testing folder to be set")
    gflags.DEFINE_string ("valid_path", "path-to-dataset/valid", "path of testing folder to be set")
    ############################################
    gflags.DEFINE_string ("save_folder", "path-to-save-results", 'path of testing folder to be set!')
    ############################################
    gflags.DEFINE_integer("workers", 8, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 10, "number of batch size")
    gflags.DEFINE_float  ("lr", 1e-3, "learning rate")
    ############################################
    gflags.DEFINE_integer("valid_every", 10, "valid model after each test_every iter.")
    gflags.DEFINE_integer("save_every",  10, "save model after each test_every iter.")
    ############################################
    gflags.DEFINE_integer("max_iter_train", 10000, "number of iteration for the training stage")
    gflags.DEFINE_integer("max_iter_valid", 200, "number of iteration for the valid stage")
    gflags.DEFINE_integer("nepochs", 2000, "number of epoch")
    gflags.DEFINE_string ("gpu_ids", "0", "gpu ids used to train")
    gflags.DEFINE_bool   ("retrain", True, "use cuda")
    gflags.DEFINE_string ("retrain_path", "path-to-retrain-model", 'path retrain')
    Flags(sys.argv)
    #############################################
    trainSet    = Dataset(Flags.train_path,Flags.test_path,Flags.valid_path,Flags.max_iter_train,"train")
    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    #############################################
    validSet    = Dataset(Flags.valid_path,Flags.test_path,Flags.valid_path,Flags.max_iter_valid,"valid")
    validLoader = DataLoader(validSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    #############################################
    loss_MSE    = torch.nn.MSELoss()
    if Flags.retrain:
        print("\n ... Retrain model")
        net     = loadModel(Flags.retrain_path)
    else:
        net     = SiameseNet()
    #############################################
    save_path   = os.path.join(Flags.save_folder,"save_data")
    model_path  = os.path.join(save_path,"models")
    #############################################
    makeFolder(save_path)
    makeFolder(model_path)
    # multi gpu
    if Flags.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = Flags.gpu_ids
        if len(Flags.gpu_ids.split(",")) > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    optimizer        = torch.optim.SGD(net.parameters(),lr = Flags.lr, momentum=0.9, nesterov=True)
    sensitivity_list = []
    loss_list        = [] 
    epoch_valid      = 0 
    for epoch in range(Flags.nepochs):
        loss_val    = 0
        print("\n ...Train at epoch " +str(epoch))
        cont_iter = 0 
        for batch_id, (img1, img2, label) in tqdm(enumerate(trainLoader, 1)):
            net.train()  
            if Flags.cuda:
                img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
            else:
                img1, img2, label = Variable(img1), Variable(img2), Variable(label)
            optimizer.zero_grad()
            output    = net.forward(img1, img2)
            loss      = loss_MSE       (output, label)
            loss_val += loss.item  ()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cont_iter += 1 
        loss_epoch = loss_val/cont_iter
        loss_list.append(loss_epoch)
        plot_loss(loss_list,save_path)
        if epoch % Flags.valid_every == 0:
            net.eval()
            print("\n ...Valid")
            sensitivity_valid = []
            for _, (valid1, valid2, label_valid) in tqdm(enumerate(validLoader, 1)):
                if Flags.cuda:
                    test1, test2  = valid1.cuda(), valid2.cuda()
                else:
                     test1, test2 = Variable(valid1), Variable(valid2)
                output_net        = net.forward(test1, test2)
                y_actual = []
                y_hat    = []
                for i in range(output_net.size()[0]):
                    output_net_np = math.ceil(output_net[i].data.cpu().numpy())
                    y_actual.append(1)
                    if output_net_np == 1.0 or output_net_np == 1:
                       y_hat.append(1)
                    else:
                       y_hat.append(0)
                TP, FP, TN, FN = measure(y_actual, y_hat)
                if TP == 0 or FN == 0:
                    sensitivity  = 0
                else:                    
                    sensitivity = 100*(TP/(TP+FN))
                sensitivity_valid.append(sensitivity)
            sensitivity_list.append(np.mean(sensitivity_valid))
            plot_sensitivity(sensitivity_list,save_path)
            if epoch % Flags.save_every == 0:
                print("\n ...Save model")
                torch.save(net.state_dict(),os.path.join(model_path,"model_"+str(epoch_valid)+'.pt'))
                epoch_valid   += 1
