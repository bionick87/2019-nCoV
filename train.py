import torch
import pickle
import torchvision
import torchvision.datasets as dset
from   torchvision      import transforms
from   dataset          import Dataset
from   torch.utils.data import DataLoader
from   torch.autograd   import Variable
import matplotlib.pyplot as plt
from   model            import SiameseNet
import time
import numpy            as np
import gflags 
import sys
from   collections      import deque
import os
from   tqdm import tqdm
from utils import *
import random


if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_bool   ("cuda", True, "use cuda")
    ############################################
    gflags.DEFINE_string ("train_path", "/vol/biomedic2/ns87/conv-19/train", "training folder")
    gflags.DEFINE_string ("test_path", "/vol/biomedic2/ns87/conv-19/test",   "path of testing folder")
    gflags.DEFINE_string ("valid_path", "/vol/biomedic2/ns87/conv-19/valid", "path of testing folder")
    ############################################
    gflags.DEFINE_string ("save_folder", "/vol/biomedic2/ns87/conv-19-save", 'path of testing folder')
    ############################################
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 10, "number of batch size")
    gflags.DEFINE_float  ("lr", 1e-4, "learning rate")

    ############################################
    gflags.DEFINE_integer("valid_every", 1, "valid model after each test_every iter.")
    gflags.DEFINE_integer("save_every",  500, "save model after each test_every iter.")
    ############################################
    
    gflags.DEFINE_integer("max_iter_train", 500, "number of iteration for the training stage")
    gflags.DEFINE_integer("max_iter_valid", 200, "number of iteration for the valid stage")
    gflags.DEFINE_integer("nepochs", 1000, "number of epoch")
    gflags.DEFINE_string ("gpu_ids", "0", "gpu ids used to train")
    Flags(sys.argv)
    #############################################
    trainSet    = Dataset(Flags.train_path,Flags.test_path,Flags.valid_path,Flags.max_iter_train,"train")
    trainLoader = DataLoader(trainSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    #############################################
    validSet    = Dataset(Flags.valid_path,Flags.test_path,Flags.valid_path,Flags.max_iter_valid,"valid")
    validLoader = DataLoader(validSet, batch_size=Flags.batch_size, shuffle=False, num_workers=Flags.workers)
    #############################################
    loss_MSE    = torch.nn.MSELoss()
    net         = SiameseNet()
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
    optimizer        = torch.optim.Adam(net.parameters(),lr = Flags.lr )
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
                pred_gt = random.randint(0, 1)
                if pred_gt == 1:
                    output_net    = net.forward(test1, test2)
                else:
                    output_net    = net.forward(test1, test1)
                y_actual = []
                y_hat    = []
                for i in range(output_net.size()[0]):
                    output_net_np = output_net[i].data.cpu().numpy()
                    y_actual.append(output_net_np)
                    print(output_net_np)
                    if output_net_np == pred_gt:
                       y_hat.append(1)
                    else:
                       y_hat.append(0)
                TP, FP, TN, FN = measure(y_actual, y_hat)
                if TP == 0 or FN == 0:
                   sensitivity  = 0
                else:                    
                    sensitivity = TP/(TP+FN)
                sensitivity_valid.append(sensitivity)
            sensitivity_list.append(np.mean(sensitivity_valid))
            plot_sensitivity(sensitivity_list,save_path)
            if epoch % Flags.save_every == 0:
                print("\n ...Save model")
                torch.save(net.state_dict(),os.path.join(model_path,"model_"+str(epoch_valid)+'.pt'))
                epoch_valid   += 1























        
    
