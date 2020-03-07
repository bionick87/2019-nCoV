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

if __name__ == '__main__':
    Flags = gflags.FLAGS
    gflags.DEFINE_bool   ("cuda", True, "use cuda")
    ############################################
    gflags.DEFINE_string ("train_path", "/vol/biomedic2/ns87/conv-19/train", "training folder")
    gflags.DEFINE_string ("test_path", "/vol/biomedic2/ns87/conv-19/test", 'path of testing folder')
    gflags.DEFINE_string ("valid_path", "/vol/biomedic2/ns87/conv-19/valid", 'path of testing folder')
    ############################################
    gflags.DEFINE_string ("save_folder", "/vol/biomedic2/ns87/conv-19-save", 'path of testing folder')
    ############################################
    gflags.DEFINE_integer("workers", 4, "number of dataLoader workers")
    gflags.DEFINE_integer("batch_size", 10, "number of batch size")
    gflags.DEFINE_float  ("lr", 1e-3, "learning rate")
    gflags.DEFINE_integer("valid_every", 1, "valid model after each test_every iter.")
    gflags.DEFINE_integer("max_iter_train", 1000, "number of iteration for the training stage")
    gflags.DEFINE_integer("max_iter_valid", 300, "number of iteration for the valid stage")
    gflags.DEFINE_integer("max_iter_save",  10, "number of iteration before saving model")
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
    loss_BCE    = torch.nn.BCEWithLogitsLoss(size_average=True)
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
    for epoch in range(Flags.nepochs):
        epoch_valid = 0 
        loss_val    = 0
        optimizer.zero_grad()
        print("\n ...Train at epoch " +str(epoch))
        for batch_id, (img1, img2, label) in tqdm(enumerate(trainLoader, 1)):
            net.train()  
            if Flags.cuda:
                img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
            else:
                img1, img2, label = Variable(img1), Variable(img2), Variable(label)
            optimizer.zero_grad()
            output    = net.forward(img1, img2)
            loss      = loss_BCE   (output, label)
            loss_val += loss.item  ()
            loss.backward()
            optimizer.step()
        if epoch % Flags.valid_every == 0:
            net.eval()
            list_err = []
            print("\n ...Valid")
            y_actual = []
            y_hat    = []
            for _, (valid1, valid2, label_valid) in tqdm(enumerate(validLoader, 1)):
                if Flags.cuda:
                    test1, test2  = valid1.cuda(), valid2.cuda()
                else:
                     test1, test2 = Variable(valid1), Variable(valid2)
                output = net.forward(test1, test2).data.cpu().numpy()
                pred   = np.argmax(output)
                y_actual.append(1)
                if pred ==1:
                   y_hat.append(1)
                else:
                   y_hat.append(0)
            TP, FP, TN, FN = measure(y_actual, y_hat)
            sensitivity    = TP/(TP+FN)
            sensitivity_list.append(sensitivity)
            plot(sensitivity_list,save_path)
            if epoch % Flags.max_iter_save == 0:
                print("\n ...Save model")
                torch.save(net.state_dict(),os.path.join(model_path,"model_"+str(epoch_valid)+'.pt'))
                epoch_valid   += 1























        
    
