#https://stackoverflow.com/questions/17856242/convert-string-to-image-in-python
from   textwrap import wrap
import text_to_image
import os
from   tqdm import tqdm
import cv2
from   sklearn.model_selection import train_test_split


def readGenSeq(path_file):
    with open(path_file, 'r') as f:
        data = f.read()
    return data

def SplitStrands(seq,nstrands):
    seq_txt         = readGenSeq(seq)
    nstrandsList    = wrap(seq_txt,nstrands)
    train, tmp_test = train_test_split(nstrandsList, test_size=0.4, random_state=42)
    test,  valid    = train_test_split(tmp_test,     test_size=0.5, random_state=42)
    return train,test,valid

def makeFolder(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)

def getData(nstrandsList,pathSave):
    cont = 0
    for s in tqdm(nstrandsList):
        text_to_image.encode(s,os.path.join(pathSave,"strands_"+str(cont)+".png"))
        img = cv2.imread(os.path.join(pathSave,"strands_"+str(cont)+".png"))
        img = cv2.resize(img,(256,256))
        cv2.imwrite(os.path.join(pathSave,"strands_"+str(cont)+".png"), img) 
        cont += 1

if __name__ == "__main__":

    # https://www.ncbi.nlm.nih.gov/nuccore/MN908947.3?report=fasta
    path_file_nCoV      = "/Users/nicolosavioli/Desktop/2019-nCoV/virus_genome/2019-nCoV.txt"
    path_file_HIV       = "/Users/nicolosavioli/Desktop/2019-nCoV/virus_genome/HIV.txt"
    # Dataset folder path 
    path_dataset_fodler = "/Users/nicolosavioli/Desktop/dataset"
    # Number of RNA vrius strands
    nstrands            = 12
    ##############################
    nCoV_train,\
    nCoV_test,\
    nCoV_valid          = SplitStrands(path_file_nCoV,nstrands)
    ##############################
    HIV_train,\
    HIV_test,\
    HIV_valid           = SplitStrands(path_file_HIV,nstrands) 
    ##############################
    makeFolder(path_dataset_fodler)
    ##############################
    train_path = os.path.join(path_dataset_fodler,"train")
    valid_path = os.path.join(path_dataset_fodler,"valid")
    test_path  = os.path.join(path_dataset_fodler,"test")
    ##############################
    makeFolder(train_path)
    makeFolder(valid_path)
    makeFolder(test_path)
    ############################## 
    save_path_file_nCoV_train = os.path.join(train_path,"positive")
    save_path_file_HIV_train  = os.path.join(train_path,"negative")
    makeFolder(save_path_file_nCoV_train)
    makeFolder(save_path_file_HIV_train)
    ##############################
    save_path_file_nCoV_valid = os.path.join(valid_path,"positive")
    save_path_file_HIV_valid  = os.path.join(valid_path,"negative")
    makeFolder(save_path_file_nCoV_valid)
    makeFolder(save_path_file_HIV_valid)
    ##############################
    save_path_file_nCoV_test = os.path.join(test_path,"positive")
    save_path_file_HIV_test  = os.path.join(test_path,"negative")
    makeFolder(save_path_file_nCoV_test)
    makeFolder(save_path_file_HIV_test)
    ##############################
    print("\n ... Generete nCoV-2019 dataset train")
    getData(nCoV_train,save_path_file_nCoV_train)
    getData(HIV_train,save_path_file_HIV_train)
    ###############################
    print("\n ... Generete nCoV-2019 dataset valid")
    getData(nCoV_valid,save_path_file_nCoV_valid)
    getData(HIV_valid,save_path_file_HIV_valid)
    ###############################
    print("\n ... Generete nCoV-2019 dataset test")
    getData(nCoV_test,save_path_file_nCoV_test)
    getData(HIV_test,save_path_file_HIV_test)
    ###############################


