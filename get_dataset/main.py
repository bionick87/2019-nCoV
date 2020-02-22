#https://stackoverflow.com/questions/17856242/convert-string-to-image-in-python
from textwrap import wrap
import text_to_image
import os
from tqdm import tqdm
import cv2

def readGenSeq(path_file):
    with open(path_file, 'r') as f:
        data = f.read()
    return data


def makeFolder(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)

def splitData(seq,nstrands,pathSave):
    nstrandsList = wrap(seq,nstrands)
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
    path_dataset_fodler = "/Users/nicolosavioli/Desktop/corona_dataset"
    makeFolder(path_dataset_fodler)
    save_path_file_nCoV      = os.path.join(path_dataset_fodler,"positive")
    save_path_file_HIV       = os.path.join(path_dataset_fodler,"negtive")
    makeFolder(save_path_file_nCoV)
    makeFolder(save_path_file_HIV)
    nCoV                = readGenSeq(path_file_nCoV)
    HIV                 = readGenSeq(path_file_HIV)
    nstrands            = 20
    print("\n ... Generete nCoV-2019 strands")
    splitData(nCoV,nstrands,save_path_file_nCoV)
    print("\n ... Generete HIV strands")
    splitData(HIV,nstrands,save_path_file_HIV)
