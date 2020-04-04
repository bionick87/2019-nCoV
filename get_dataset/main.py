#https://stackoverflow.com/questions/17856242/convert-string-to-image-in-python
from   textwrap import wrap
import text_to_image
import os
from   tqdm import tqdm
import cv2
from   sklearn.model_selection import train_test_split
import re


def translate(seq): 
    table = { 
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                  
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 
        'TAC':'Y', 'TAT':'Y', 'TAA':'', 'TAG':'', 
        'TGC':'C', 'TGT':'C', 'TGA':'', 'TGG':'W', 
    } 
    protein = "" 
    codons  = [seq[i:i+3] for i in range(0, len(seq), 3)]
    for codon in codons:
        if len(codon)>2:
           protein+= table[codon] 
    return protein 


def readGenSeq(path_file):
    with open(path_file, 'r') as f:
        data = f.read()
    data = data.replace("\n", "") 
    data = data.replace("\r", "") 
    return data

def SplitStrands(seq,nstrands):
    seq_txt         = translate(readGenSeq(seq))
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


def getDataset():
    # https://www.ncbi.nlm.nih.gov/nuccore/MN908947.3?report=fasta
    path_file_nCoV      = "./virus_genome/2019-nCoV.txt"
    path_file_HIV       = "./virus_genome/HIV.txt"
    path_ebola          = "./virus_genome/ebola.txt"
    # Dataset folder path 
    path_dataset_fodler = "/vol/biomedic2/ns87/conv-19"
    # Number of RNA vrius strands
    nstrands            = 10
    ##############################
    nCoV_train,\
    nCoV_test,\
    nCoV_valid          = SplitStrands(path_file_nCoV,nstrands)
    ##############################
    HIV_train,\
    HIV_test,\
    HIV_valid           = SplitStrands(path_file_HIV,nstrands) 
    ##############################
    ebola_train,\
    ebola_test,\
    ebola_valid         = SplitStrands(path_file_HIV,nstrands) 
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
    getData(ebola_train,save_path_file_HIV_train)
    ###############################
    print("\n ... Generete nCoV-2019 dataset valid")
    getData(nCoV_valid,save_path_file_nCoV_valid)
    getData(HIV_valid,save_path_file_HIV_valid)
    getData(ebola_valid,save_path_file_HIV_valid)
    ##############################
    print("\n ... Generete nCoV-2019 dataset test")
    getData(nCoV_test,save_path_file_nCoV_test)
    getData(HIV_test,save_path_file_HIV_test)
    getData(ebola_test,save_path_file_HIV_test)
    ##############################

def getHR1Domain_target():
    seq_hr1       = "./virus_genome/HR1.txt"
    path_save     = "./virus_genome"
    protein_hr1   = readGenSeq(seq_hr1)
    nstrandsP     = wrap(protein_hr1,10)
    getData(nstrandsP,path_save)

def getPep():
    regex = {
	"capital_letters": re.compile("[A-Z]")
    }
    clean     = []
    cont      = 0
    path_file = "./virus_genome/peptite/antiviral.fasta"
    pathSave  = "path-to/pepdata"
    with open(path_file, 'r') as f:
        lines = f.readlines()
    for l in lines:
        clean.append(l.replace("\n", ""))    
    for pep in clean:
        # string preprocessing
        if ">" in pep: 
            continue
        if "-" in pep:
            continue
        if "(" in pep:
            continue
        if regex["capital_letters"].match(pep):
            print("..." + pep)
            text_to_image.encode(pep,os.path.join(pathSave,pep+".png"))
            img = cv2.imread(os.path.join(pathSave,pep+".png"))
            img = cv2.resize(img,(256,256))
            cv2.imwrite(os.path.join(pathSave,pep+".png"), img) 

if __name__ == "__main__":
    # Generation dataset
    getDataset()
    # Generation HR1 images
    # getHR1Domain_target()
    # generation SATPdb peptides images
    # getPep()






