# Artificial Intelligence model for COVID-2019 drug discovery 

The novel coronavirus (2019-nCoV) has been declared to be a new international health emergence and no specific drug has been yet identified.
Several methods are currently being evaluated such as protease and glycosylated spike (S) protein inhibitors, that outlines the main fusion site among coronavirus and host cells. 
Notwithstanding, the Heptad Repeat 1 (HR1) domain on the glycosylated spike (S) protein is the region with less mutability and then the most encouraging target for new inhibitors drugs. 
The novelty of the proposed approach, compared to others, lies in a precise training of a deep neural network toward the 2019-nCoV virus.
Where a Siamese Neural Network (SNN) has been trained to distingue the whole 2019-nCoV protein sequence amongst two different viruses family such as HIV-1 and Ebola.
In this way, the present deep learning system has precise knowledge of peptide linkage among 2019-nCoV protein structure and differently, of other works, is not trivially trained on public datasets that have not been provided any ligand-peptide information for 2019-nCoV. 
Suddenly, the SNN shows a sensitivity of 83% of peptide affinity classification, where $3027$ peptides on SATPdb bank have been tested towards the specific region HR1 of 2019-nCoV exhibiting an affinity of 93% for the peptidyl-prolyl cis-trans isomerase (PPIase) peptide. 
This affinity between PPIase and HR1 can open new horizons of research since several scientific papers have already shown that CsA immunosuppression drug, a main inhibitor of PPIase,  suppress the reproduction of different CoV virus included SARS-CoV and MERS-CoV.


## The code is released for research purposes only and not for commercial purposes.


![alt text](img/deep_model.png)


## Prerequisites

Before getting started, it's important to have a working environment with all dependencies satisfied. For this, we recommend using the Anaconda distribution of Python 3.5.

```
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

So PyTorch must be installed, please make sure that cuDNN is installed correctly (https://developer.nvidia.com/cudnn).

```
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

Then install the following libraries:

```
pip install torchvision
pip install opencv-python
pip install matplotlib
pip install tqdm
pip install textwrap3
pip install python-gflags
pip install text-to-image
```

## Use

### Generation of the dataset (COVID-2019 vs Ebola, HIV) 

 Go to ./get_dataset folder then open the main file 

Uncomment what you want to generate:
 
```

if __name__ == "__main__":
    # Generation dataset
    # getDataset()
    # Generation HR1 images
    # getHR1Domain_target()
    # generation SATPdb peptides images
    # getPep()

```

Then remember to change the paths within each function, or download the data (dataset-nConV-2019 and pepdata of SATPdb) from:

https://drive.google.com/open?id=1buUylzkMAM91Qs7z-ndvUKlvr4wbaiSq

After the unzip of data.zip you will find two folders:

i)  ./data/Dataset-nConV-2019: the dataset that I used to train all the models
ii) ./data/pepdata:  The SATPdb peptide for the inference 


### Training Setting 

Open the file train.py

Change the paths where you have located the dataset:


```
gflags.DEFINE_string ("train_path", "path-to-dataset/Dataset-nConV-2019/train", "training folder to be set")
gflags.DEFINE_string ("test_path",  "path-to-dataset/Dataset-nConV-2019/test",   "path of testing folder to be set")
gflags.DEFINE_string ("valid_path", "path-to-dataset/Dataset-nConV-2019/valid", "path of testing folder to be set")
gflags.DEFINE_string ("save_folder", "path-to-save-results", 'path of testing folder to be set!')

```

Single GPU: 

```
gflags.DEFINE_string ("gpu_ids", "0", "gpu ids used to train")
```


Multi-GPU:

```
gflags.DEFINE_string ("gpu_ids", "0,1", "gpu ids used to train")
```

Number of CPU threads setting:

```
gflags.DEFINE_integer("workers", 8, "number of dataLoader workers")

```

For running the training:

```
sh run_train.sh
```


### Inference Setting 

#### Download pre-trained Alexnet 2019-nCoV Siamese Neural Network model from this link:

https://drive.google.com/open?id=18Zu05OagMmaHfQoRJ1_vgM0zwfWwM0Yy


Go to the folder ./inference and open the file inference.py and change the following paths:

```
model_path     = "path-to-model/AlexNet_pretrain.pt"
train_path     = "path-to-datset/train"
test_path      = "path-to-datset/test"
valid_path     = "path-to-datset/valid"
path_pep       = "path-to-pep/pepdata"
path_hr1       = "./virus_genome/hr1"
```

then:

```
sh run_inference.sh
```

The pepdata folder is all 3027 SATPdb peptides (./data/pepdata) 



## Authors

** Nicolò Savioli, PhD **

Please if you find this code useful for all your research activities, cite it:

```
@misc{savioli2020oneshot,
    title={One-shot screening of potential peptide ligands on HR1 domain in COVID-19 glycosylated spike (S) protein with deep siamese network},
    author={Nicolò Savioli},
    year={2020},
    eprint={2004.02136},
    archivePrefix={arXiv},
    primaryClass={q-bio.QM}
}
```

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

This [license](./LICENSE.md) applies to all published OONI data.
