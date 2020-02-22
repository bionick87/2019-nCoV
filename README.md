# AI against COVID-2019


## Purpose 

Deep learning application to track COVID-2019 virus evolution by analyzing the RNA sequence between two COVID-2019 samples at different time points.
The network is based on a deep Siamese Neural Network (https://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) that takes a strand of COVID-2019 sequences as input. More specifically, each strand is converted into a grayscale image of 256x256 pixel size and passed through two convolutions convolutional networks. However, the net is trained towards negative examples with Human Immunodeficiency Viruses (HIV) strand examples.
The aim is to find nucleotide sequences that stably characterizes COVID-2019 for finding a specific RNA biomarker strand.

# Wuhan-Hu-1 isolated coronavirus 2 complete genome sequences have been downloaded from link below and saved in the project folder ./virus_genome

https://www.ncbi.nlm.nih.gov/nuccore/MN908947.3?report=fasta

# The Human Immunodeficiency Viruses (HIV) complete genome sequence has been downloaded from link below and saved in the project folder ./virus_genome

https://www.ncbi.nlm.nih.gov/nuccore/NC_001722.1?report=fasta






The code is released for research purposes only and not for commercial purposes.

![alt text](img/img.jpg)


## USE

* 


## TODO
  * cluster testing with pairwise 
  * Convert into nucleotide bases in an image to pass it towards a convolutional network.


### Prerequisites

Before getting started, it's important to have a working environment with all dependencies satisfied. For this, we recommend using the Anaconda distribution of Python 3.5.

```
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

So pytorch must be installed, please make sure that cuDNN is installed correctly (https://developer.nvidia.com/cudnn).

```
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

Then install the following libraries:

```
pip install torchvision
pip install opencv-python
pip install matplotlib
pip install tqdm
```


### Installing

To start the code: 

```
python main.py
```

## Authors

* ** Nicolò Savioli, PhD ** - *Initial work* 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc