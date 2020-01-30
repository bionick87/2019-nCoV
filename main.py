#http://biopython.org/DIST/docs/tutorial/Tutorial.html
#https://towardsdatascience.com/pairwise-sequence-alignment-using-biopython-d1a9d0ba861f

from Bio.Seq import Seq
from Bio.Cluster import kcluster 
from Bio.Cluster import somcluster 
import numpy as np 
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import time
import multiprocessing 

def readGenSeq(path_file):
    with open(path_file, 'r') as f:
        data = f.read()
    return data

def multiprocessing_func(nCoV_seq,hiv_seq):
    alignments     = pairwise2.align.globalxx(nCoV_seq,hiv_seq)
    oh             = open("./out/out.txt", "w")
    for a in alignments:
        oh.write(format_alignment(*a))

if __name__ == "__main__":
    # https://www.ncbi.nlm.nih.gov/nuccore/MN908947.3?report=fasta
    path_file_nCoV = "/mnt/storage/home/nsavioli/2019-nCoV/virus_genome/2019-nCoV.txt"
    # https://www.ncbi.nlm.nih.gov/nuccore/NC_001722.1?report=fasta
    path_file_hiv  = "/mnt/storage/home/nsavioli/2019-nCoV/virus_genome/HIV.txt"
    path_save      = "/mnt/storage/home/nsavioli/out/result.txt"
    nCoV_seq_str   = readGenSeq(path_file_nCoV)
    HIV_seq_str    = readGenSeq(path_file_hiv)
    nCoV_seq_str  = "ACGGGT"
    HIV_seq_str   = "ACG"
    n_process      = 10
    nCoV_seq       = Seq(nCoV_seq_str)
    hiv_seq        = Seq(HIV_seq_str)
    starttime      = time.time()
    processes      = []
    for i in range(0,n_process):
        p = multiprocessing.Process(target=multiprocessing_func, args=(nCoV_seq,hiv_seq))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    print('That took {} seconds'.format(time.time() - starttime))










