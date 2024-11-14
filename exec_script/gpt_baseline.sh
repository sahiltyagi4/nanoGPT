#!/bin/bash

# install tiktoken and hugging face datasets
cd /workspace/nonoGPT

outdir='/workspace/output/'
dir='/workspace/output/'
interface='eth0'
addr='172.17.0.2'
batchsize=32
worldsize=1
rank=1

procrank=$(($rank-1))
container='worker'$rank
echo '###### going to launch training for rank '$procrank ' on container '$container' with bsz '$batchsize

python3 -m run_nanoGPT --dir=$dir --interface=$interface --batch-size=$batchsize --world-size=$worldsize \
--master-addr=$addr --rank=$procrank --out-dir=$outdir &