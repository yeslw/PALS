#!/bin/bash

MODEL="/2T/zhuhe/data/decapoda-research-llama-7B-hf"

for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model $MODEL \
  --nsamples 128 \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --prune_method ${method} \
  --save out/llama_7b/unstructured/$method/
done 

for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model $MODEL \
  --nsamples 128 \
  --sparsity_ratio 0.5 \
  --sparsity_type 4:8 \
  --prune_method ${method} \
  --save out/llama_7b/4_8/$method/
done 

for method in ablate_mag_seq ablate_wanda_seq ablate_mag_iter ablate_wanda_iter 
do 
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model $MODEL \
  --nsamples 128 \
  --sparsity_ratio 0.5 \
  --sparsity_type 2:4 \
  --prune_method ${method} \
  --save out/llama_7b/2_4/$method/
done
