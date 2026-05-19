# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
github link-https://github.com/Chandaluri-Sathwik/da6401_assignment_3.git
wandb link-https://wandb.ai/sai-sathwik/da6401-a3/reports/DA6401-Assignment-3-ME23B238--VmlldzoxNjkwNjY4NA?accessToken=hosko9mdo4octd9218cqh996i7bws467apvw93nznn7miqu2qgv18fmkq8xambu7