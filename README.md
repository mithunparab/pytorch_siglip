
# PyTorch SigLIP

This repository contains implementations and experiments for training and fine-tuning the SigLIP model, a multimodal model for image-text tasks. The project is divided into two main directories:

- **`pytorch_siglip_finetuning/`**: Contains scripts and utilities for fine-tuning a pre-trained SigLIP model using Distributed Data Parallel (DDP). Includes support for multilingual captions and zero-shot classification.

- **`pytorch_siglip_from_scratch/`**: Provides an implementation for training the SigLIP model from scratch using Distributed Data Parallel (DDP), including custom dataset handling, model components, and training utilities.

## Dataset

<a href="https://www.kaggle.com/datasets/adityajn105/flickr8k" target="_blank">Flickr 8k Dataset</a> dataset is used for training and evaluation. The dataset consists of images and their corresponding captions in multiple languages. The dataset should be organized as follows:

flickr8k/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.txt

For captions translation, we used the <a href="https://github.com/AI4Bharat/IndicTrans2" target="_blank">IndicTrans2</a> model to translate the captions into Hindi, Marathi, and Hinglish.

---
Refer to the individual `README.md` files in each directory for detailed documentation.
