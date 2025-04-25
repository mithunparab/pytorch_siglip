# PyTorch SigLIP Fine-Tuning and Zero-Shot Classification

This repository provides a framework for fine-tuning the SigLIP model using Distributed Data Parallel (DDP) and performing zero-shot classification with multilingual support.

## Project Structure

```
.
├── config.yaml            # Configuration file for training and inference
├── train.py               # Script for fine-tuning the SigLIP model
├── test.py                # Script for zero-shot classification
├── utils.py               # Utility functions and classes
├── README.md              # Documentation for the project
└── run.ipynb              # Jupyter Notebook for interactive experimentation
```


## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers library
- Additional dependencies: `numpy`, `pandas`, `tqdm`, `scikit-learn`, `Pillow`, `matplotlib`, `pyyaml`

## Dataset
<a href="https://www.kaggle.com/datasets/adityajn105/flickr8k" target="_blank">Flickr 8k Dataset</a> dataset is used for training and evaluation. The dataset consists of images and their corresponding captions in multiple languages. The dataset should be organized as follows:
```
flickr8k/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.txt
```

For captions translation, we used the <a href="https://github.com/AI4Bharat/IndicTrans2" target="_blank">IndicTrans2</a> model to translate the captions into Hindi, Marathi, and Hinglish.

## Configuration

The configuration file `config.yaml` contains all the necessary parameters for training and inference. Key sections include:

- `model_name`: Hugging Face model name (e.g., `google/siglip-base-patch16-256-multilingual`)
- `training`: Training parameters such as batch size, epochs, optimizer, and scheduler settings
- `dataset`: Dataset-related configurations like text sequence length and dataloader settings
- `languages`: List of supported languages and corresponding caption files
- `image_path`: Path to the directory containing images

Supported languages include English, Hindi, Marathi, and Hinglish.
## Training

To fine-tune the SigLIP model, run the `train.py` script:

```bash
python train.py --config config.yaml
```
or for distributed training:

```bash
export Num_GPUS=2 # Set this to the number of GPUs you want to use
torchrun --nproc_per_node=$Num_GPUS --nnodes=1 --node_rank=0  train.py --config config.yaml
```

### Key Features

- **Distributed Training**: Utilizes PyTorch's Distributed Data Parallel (DDP) for multi-GPU training.
- **Gradient Clipping**: Prevents exploding gradients during training.
- **Custom Loss Function**: Implements a custom loss function for SigLIP.

## Zero-Shot Classification

To perform zero-shot classification, use the `test.py` script:

```bash
python test.py --config config.yaml --image <path_to_image> --labels <label1> <label2> ... --language <language>
```

### Example

```bash
python test.py --config config.yaml --image /path/to/image.jpg --labels "cat" "dog" "car" --language english
```

### Features

- **Multilingual Support**: Supports prompts in English, Hindi, Marathi, and Hinglish.
- **Visualization**: Displays the input image along with the predicted label and probability.
