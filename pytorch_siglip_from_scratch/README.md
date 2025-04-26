# PyTorch SigLIP From Scratch

This directory provides an implementation for training the SigLIP model from scratch using Distributed Data Parallel (DDP). It includes custom dataset handling, model components, and training utilities.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers library
- Additional dependencies: `numpy`, `pandas`, `tqdm`, `scikit-learn`, `Pillow`, `matplotlib`, `pyyaml`

## Dataset

<a href="https://www.kaggle.com/datasets/adityajn105/flickr8k" target="_blank">Flickr 8k Dataset</a> dataset is used for training and evaluation. The dataset consists of images and their corresponding captions in multiple languages. The dataset should be organized as follows:

flickr8k/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.txt

For captions translation, we used the <a href="https://github.com/AI4Bharat/IndicTrans2" target="_blank">IndicTrans2</a> model to translate the captions into Hindi, Marathi, and Hinglish.

## Configuration

The configuration file `config.yaml` contains all the necessary parameters for training. Key sections include:

- `model`: Model parameters such as encoder settings and projection dimensions
- `training`: Training parameters like batch size, epochs, optimizer, and scheduler settings
- `dataset`: Dataset-related configurations like image size and dataloader settings
- `languages`: List of supported languages and corresponding caption files
- `image_path`: Path to the directory containing images

## Training

To train the SigLIP model from scratch, run the `train.py` script:

python train.py --config config.yaml

For distributed training:

export Num_GPUS=2 # Set this to the number of GPUs you want to use
torchrun --nproc_per_node=$Num_GPUS --nnodes=1 --node_rank=0 train.py --config config.yaml

### Key Features

- **Distributed Training**: Utilizes PyTorch's Distributed Data Parallel (DDP) for multi-GPU training.
- **Custom Augmentations**: Includes cutout, color jitter, and random resized crop augmentations.
- **Custom Loss Function**: Implements a custom loss function for SigLIP.

## Evaluation

To evaluate the trained model, use the `test.py` script:

python test.py --config config.yaml --image <path_to_image> --labels <label1> <label2> ... --language <language>

### Example

python test.py --config config.yaml --image /path/to/image.jpg --labels "cat" "dog" "car" --language english

### Features

- **Multilingual Support**: Supports prompts in English, Hindi, Marathi, and Hinglish.
- **Visualization**: Displays the input image along with the predicted label and probability.
