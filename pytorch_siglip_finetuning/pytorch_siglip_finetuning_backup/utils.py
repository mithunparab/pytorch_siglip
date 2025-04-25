import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import AutoModel
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.optim.lr_scheduler import LambdaLR
import yaml
from types import SimpleNamespace
import warnings

# --- DDP Helper Functions ---
def setup_ddp(rank, world_size):
    """
    Sets up Distributed Data Parallel (DDP) environment.
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if not dist.is_initialized():
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup_ddp():
    """
    Cleans up the DDP environment.
    """
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

def is_dist_avail_and_initialized():
    """
    Checks if distributed training is available and initialized.
    """
    if not dist.is_available():
        return False
    try:
        return dist.is_initialized()
    except Exception:
        return False

def get_world_size():
    """
    Returns the world size in DDP.
    """
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank():
    """
    Returns the rank of the current process in DDP.
    """
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    """
    Checks if the current process is the main process.
    """
    return get_rank() == 0

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """
    Loads and validates the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        SimpleNamespace: Parsed configuration object.
    """
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)

        def dict_to_sns(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    d[k] = dict_to_sns(v)
                return SimpleNamespace(**d)
            elif isinstance(d, list):
                return [dict_to_sns(item) for item in d]
            else:
                return d

        config = dict_to_sns(config_dict)

        if not hasattr(config, 'device') or config.device == 'auto':
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dl_cfg = config.dataset.dataloader
        if dl_cfg.num_workers == 'auto':
            try:
                dl_cfg.num_workers = (os.cpu_count() // 2) if os.cpu_count() and os.cpu_count() > 1 else 2
            except NotImplementedError:
                dl_cfg.num_workers = 2
        elif not isinstance(dl_cfg.num_workers, int) or dl_cfg.num_workers < 0:
            dl_cfg.num_workers = 2

        if not hasattr(config, 'model_name') or not config.model_name:
            raise ValueError("Config missing required parameter 'model_name'")

        if not hasattr(config, 'languages') or not config.languages:
            raise ValueError("Config missing 'languages' list.")
        if not hasattr(config, 'captions_files'):
            raise ValueError("Config missing 'captions_files' object.")

        captions_files_dict = vars(config.captions_files)
        for lang in config.languages:
            if lang not in captions_files_dict:
                raise ValueError(f"Lang '{lang}' not in 'captions_files'.")
            if not os.path.isfile(captions_files_dict[lang]):
                warnings.warn(f"Caption file '{lang}' not found: '{captions_files_dict[lang]}'", UserWarning)

        if not os.path.exists(config.image_path):
            warnings.warn(f"Image path '{config.image_path}' not found.", UserWarning)

        return config
    except Exception as e:
        raise RuntimeError(f"Error loading/processing config {config_path}: {e}")

# --- Dataset ---
class CLIPDataset(Dataset):
    """
    Dataset for loading images and text pairs.

    Args:
        image_paths (list): List of image file paths.
        texts (list): List of text captions.
        config (SimpleNamespace): Configuration object.
    """
    def __init__(self, image_paths, texts, config):
        self.image_paths = image_paths
        self.texts = texts
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        text = str(self.texts[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            return image, text
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None
        except Exception:
            return None

# --- HF Model Wrapper ---
class SigLIPWrapper(nn.Module):
    """
    Wrapper for Hugging Face AutoModel for SigLIP fine-tuning.

    Args:
        model_name (str): Name of the Hugging Face model.
        config (SimpleNamespace): Configuration object.
    """
    def __init__(self, model_name, config):
        super().__init__()
        self.config = config
        try:
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}' from Hugging Face Hub: {e}")

    def forward(self, batch):
        """
        Forward pass through the model.

        Args:
            batch (dict): Dictionary containing input tensors.

        Returns:
            torch.Tensor: Logits per image.
        """
        try:
            outputs = self.model(**batch)
            return outputs.logits_per_image
        except Exception as e:
            raise RuntimeError(f"Error during forward pass: {e}")

# --- Loss Function ---
class SigLIPLoss(nn.Module):
    """
    Custom loss function for SigLIP.

    Args:
        logits (torch.Tensor): N x N logits tensor.

    Returns:
        torch.Tensor: Computed loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        n = logits.size(0)
        if n == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        lbl = 2 * torch.eye(n, device=logits.device) - 1
        loss_p = -torch.nn.functional.logsigmoid(lbl * logits)
        return torch.mean(torch.sum(loss_p, dim=1))

# --- LR Scheduler Helper ---
def get_lr_scheduler(optimizer, config):
    """
    Creates a learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer instance.
        config (SimpleNamespace): Configuration object.

    Returns:
        LambdaLR: Learning rate scheduler.
    """
    s_cfg = config.training.scheduler
    t_cfg = config.training
    o_cfg = config.training.optimizer
    lr_s = o_cfg.lr
    lr_m = lr_s * s_cfg.lr_max_mult * t_cfg.batch_size
    lr_min = s_cfg.lr_min
    tot_ep = t_cfg.epochs
    r_ep = s_cfg.lr_ramp_ep
    s_ep = s_cfg.lr_sus_ep
    d_ep = max(0, tot_ep - r_ep - s_ep)

    def lr_lambda(curr_ep):
        if curr_ep < r_ep:
            lr = lr_s + (curr_ep / (r_ep if r_ep > 0 else 1)) * (lr_m - lr_s)
        elif curr_ep < r_ep + s_ep:
            lr = lr_m
        else:
            if d_ep <= 0:
                lr = lr_min
            else:
                prog = (curr_ep - r_ep - s_ep) / d_ep
                cos_d = 0.5 * (1 + np.cos(np.pi * max(0., min(1., prog))))
                lr = lr_min + (lr_m - lr_min) * cos_d
        return lr / lr_s if lr_s > 1e-9 else 0.

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# --- Collate Function ---
def hf_collate_fn(batch, processor, max_length):
    """
    Collate function for processing batches using Hugging Face AutoProcessor.

    Args:
        batch (list): List of (PIL image, text) pairs.
        processor (transformers.AutoProcessor): Hugging Face processor.
        max_length (int): Maximum text sequence length.

    Returns:
        dict: Processed batch ready for the model.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images, texts = zip(*batch)

    try:
        inputs = processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        return inputs
    except Exception as e:
        raise RuntimeError(f"Error during collation: {e}")