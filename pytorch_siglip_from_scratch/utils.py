import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from transformers import DistilBertModel, DistilBertTokenizer
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
    Checks if DDP is available and initialized.
    """
    if not dist.is_available():
        return False
    try:
        if not dist.is_initialized():
            return False
    except Exception:
        return False
    return True


def get_world_size():
    """
    Returns the world size for DDP.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Returns the rank of the current process in DDP.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Checks if the current process is the main process.
    """
    return get_rank() == 0


# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """
    Loads and validates the configuration from a YAML file.
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

        # Device configuration
        if not hasattr(config, 'device') or config.device == 'auto':
            config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                _ = torch.device(config.device)
            except Exception:
                config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DataLoader workers
        dl_cfg = config.dataset.dataloader
        if dl_cfg.num_workers == 'auto':
            try:
                dl_cfg.num_workers = (os.cpu_count() // 2) if os.cpu_count() and os.cpu_count() > 1 else 2
            except NotImplementedError:
                dl_cfg.num_workers = 2
        elif not isinstance(dl_cfg.num_workers, int) or dl_cfg.num_workers < 0:
            dl_cfg.num_workers = 2

        # Image size validation
        img_cfg = config.dataset
        if isinstance(img_cfg.image_size, list) and len(img_cfg.image_size) == 2:
            img_cfg.image_size = tuple(img_cfg.image_size)
        else:
            raise ValueError("config.dataset.image_size must be a list of two integers [H, W]")

        # Multilingual captions validation
        if not hasattr(config, 'languages') or not isinstance(config.languages, list) or not config.languages:
            raise ValueError("Config missing 'languages' list or empty.")
        if not hasattr(config, 'captions_files') or not isinstance(config.captions_files, SimpleNamespace):
            raise ValueError("Config missing 'captions_files' object.")
        captions_files_dict = vars(config.captions_files)
        for lang in config.languages:
            if lang not in captions_files_dict:
                raise ValueError(f"Lang '{lang}' not in 'captions_files'.")
            lang_file = captions_files_dict[lang]
            if not os.path.isfile(lang_file):
                warnings.warn(f"Caption file '{lang}' not found: '{lang_file}'", UserWarning)

        # Image path validation
        if not os.path.exists(config.image_path):
            warnings.warn(f"Image path '{config.image_path}' not found.", UserWarning)

        return config
    except Exception as e:
        raise RuntimeError(f"Error loading/processing config {config_path}: {e}")


# --- Custom Augmentation ---
class Cutout(nn.Module):
    """
    Applies random cutout augmentation to an image tensor.
    """
    def __init__(self, min_size=0.02, max_size=0.15):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            return img
        if torch.rand(1).item() > 0.5:
            h, w = img.size(1), img.size(2)
            size_h = int(h * np.random.uniform(self.min_size, self.max_size))
            size_w = int(w * np.random.uniform(self.min_size, self.max_size))
            x1 = np.random.randint(0, max(1, w - size_w))
            y1 = np.random.randint(0, max(1, h - size_h))
            img_cutout = img.clone()
            try:
                img_cutout[:, y1:y1+size_h, x1:x1+size_w] = 0.0
            except Exception:
                pass
            return img_cutout
        return img


# --- Dataset ---
class CLIPDataset(Dataset):
    """
    Dataset for CLIP-style training with image-text pairs.
    """
    def __init__(self, image_paths, texts, config, augment=False):
        self.image_paths = image_paths
        self.texts = texts
        self.augment = augment
        self.config = config
        self.image_size = config.dataset.image_size

        self.tokenizer = DistilBertTokenizer.from_pretrained(config.model.text_encoder.preset)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize_transform = transforms.Resize(self.image_size)

        self.augment_transform = None
        self.cutout_transform = None
        if self.augment and config.training.use_augmentations:
            aug_params = config.augmentation
            pil_transforms = [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(aug_params.random_resized_crop.scale_min, aug_params.random_resized_crop.scale_max),
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=aug_params.random_horizontal_flip.p),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=aug_params.color_jitter.brightness,
                        contrast=aug_params.color_jitter.contrast,
                        saturation=aug_params.color_jitter.saturation,
                        hue=aug_params.color_jitter.hue
                    )
                ], p=0.8),
                transforms.RandomRotation(aug_params.random_rotation.degrees),
            ]
            self.augment_transform = transforms.Compose(pil_transforms)
            if hasattr(aug_params, 'cutout'):
                self.cutout_transform = Cutout(min_size=aug_params.cutout.min_size, max_size=aug_params.cutout.max_size)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        text = str(self.texts[idx])

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None

        try:
            if self.augment and self.augment_transform:
                image_processed_pil = self.augment_transform(image)
                image_tensor = self.to_tensor(image_processed_pil)
                if self.cutout_transform:
                    image_tensor = self.cutout_transform(image_tensor)
            else:
                image_resized_pil = self.resize_transform(image)
                image_tensor = self.to_tensor(image_resized_pil)

            if image_tensor.shape[1:] != self.image_size:
                image_resized_pil = self.resize_transform(image)
                image_tensor = self.to_tensor(image_resized_pil)

            image_processed = self.normalize(image_tensor)
        except Exception:
            return None

        try:
            inputs = self.tokenizer(
                text, max_length=self.config.dataset.text_sequence_length,
                padding="max_length", truncation=True, return_tensors="pt"
            )
        except Exception:
            return None

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {
            "images": image_processed,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


# --- Model Components ---
class ProjectionHead(nn.Module):
    """
    Projection head for embedding alignment.
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        p = self.projection(x)
        x = self.gelu(p)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + p
        return self.layer_norm(x)


class ImageEncoder(nn.Module):
    """
    Image encoder using EfficientNet.
    """
    def __init__(self, config):
        super().__init__()
        m_cfg = config.model.image_encoder
        p_cfg = config.model.projection
        w = EfficientNet_V2_S_Weights.DEFAULT if m_cfg.pretrained else None
        self.backbone = efficientnet_v2_s(weights=w)
        self.bb_out = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.projection = ProjectionHead(self.bb_out, p_cfg.embedding_dim, p_cfg.dropout)

    def forward(self, x):
        return self.projection(self.backbone(x))


class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT.
    """
    def __init__(self, config):
        super().__init__()
        m_cfg = config.model.text_encoder
        p_cfg = config.model.projection
        self.backbone = DistilBertModel.from_pretrained(m_cfg.preset)
        self.bb_out = self.backbone.config.hidden_size
        self.projection = ProjectionHead(self.bb_out, p_cfg.embedding_dim, p_cfg.dropout)

    def forward(self, input_ids, attention_mask):
        o = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).expand_as(o)
        s_emb = torch.sum(o * m, dim=1)
        s_mask = torch.clamp(m.sum(dim=1), min=1e-9)
        return self.projection(s_emb / s_mask)


class SigLIP(nn.Module):
    """
    SigLIP model combining image and text encoders.
    """
    def __init__(self, image_encoder, text_encoder, config):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.config = config
        sig_cfg = config.model.siglip
        self.register_buffer('logit_scale', torch.tensor(float(sig_cfg.logit_scale_init)))
        self.register_buffer('logit_bias', torch.tensor(float(sig_cfg.logit_bias_init)))

    def forward(self, batch):
        img_f = nn.functional.normalize(self.image_encoder(batch["images"]), dim=-1)
        txt_f = nn.functional.normalize(self.text_encoder(batch["input_ids"], batch["attention_mask"]), dim=-1)
        logits = img_f @ txt_f.T
        return self.logit_scale * logits + self.logit_bias


# --- Loss Function ---
class SigLIPLoss(nn.Module):
    """
    Custom loss function for SigLIP.
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
    Returns a learning rate scheduler based on the configuration.
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
def collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        if "stack expects each tensor to be equal size" in str(e):
            shapes = {}
            for i, item in enumerate(batch):
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        if key not in shapes:
                            shapes[key] = []
                        shapes[key].append((i, value.shape))
            for key, shape_list in shapes.items():
                if len(set(s[1] for s in shape_list)) > 1:
                    print(f"ERROR during collation: Inconsistent shapes for key '{key}':")
                    for idx, shape in shape_list:
                        print(f"  Item index {idx}: {shape}")
                    break
        return None
    except Exception as e:
        print(f"ERROR during collation: {e}")
        return None
