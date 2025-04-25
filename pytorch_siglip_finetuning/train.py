import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import argparse
import warnings
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor
from pathlib import Path
from utils import (
    load_config, CLIPDataset, SigLIPWrapper, SigLIPLoss, get_lr_scheduler,
    hf_collate_fn, setup_ddp, cleanup_ddp, get_rank, get_world_size, is_main_process,
    is_dist_avail_and_initialized
)

def train_one_epoch(model, criterion, optimizer, scaler, train_loader, train_sampler, device, epoch, config, use_amp):
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: Gradient scaler for mixed precision.
        train_loader: DataLoader for training data.
        train_sampler: Distributed sampler for training data.
        device: Device to use for training.
        epoch: Current epoch number.
        config: Configuration object.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    train_loss_accum = torch.tensor(0.0).to(device)
    processed_batches = 0
    if is_dist_avail_and_initialized():
        train_sampler.set_epoch(epoch)

    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1} [Train]", leave=False) if is_main_process() else None

    for step, batch in enumerate(train_loader):
        if batch is None:
            if pbar: pbar.update(1)
            continue
        try:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        except Exception as e:
            if pbar: pbar.update(1)
            continue

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)
            loss = criterion(logits)

        if torch.isnan(loss) or torch.isinf(loss):
            if pbar: pbar.update(1)
            continue

        scaler.scale(loss).backward()
        if config.training.gradient_clipping > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), config.training.gradient_clipping)
        scaler.step(optimizer)
        scaler.update()

        train_loss_accum += loss.detach()
        processed_batches += 1
        if pbar:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

    if pbar: pbar.close()
    if is_dist_avail_and_initialized():
        dist.all_reduce(train_loss_accum, op=dist.ReduceOp.SUM)

    total_processed_batches = processed_batches * get_world_size() if is_dist_avail_and_initialized() else processed_batches
    avg_train_loss = train_loss_accum.item() / total_processed_batches if total_processed_batches > 0 else 0.0
    return avg_train_loss

@torch.no_grad()
def validate_one_epoch(model, criterion, valid_loader, device, epoch, config, use_amp):
    """
    Validate the model for one epoch.

    Args:
        model: The model to validate.
        criterion: Loss function.
        valid_loader: DataLoader for validation data.
        device: Device to use for validation.
        epoch: Current epoch number.
        config: Configuration object.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Average validation loss for the epoch.
    """
    model.eval()
    val_loss_accum = torch.tensor(0.0).to(device)
    val_batches = 0
    pbar = tqdm(total=len(valid_loader), desc=f"Epoch {epoch+1} [Valid]", leave=False) if is_main_process() else None

    for batch in valid_loader:
        if batch is None:
            continue
        try:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        except Exception:
            continue

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)
            loss = criterion(logits)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            val_loss_accum += loss.detach()
            val_batches += 1
        if pbar:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)

    if pbar: pbar.close()
    if is_dist_avail_and_initialized():
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)

    total_val_batches = val_batches * get_world_size() if is_dist_avail_and_initialized() else val_batches
    avg_val_loss = val_loss_accum.item() / total_val_batches if total_val_batches > 0 else float('inf')
    return avg_val_loss

def main_worker(rank, world_size, config):
    """
    Main function for each DDP process.

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
        config: Configuration object.
    """
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    if is_main_process():
        print(f"Running DDP on {world_size} GPUs. Process {rank} on {device}.")

    seed = config.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_dfs = []
    captions_files_dict = vars(config.captions_files)
    if not os.path.exists(config.image_path):
        cleanup_ddp()
        return
    for lang in config.languages:
        captions_file = captions_files_dict.get(lang)
        if not captions_file or not os.path.isfile(captions_file):
            continue
        try:
            df_lang = pd.read_csv(captions_file)
            if 'image' not in df_lang.columns or 'caption' not in df_lang.columns:
                continue
            df_lang["image_path"] = df_lang["image"].apply(lambda x: os.path.join(config.image_path, x))
            df_lang["language"] = lang
            all_dfs.append(df_lang)
        except Exception:
            continue
    if not all_dfs:
        cleanup_ddp()
        return
    full_df = pd.concat(all_dfs, ignore_index=True)

    gkf = GroupKFold(n_splits=5)
    full_df['fold'] = -1
    for fold, (_, valid_idx) in enumerate(gkf.split(full_df, groups=full_df["image"])):
        full_df.loc[valid_idx, 'fold'] = fold
    train_df = full_df[full_df['fold'] != 0].reset_index(drop=True)
    valid_df = full_df[full_df['fold'] == 0].reset_index(drop=True)

    train_ds = CLIPDataset(train_df["image_path"].values, train_df["caption"].values, config)
    valid_ds = CLIPDataset(valid_df["image_path"].values, valid_df["caption"].values, config)

    processor = AutoProcessor.from_pretrained(config.model_name)

    dl_cfg = config.dataset.dataloader
    train_batch_size = config.training.batch_size
    valid_batch_size = int(train_batch_size * 1.5)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)

    collate_func = lambda batch: hf_collate_fn(batch, processor, config.dataset.text_sequence_length)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, sampler=train_sampler,
                              num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
                              drop_last=dl_cfg.drop_last_train, collate_fn=collate_func)
    valid_loader = DataLoader(valid_ds, batch_size=valid_batch_size, sampler=valid_sampler,
                              num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
                              drop_last=False, collate_fn=collate_func)

    model = SigLIPWrapper(config.model_name, config).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    criterion = SigLIPLoss().to(device)
    optimizer = optim.AdamW(model.module.parameters(), lr=config.training.optimizer.lr, weight_decay=config.training.optimizer.weight_decay)
    scheduler = get_lr_scheduler(optimizer, config)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_loss = float('inf')

    for epoch in range(config.training.epochs):
        avg_train_loss = train_one_epoch(model, criterion, optimizer, scaler, train_loader, train_sampler, device, epoch, config, use_amp)
        avg_val_loss = validate_one_epoch(model, criterion, valid_loader, device, epoch, config, use_amp)
        scheduler.step()

        if is_main_process():
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_path = Path(config.model_save_path)
                if save_path.parent != Path("."):
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.module.state_dict(), save_path)

        if is_dist_avail_and_initialized():
            dist.barrier()

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SigLIP model using DDP and Hugging Face.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank (set by launcher).")
    args = parser.parse_args()

    config = load_config(args.config)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if rank == -1 or local_rank == -1:
            exit(1)
        main_worker(rank=rank, world_size=world_size, config=config)
    elif torch.cuda.is_available():
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        main_worker(rank=0, world_size=1, config=config)
    else:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        main_worker(rank=0, world_size=1, config=config)
