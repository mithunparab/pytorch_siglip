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
from pathlib import Path
from utils import (
    load_config, CLIPDataset, ImageEncoder, TextEncoder, SigLIP,
    SigLIPLoss, get_lr_scheduler, collate_fn,
    setup_ddp, cleanup_ddp, get_rank, get_world_size, is_main_process,
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
            continue
        try:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        except Exception:
            continue

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch)
            loss = criterion(logits)

        if torch.isnan(loss) or torch.isinf(loss):
            if is_main_process():
                warnings.warn(f"NaN/Inf loss step {step+1}. Skipping.", UserWarning)
            if pbar:
                pbar.update(1)
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

    if pbar:
        pbar.close()
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

    if pbar:
        pbar.close()
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
    if is_main_process():
        print(f"Seed set for rank {rank}: {seed}")

    # Load multilingual data
    all_dfs = []
    captions_files_dict = vars(config.captions_files)
    if is_main_process():
        print("Loading multilingual data...")
    if not os.path.exists(config.image_path):
        if is_main_process():
            print(f"\nERROR: Image directory not found at '{config.image_path}'")
        cleanup_ddp()
        return

    for lang in config.languages:
        captions_file = captions_files_dict.get(lang)
        if not captions_file or not os.path.isfile(captions_file):
            if is_main_process():
                warnings.warn(f"Caption file for '{lang}' invalid or missing. Skipping.", UserWarning)
            continue
        try:
            df_lang = pd.read_csv(captions_file)
            if 'image' not in df_lang.columns or 'caption' not in df_lang.columns:
                if is_main_process():
                    warnings.warn(f"Caption file for '{lang}' ({captions_file}) missing 'image' or 'caption' column. Skipping.", UserWarning)
                continue

            df_lang["image_path"] = df_lang["image"].apply(lambda x: os.path.join(config.image_path, x))
            df_lang["language"] = lang
            all_dfs.append(df_lang)
            if is_main_process():
                print(f"  -> Loaded {len(df_lang)} captions for '{lang}'")

        except Exception as e:
            if is_main_process():
                warnings.warn(f"Error reading/processing captions file for '{lang}' ({captions_file}): {e}. Skipping.", UserWarning)
            continue

    if not all_dfs:
        if is_main_process():
            print("\nERROR: No valid caption data loaded. Check config paths and file contents.")
        cleanup_ddp()
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    if is_main_process():
        print(f"Total combined captions loaded: {len(full_df)}")

    # Data splitting
    gkf = GroupKFold(n_splits=5)
    full_df['fold'] = -1
    for fold, (_, valid_idx) in enumerate(gkf.split(full_df, groups=full_df["image"])):
        full_df.loc[valid_idx, 'fold'] = fold
    train_df = full_df[full_df['fold'] != 0].reset_index(drop=True)
    valid_df = full_df[full_df['fold'] == 0].reset_index(drop=True)
    if is_main_process():
        print(f"  -> Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")

    # Create datasets
    try:
        train_ds = CLIPDataset(train_df["image_path"].values, train_df["caption"].values, config, augment=True)
        valid_ds = CLIPDataset(valid_df["image_path"].values, valid_df["caption"].values, config, augment=False)
    except Exception as e:
        if is_main_process():
            print(f"\nERROR creating Datasets: {e}")
        cleanup_ddp()
        return

    # Create DDP DataLoaders
    dl_cfg = config.dataset.dataloader
    train_batch_size = config.training.batch_size
    valid_batch_size = int(train_batch_size * 1.5)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, sampler=train_sampler,
                              num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
                              drop_last=dl_cfg.drop_last_train, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=valid_batch_size, sampler=valid_sampler,
                              num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
                              drop_last=False, collate_fn=collate_fn)

    # Initialize model
    try:
        image_encoder = ImageEncoder(config).to(device)
        text_encoder = TextEncoder(config).to(device)
        model = SigLIP(image_encoder, text_encoder, config).to(device)
    except Exception as e:
        if is_main_process():
            print(f"\nERROR initializing model: {e}")
        cleanup_ddp()
        return

    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])

    # Setup Loss, Optimizer, Scheduler
    criterion = SigLIPLoss().to(device)
    optimizer = optim.AdamW(model.module.parameters(), lr=config.training.optimizer.lr, weight_decay=config.training.optimizer.weight_decay)
    scheduler = get_lr_scheduler(optimizer, config)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training Loop
    best_loss = float('inf')
    for epoch in range(config.training.epochs):
        epoch_lr = optimizer.param_groups[0]['lr']
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
    parser = argparse.ArgumentParser(description="Train SigLIP model using DDP.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank (set by launcher).")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"FATAL Error loading config '{args.config}': {e}")
        exit(1)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if rank == -1 or local_rank == -1:
            print("Error: DDP env vars not set. Use torchrun.")
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