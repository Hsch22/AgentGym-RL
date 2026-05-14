from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingDataCursor:
    epoch_idx: int
    batch_idx_in_epoch: int


def compute_steps_per_epoch(dataset_len: int, batch_size: int, drop_last: bool = True) -> int:
    if dataset_len < 0:
        raise ValueError(f"dataset_len must be non-negative, got {dataset_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if drop_last:
        return dataset_len // batch_size
    return (dataset_len + batch_size - 1) // batch_size


def cursor_from_completed_steps(completed_steps: int, steps_per_epoch: int) -> TrainingDataCursor:
    if completed_steps < 0:
        raise ValueError(f"completed_steps must be non-negative, got {completed_steps}")
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    return TrainingDataCursor(
        epoch_idx=completed_steps // steps_per_epoch,
        batch_idx_in_epoch=completed_steps % steps_per_epoch,
    )


def epoch_indices(dataset_len: int, epoch_idx: int, shuffle: bool, seed: int) -> list[int]:
    if dataset_len < 0:
        raise ValueError(f"dataset_len must be non-negative, got {dataset_len}")
    if epoch_idx < 0:
        raise ValueError(f"epoch_idx must be non-negative, got {epoch_idx}")
    if not shuffle:
        return list(range(dataset_len))
    if dataset_len == 0:
        return []

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = list(range(dataset_len))
    for _ in range(epoch_idx + 1):
        # Match torch.utils.data.RandomSampler exactly. With num_samples == len(dataset),
        # RandomSampler still consumes a second zero-length-sliced randperm each epoch.
        indices = torch.randperm(dataset_len, generator=generator).tolist()
        _ = torch.randperm(dataset_len, generator=generator)
    return indices


def batch_indices_at_cursor(
    dataset_len: int,
    batch_size: int,
    epoch_idx: int,
    batch_idx_in_epoch: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = True,
) -> list[int]:
    steps_per_epoch = compute_steps_per_epoch(dataset_len, batch_size, drop_last=drop_last)
    if steps_per_epoch <= 0:
        raise ValueError(
            f"dataset_len={dataset_len} and batch_size={batch_size} produce no batches with drop_last={drop_last}"
        )
    if batch_idx_in_epoch < 0 or batch_idx_in_epoch >= steps_per_epoch:
        raise ValueError(
            f"batch_idx_in_epoch must be in [0, {steps_per_epoch}), got {batch_idx_in_epoch}"
        )

    indices = epoch_indices(dataset_len, epoch_idx, shuffle=shuffle, seed=seed)
    start = batch_idx_in_epoch * batch_size
    end = start + batch_size
    return indices[start:end]
