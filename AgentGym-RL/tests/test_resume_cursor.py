import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler

_RESUME_CURSOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "utils"
    / "agent_dataset"
    / "resume_cursor.py"
)
_SPEC = importlib.util.spec_from_file_location("resume_cursor", _RESUME_CURSOR_PATH)
resume_cursor = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = resume_cursor
_SPEC.loader.exec_module(resume_cursor)

batch_indices_at_cursor = resume_cursor.batch_indices_at_cursor
compute_steps_per_epoch = resume_cursor.compute_steps_per_epoch
cursor_from_completed_steps = resume_cursor.cursor_from_completed_steps
epoch_indices = resume_cursor.epoch_indices


def _random_sampler_epoch_indices(dataset_len, seed, epochs):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(range(dataset_len), generator=generator)
    return [list(iter(sampler)) for _ in range(epochs)]


def test_shuffle_true_matches_random_sampler_across_epochs():
    dataset_len = 11
    seed = 7
    expected_epochs = _random_sampler_epoch_indices(dataset_len, seed, epochs=4)

    for epoch_idx, expected in enumerate(expected_epochs):
        assert epoch_indices(dataset_len, epoch_idx, shuffle=True, seed=seed) == expected


def test_resume_cursor_mid_epoch_and_epoch_boundary():
    steps_per_epoch = compute_steps_per_epoch(dataset_len=10, batch_size=3, drop_last=True)
    assert steps_per_epoch == 3

    mid_epoch = cursor_from_completed_steps(2, steps_per_epoch)
    epoch_boundary = cursor_from_completed_steps(3, steps_per_epoch)

    assert (mid_epoch.epoch_idx, mid_epoch.batch_idx_in_epoch) == (0, 2)
    assert (epoch_boundary.epoch_idx, epoch_boundary.batch_idx_in_epoch) == (1, 0)


def test_shuffle_true_resume_batch_does_not_repeat_previous_batch():
    dataset_len = 10
    batch_size = 3
    seed = 13
    previous_batch = batch_indices_at_cursor(dataset_len, batch_size, 0, 1, True, seed)
    resumed_batch = batch_indices_at_cursor(dataset_len, batch_size, 0, 2, True, seed)

    assert previous_batch != resumed_batch
    assert resumed_batch == epoch_indices(dataset_len, 0, True, seed)[6:9]


def test_shuffle_false_resume_batches_are_sequential():
    assert batch_indices_at_cursor(10, 4, 0, 0, shuffle=False, seed=1) == [0, 1, 2, 3]
    assert batch_indices_at_cursor(10, 4, 0, 1, shuffle=False, seed=1) == [4, 5, 6, 7]
    assert batch_indices_at_cursor(10, 4, 1, 0, shuffle=False, seed=1) == [0, 1, 2, 3]


def test_invalid_completed_training_cursor_is_rejected():
    with pytest.raises(ValueError):
        cursor_from_completed_steps(1, 0)
