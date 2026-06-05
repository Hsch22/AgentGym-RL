# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import math
import random
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.agent_trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.agent_dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.agent_dataset.resume_cursor import (
    compute_steps_per_epoch,
    cursor_from_completed_steps,
    epoch_indices,
)
from abc import ABC, abstractmethod

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def find_latest_ckpt_path_aistudio(path, directory_format="global_step_{}"):
    if path is None:
        return None

    from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename
    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        print("Checkpoint tracker file does not exist: %s", tracker_file)
        return None

    from aistudio_checkpoint.aistudio_base_checkpointer import load_checkpoint
    with open(tracker_file, "r") as f:
        iteration, resuming_path = f.read().split("\n")
    ckpt_path = os.path.join(load_checkpoint(resuming_path=resuming_path), directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch['response_mask']

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # === LOCAL CHANGE：接收 MClaw 已生成的 advantages/returns ===
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        response_mask = data.batch['response_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        response_mask = data.batch['response_mask']
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        response_mask = data.batch['response_mask']

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'mclaw':
        # MClaw 在树搜索 rollout 内生成 advantage，这里只校验并补齐 returns。
        if 'advantages' not in data.batch.keys():
            raise ValueError("MClaw rollout must populate advantages in DataProto")
        if 'returns' not in data.batch.keys():
            data.batch['returns'] = data.batch['advantages'].clone()
    else:
        raise NotImplementedError
    return data


class RoundsScheduler(ABC):
    @abstractmethod
    def step(self):
        raise NotImplementedError
    
    @abstractmethod
    def set_global_steps(self, global_steps: int):
        raise NotImplementedError

    @abstractmethod
    def get_rounds(self):
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict):
        raise NotImplementedError
    

class FixedRoundsScheduler(RoundsScheduler):
    def __init__(self, rounds: int):
        self.max_rounds = rounds

    def step(self):
        pass

    def set_global_steps(self, global_steps: int):
        pass

    def get_rounds(self):
        return self.max_rounds

    def state_dict(self):
        return {
            'type': 'fixed',
            'max_rounds': self.max_rounds,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get('type') != 'fixed':
            raise ValueError(f"Cannot load rounds scheduler state with type={state_dict.get('type')} into fixed scheduler")
        if int(state_dict['max_rounds']) != int(self.max_rounds):
            raise ValueError(
                f"Fixed rounds mismatch: checkpoint={state_dict['max_rounds']} current={self.max_rounds}"
            )


class StepRoundsScheduler(RoundsScheduler):
    def __init__(self, steps_scaling_inter: int, rounds_ls: List[int]):
        self.rounds_ls = rounds_ls
        self.steps_scaling_inter = steps_scaling_inter
        self.max_rounds = rounds_ls[0]
        self.current_stage = 0
        self.global_steps = 1 # start from 1

    def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps
        if (self.global_steps // self.steps_scaling_inter < len(self.rounds_ls)):
            self.current_stage = self.global_steps // self.steps_scaling_inter
        else:
            self.current_stage = len(self.rounds_ls) - 1
        self.max_rounds = self.rounds_ls[self.current_stage]
    
    def step(self):
        if self.current_stage + 1 < len(self.rounds_ls) and self.global_steps % self.steps_scaling_inter == 0:
            self.current_stage += 1
            self.max_rounds = self.rounds_ls[self.current_stage]
        self.global_steps += 1

    def get_rounds(self):
        return self.max_rounds

    def state_dict(self):
        return {
            'type': 'scaling_inter_stepwise',
            'rounds_ls': list(self.rounds_ls),
            'steps_scaling_inter': self.steps_scaling_inter,
            'max_rounds': self.max_rounds,
            'current_stage': self.current_stage,
            'global_steps': self.global_steps,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get('type') != 'scaling_inter_stepwise':
            raise ValueError(
                f"Cannot load rounds scheduler state with type={state_dict.get('type')} "
                "into scaling_inter_stepwise scheduler"
            )
        if list(state_dict['rounds_ls']) != list(self.rounds_ls):
            raise ValueError(f"Rounds schedule mismatch: checkpoint={state_dict['rounds_ls']} current={self.rounds_ls}")
        if int(state_dict['steps_scaling_inter']) != int(self.steps_scaling_inter):
            raise ValueError(
                "Rounds scheduler interval mismatch: "
                f"checkpoint={state_dict['steps_scaling_inter']} current={self.steps_scaling_inter}"
            )
        self.global_steps = int(state_dict['global_steps'])
        self.current_stage = int(state_dict['current_stage'])
        self.max_rounds = int(state_dict['max_rounds'])


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


ROLLOUT_MONITOR_KEY = "rollout_monitor"
TRAINER_STATE_NAME = "trainer_state.pt"
DATA_STATE_NAME = "data.pt"


def _atomic_write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, 'w') as f:
        f.write(text)
    os.replace(tmp_path, path)


def _atomic_torch_save(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _torch_load(path: str, **kwargs):
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _parse_global_step_from_path(global_step_folder: str) -> int:
    folder_name = os.path.basename(os.path.normpath(global_step_folder))
    if not folder_name.startswith('global_step_'):
        raise ValueError(f"Checkpoint folder must be named global_step_N, got {global_step_folder}")
    return int(folder_name.split('global_step_')[-1])


def _safe_ratio(numerator, denominator):
    # === LOCAL CHANGE：rollout 有效性指标的安全除法辅助函数。 ===
    return float(numerator) / float(denominator) if denominator else 0.0


def _collect_rollout_monitor_totals(data: DataProto):
    # === LOCAL CHANGE：聚合非张量形式的 rollout monitor 记录。 ===
    # rollout monitor 是非张量 batch，需在 driver 侧汇总后再进日志系统。
    records = data.non_tensor_batch.get(ROLLOUT_MONITOR_KEY)
    totals = {}
    if records is None:
        return totals
    for record in records.reshape(-1):
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if key.endswith("_max"):
                    totals[key] = max(totals.get(key, 0.0), float(value))
                else:
                    totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def compute_rollout_monitor_metrics(data: DataProto, prefix="rollout"):
    # === LOCAL CHANGE：将 rollout monitor 计数转换为日志指标。 ===
    totals = _collect_rollout_monitor_totals(data)
    if not totals:
        return {}

    metrics = {}

    def add_bucket(metric_prefix, source_prefix=""):
        candidate_total = totals.get(f"{source_prefix}candidate_total", 0.0)
        candidate_string_valid = totals.get(f"{source_prefix}candidate_string_valid", 0.0)
        taken_total = totals.get(f"{source_prefix}taken_total", 0.0)
        taken_string_valid = totals.get(f"{source_prefix}taken_string_valid", 0.0)
        taken_env_valid = totals.get(f"{source_prefix}taken_env_valid", 0.0)
        string_valid_env_invalid = totals.get(f"{source_prefix}string_valid_env_invalid", 0.0)
        raw_chars_sum = totals.get(f"{source_prefix}taken_action_raw_chars_sum", 0.0)
        raw_chars_max = totals.get(f"{source_prefix}taken_action_raw_chars_max", 0.0)
        norm_chars_sum = totals.get(f"{source_prefix}taken_action_norm_chars_sum", 0.0)
        norm_chars_max = totals.get(f"{source_prefix}taken_action_norm_chars_max", 0.0)

        metrics.update({
            f"{metric_prefix}/candidate_total_count": candidate_total,
            f"{metric_prefix}/candidate_string_valid_count": candidate_string_valid,
            f"{metric_prefix}/candidate_string_valid_ratio": _safe_ratio(candidate_string_valid, candidate_total),
            f"{metric_prefix}/taken_total_count": taken_total,
            f"{metric_prefix}/taken_string_valid_count": taken_string_valid,
            f"{metric_prefix}/taken_env_valid_count": taken_env_valid,
            f"{metric_prefix}/string_valid_env_invalid_count": string_valid_env_invalid,
            f"{metric_prefix}/taken_string_valid_ratio": _safe_ratio(taken_string_valid, taken_total),
            f"{metric_prefix}/taken_env_valid_ratio": _safe_ratio(taken_env_valid, taken_total),
            f"{metric_prefix}/string_valid_env_invalid_ratio": _safe_ratio(string_valid_env_invalid, taken_total),
            f"{metric_prefix}/taken_action_raw_chars_mean": _safe_ratio(raw_chars_sum, taken_total),
            f"{metric_prefix}/taken_action_raw_chars_max": raw_chars_max,
            f"{metric_prefix}/taken_action_norm_chars_mean": _safe_ratio(norm_chars_sum, taken_total),
            f"{metric_prefix}/taken_action_norm_chars_max": norm_chars_max,
        })

    add_bucket(prefix)
    add_bucket(f"{prefix}/round0", "round0_")
    add_bucket(f"{prefix}/later", "later_")
    later_clustered_action = totals.get("later_clustered_action", 0.0)
    later_skipped_action = totals.get("later_skipped_action", 0.0)
    later_scheduled_action = later_clustered_action + later_skipped_action
    metrics.update({
        f"{prefix}/later_clustered_action_count": later_clustered_action,
        f"{prefix}/later_skipped_action_count": later_skipped_action,
        f"{prefix}/later_clustered_action_ratio": _safe_ratio(later_clustered_action, later_scheduled_action),
        f"{prefix}/later_skipped_action_ratio": _safe_ratio(later_skipped_action, later_scheduled_action),
    })
    return metrics


def drop_rollout_monitor(data: DataProto):
    # === LOCAL CHANGE：在 PPO batch 处理前移除 monitor 负载。 ===
    # monitor 只用于指标统计，删除后避免进入 PPO batch 分片和 actor update。
    data.non_tensor_batch.pop(ROLLOUT_MONITOR_KEY, None)


def _parse_g2rl_stop_after_steps(g2rl_config):
    stop_after_steps = g2rl_config.get('stop_after_steps', None)
    if stop_after_steps is None:
        return None
    if isinstance(stop_after_steps, str):
        normalized = stop_after_steps.strip().lower()
        if normalized in ('', 'none', 'null'):
            return None
        try:
            return int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"algorithm.g2rl.stop_after_steps must be null or an integer, got {stop_after_steps!r}"
            ) from exc
    if isinstance(stop_after_steps, bool):
        raise ValueError(f"algorithm.g2rl.stop_after_steps must be null or an integer, got {stop_after_steps!r}")
    if isinstance(stop_after_steps, float) and not stop_after_steps.is_integer():
        raise ValueError(f"algorithm.g2rl.stop_after_steps must be null or an integer, got {stop_after_steps!r}")
    try:
        return int(stop_after_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"algorithm.g2rl.stop_after_steps must be null or an integer, got {stop_after_steps!r}"
        ) from exc


def is_g2rl_active(g2rl_config, global_steps: int) -> bool:
    if not bool(g2rl_config.get('enabled', False)):
        return False
    stop_after_steps = _parse_g2rl_stop_after_steps(g2rl_config)
    return stop_after_steps is None or stop_after_steps <= 0 or int(global_steps) <= stop_after_steps


def _g2rl_stop_after_steps_metric(g2rl_config) -> float:
    stop_after_steps = _parse_g2rl_stop_after_steps(g2rl_config)
    return float(stop_after_steps) if stop_after_steps is not None else 0.0


def get_g2rl_novelty_torch_dtype(g2rl_config):
    novelty_dtype = str(g2rl_config.get('novelty_dtype', 'float64')).strip().lower()
    if novelty_dtype == 'float32':
        return torch.float32
    if novelty_dtype == 'float64':
        return torch.float64
    raise ValueError(f"algorithm.g2rl.novelty_dtype must be 'float32' or 'float64', got {novelty_dtype!r}")


def apply_g2rl_reward_shaping(data: DataProto, g2rl_config):
    """Apply paper-style G2RL trajectory reward shaping before GRPO standardization."""
    if 'g2rl_features' not in data.batch.keys():
        raise ValueError("G2RL reward shaping requires g2rl_features in the batch")

    token_level_rewards = data.batch['token_level_rewards']
    token_level_scores = data.batch['token_level_scores']
    response_mask = data.batch['response_mask']
    features = data.batch.pop('g2rl_features')
    index = data.non_tensor_batch['uid']

    device = token_level_rewards.device
    novelty_torch_dtype = get_g2rl_novelty_torch_dtype(g2rl_config)
    base_scores = token_level_scores.sum(dim=-1).float()
    if bool(g2rl_config.get('zero_one_to_signed', True)):
        signed_scores = base_scores * 2.0 - 1.0
    else:
        signed_scores = base_scores

    lambda_coef = float(g2rl_config.get('lambda_coef', 1.0))
    reward_clip = float(g2rl_config.get('reward_clip', 3.0))
    eps = float(g2rl_config.get('eps', 1e-6))
    skip_all_failed_groups = bool(g2rl_config.get('skip_all_failed_groups', False))
    success_only_novelty = bool(g2rl_config.get('success_only_novelty', False))
    success_mask = base_scores > 0.0

    shaped_scores = signed_scores.clone()
    novelty_scores = torch.zeros_like(signed_scores)
    factor_scores = torch.ones_like(signed_scores)
    group_sizes = []
    skipped_all_failed_group_count = 0
    shaped_group_count = 0
    shaped_success_only_count = 0
    shaped_row_count = 0

    id2indices = {}
    for row_idx, uid in enumerate(index):
        id2indices.setdefault(uid, []).append(row_idx)

    with torch.no_grad():
        for row_indices in id2indices.values():
            all_group_idx = torch.tensor(row_indices, dtype=torch.long, device=device)
            group_success = success_mask.index_select(0, all_group_idx)
            success_count = int(group_success.sum().item())

            if skip_all_failed_groups and success_count == 0:
                skipped_all_failed_group_count += 1
                continue

            if (skip_all_failed_groups or success_only_novelty) and success_count < 2:
                continue

            selected_indices = row_indices
            if success_only_novelty:
                group_success_list = group_success.detach().cpu().tolist()
                selected_indices = [
                    row_idx for row_idx, is_success in zip(row_indices, group_success_list) if is_success
                ]

            if len(selected_indices) <= 1:
                continue

            group_idx = torch.tensor(selected_indices, dtype=torch.long, device=device)
            feature_idx = torch.tensor(selected_indices, dtype=torch.long, device=features.device)
            group_features = features.index_select(0, feature_idx).to(device=device, dtype=novelty_torch_dtype)
            group_signed = signed_scores.index_select(0, group_idx).to(dtype=novelty_torch_dtype)

            norms = torch.linalg.vector_norm(group_features, dim=-1, keepdim=True).clamp_min(eps)
            unit_features = group_features / norms
            similarities = torch.matmul(unit_features, unit_features.transpose(0, 1)).clamp(-1.0, 1.0)
            sim_sq = similarities.square()

            group_novelty = []
            for local_i in range(len(selected_indices)):
                mask = torch.ones(len(selected_indices), dtype=torch.bool, device=device)
                mask[local_i] = False
                weights = torch.softmax(group_signed[mask], dim=0)
                explained = torch.sum(weights * sim_sq[local_i, mask])
                novelty = torch.sqrt(torch.clamp(1.0 - explained, min=0.0, max=1.0))
                group_novelty.append(novelty)
            group_novelty = torch.stack(group_novelty)

            if bool(g2rl_config.get('normalize_novelty', True)):
                novelty_min = torch.min(group_novelty)
                novelty_max = torch.max(group_novelty)
                if float((novelty_max - novelty_min).item()) > eps:
                    normalized_novelty = (group_novelty - novelty_min) / (novelty_max - novelty_min + eps)
                else:
                    normalized_novelty = torch.zeros_like(group_novelty)
            else:
                normalized_novelty = group_novelty

            factors = 1.0 + lambda_coef * normalized_novelty
            group_shaped = torch.clamp(group_signed * factors, min=-reward_clip, max=reward_clip)

            shaped_scores.index_copy_(0, group_idx, group_shaped.to(dtype=shaped_scores.dtype))
            novelty_scores.index_copy_(0, group_idx, group_novelty.to(dtype=novelty_scores.dtype))
            factor_scores.index_copy_(0, group_idx, factors.to(dtype=factor_scores.dtype))
            group_sizes.append(float(len(selected_indices)))
            shaped_group_count += 1
            shaped_row_count += len(selected_indices)
            if success_only_novelty:
                shaped_success_only_count += 1

    dense_adjustment = token_level_rewards - token_level_scores
    shaped_outcome = torch.zeros_like(token_level_rewards)
    valid_response_length = response_mask.sum(dim=-1).long().clamp_min(1)
    last_token_idx = valid_response_length - 1
    shaped_outcome[torch.arange(shaped_outcome.size(0), device=device), last_token_idx] = shaped_scores
    data.batch['token_level_rewards'] = dense_adjustment + shaped_outcome

    metrics = {
        'g2rl/enabled': 1.0,
        'g2rl/active': 1.0,
        'g2rl/stop_after_steps': _g2rl_stop_after_steps_metric(g2rl_config),
        'g2rl/base_signed_reward_mean': torch.mean(signed_scores).detach().item(),
        'g2rl/shaped_reward_mean': torch.mean(shaped_scores).detach().item(),
        'g2rl/shaped_reward_max': torch.max(shaped_scores).detach().item(),
        'g2rl/shaped_reward_min': torch.min(shaped_scores).detach().item(),
        'g2rl/novelty_mean': torch.mean(novelty_scores).detach().item(),
        'g2rl/novelty_max': torch.max(novelty_scores).detach().item(),
        'g2rl/novelty_min': torch.min(novelty_scores).detach().item(),
        'g2rl/factor_mean': torch.mean(factor_scores).detach().item(),
        'g2rl/clipfrac': torch.mean((torch.abs(shaped_scores) >= reward_clip).float()).detach().item(),
        'g2rl/group_size_mean': float(np.mean(group_sizes)) if group_sizes else 1.0,
        'g2rl/skipped_all_failed_group_count': float(skipped_all_failed_group_count),
        'g2rl/shaped_group_count': float(shaped_group_count),
        'g2rl/shaped_success_only_count': float(shaped_success_only_count),
        'g2rl/shaped_row_ratio': float(shaped_row_count) / max(float(signed_scores.numel()), 1.0),
        'g2rl/success_row_ratio': torch.mean(success_mask.float()).detach().item(),
    }
    return data, metrics


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    task_scores = batch.batch["task_scores"].sum(-1)
    task_rounds = batch.batch["task_rounds"]

    response_length = batch.batch['response_mask'].sum(-1).float()
    prompt_length = batch.batch['attention_mask'].sum(-1).float() - response_length

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    response_mask = batch.batch['response_mask'].bool()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # task score
        'critic/task_score/mean':
            torch.mean(task_scores).detach().item(),
        'critic/task_score/max':
            torch.max(task_scores).detach().item(),
        'critic/task_score/min':
            torch.min(task_scores).detach().item(),
        # task round
        'critic/task_round/mean':
            torch.mean(task_rounds).detach().item(),
        'critic/task_round/max':
            torch.max(task_rounds).detach().item(),
        'critic/task_round/min':
            torch.min(task_rounds).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    num_overall_tokens = torch.sum(batch.batch['attention_mask']).item()
    num_response_tokens = torch.sum(batch.batch['response_mask']).item()

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'mclaw':
            self.use_critic = False  # MClaw uses co-located Q-critic, not verl's CriticWorker
        else:
            raise NotImplementedError

        self.global_steps = 0
        self.completed_steps = 0
        self.epoch_idx = 0
        self.batch_idx_in_epoch = 0
        self._epoch_indices_cache: Optional[tuple[int, list[int]]] = None
        self.early_stop_best = None
        self.early_stop_bad_checks = 0

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        g2rl_config = config.algorithm.get('g2rl', None)
        if g2rl_config is not None:
            _parse_g2rl_stop_after_steps(g2rl_config)
            get_g2rl_novelty_torch_dtype(g2rl_config)

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # === LOCAL CHANGE：可选构造固定 eval batch （训练初始化时缓存的一批 prompt），用于 rollout 评估。 ===
        from torch.utils.data import DataLoader, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            data_file=self.config.data.train_file,
            tokenizer=self.tokenizer,
            data_config=self.config.data,
            agentgym_config=self.config.actor_rollout_ref.agentgym,
        )
        self.train_batch_size = int(self.config.data.train_batch_size)
        self.data_seed = int(self.config.data.get('seed', 1))
        self.train_dataset_len = len(self.train_dataset)
        self.steps_per_epoch = compute_steps_per_epoch(
            dataset_len=self.train_dataset_len,
            batch_size=self.train_batch_size,
            drop_last=True,
        )
        assert self.steps_per_epoch >= 1

        print(f'Size of train dataloader: {self.steps_per_epoch}')

        self.val_dataloader = None
        self.val_fixed_batches = []
        val_files = self.config.data.get('val_files', None)
        test_freq = int(self.config.trainer.get('test_freq', -1))
        test_batches = int(self.config.trainer.get('test_batches', 1))
        if test_freq > 0 and val_files and test_batches > 0:
            # 固定 eval batch，训练中每次 test_freq 对比同一批任务，降低采样噪声。
            self.val_dataset = RLHFDataset(
                data_file=val_files,
                tokenizer=self.tokenizer,
                data_config=self.config.data,
                agentgym_config=self.config.actor_rollout_ref.agentgym,
            )
            val_batch_size = self.config.data.val_batch_size or self.config.data.train_batch_size
            val_sampler = SequentialSampler(data_source=self.val_dataset)
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                             batch_size=val_batch_size,
                                             drop_last=False,
                                             collate_fn=collate_fn,
                                             sampler=val_sampler)
            for batch_idx, batch_dict in enumerate(self.val_dataloader):
                if batch_idx >= test_batches:
                    break
                self.val_fixed_batches.append(batch_dict)
            assert len(self.val_fixed_batches) >= 1
            print(
                f'Size of val dataloader: {len(self.val_dataloader)}; '
                f'fixed eval batches: {len(self.val_fixed_batches)}'
            )

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = int(total_training_steps)
        if self.config.algorithm.rounds_ctrl.type == 'fixed':
            self.rounds_scheduler = FixedRoundsScheduler(rounds=self.config.algorithm.rounds_ctrl.rounds)
        elif self.config.algorithm.rounds_ctrl.type == 'scaling_inter_stepwise':
            self.rounds_scheduler = StepRoundsScheduler(steps_scaling_inter=self.config.algorithm.rounds_ctrl.steps_scaling_inter,
                                                   rounds_ls=self.config.algorithm.rounds_ctrl.rounds)
        else:
            raise NotImplementedError
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
            self.config.critic.optim.total_training_steps = self.total_training_steps

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _get_epoch_indices(self, epoch_idx: int) -> list[int]:
        cached = self._epoch_indices_cache
        if cached is not None and cached[0] == epoch_idx:
            return cached[1]

        indices = epoch_indices(
            dataset_len=self.train_dataset_len,
            epoch_idx=epoch_idx,
            shuffle=bool(self.config.data.shuffle),
            seed=self.data_seed,
        )
        self._epoch_indices_cache = (epoch_idx, indices)
        return indices

    def _get_train_batch_indices(self, epoch_idx: int, batch_idx_in_epoch: int) -> list[int]:
        if batch_idx_in_epoch < 0 or batch_idx_in_epoch >= self.steps_per_epoch:
            raise ValueError(
                f"batch_idx_in_epoch must be in [0, {self.steps_per_epoch}), got {batch_idx_in_epoch}"
            )
        indices = self._get_epoch_indices(epoch_idx)
        start = batch_idx_in_epoch * self.train_batch_size
        end = start + self.train_batch_size
        return indices[start:end]

    def _get_train_batch(self) -> dict:
        indices = self._get_train_batch_indices(self.epoch_idx, self.batch_idx_in_epoch)
        return collate_fn([self.train_dataset[index] for index in indices])

    def _set_cursor_from_completed_steps(self, completed_steps: int):
        cursor = cursor_from_completed_steps(completed_steps, self.steps_per_epoch)
        self.completed_steps = int(completed_steps)
        self.global_steps = int(completed_steps)
        self.epoch_idx = cursor.epoch_idx
        self.batch_idx_in_epoch = cursor.batch_idx_in_epoch
        self._epoch_indices_cache = None

    def _advance_train_cursor(self):
        self.batch_idx_in_epoch += 1
        if self.batch_idx_in_epoch >= self.steps_per_epoch:
            self.epoch_idx += 1
            self.batch_idx_in_epoch = 0
            self._epoch_indices_cache = None

    def _driver_rng_state(self):
        return {
            'torch_cpu': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        }

    def _load_driver_rng_state(self, rng_state):
        if not rng_state:
            return
        torch.set_rng_state(rng_state['torch_cpu'])
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['random'])

    def _actor_world_size(self) -> int:
        return int(getattr(self.actor_rollout_wg, 'world_size'))

    def _checkpoint_train_state(self):
        return {
            'version': 1,
            'global_steps': int(self.global_steps),
            'completed_steps': int(self.completed_steps),
            'epoch_idx': int(self.epoch_idx),
            'batch_idx_in_epoch': int(self.batch_idx_in_epoch),
            'steps_per_epoch': int(self.steps_per_epoch),
            'total_training_steps': int(self.total_training_steps),
            'dataset_len': int(self.train_dataset_len),
            'data_seed': int(self.data_seed),
            'shuffle': bool(self.config.data.shuffle),
            'train_batch_size': int(self.train_batch_size),
            'rollout_n': int(self.config.actor_rollout_ref.rollout.n),
            'world_size': int(self._actor_world_size()),
            'rounds_scheduler': self.rounds_scheduler.state_dict(),
            'driver_rng': self._driver_rng_state(),
        }

    def _checkpoint_data_state(self):
        return {
            'version': 1,
            'dataset_len': int(self.train_dataset_len),
            'data_file': str(self.config.data.train_file),
            'completed_steps': int(self.completed_steps),
            'epoch_idx': int(self.epoch_idx),
            'batch_idx_in_epoch': int(self.batch_idx_in_epoch),
            'steps_per_epoch': int(self.steps_per_epoch),
            'data_seed': int(self.data_seed),
            'shuffle': bool(self.config.data.shuffle),
            'train_batch_size': int(self.train_batch_size),
        }

    def _require_checkpoint_file(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Incomplete checkpoint: missing {path}")

    def _validate_worker_checkpoint_files(self, role_path: str, role_name: str):
        if not os.path.isdir(role_path):
            raise FileNotFoundError(f"Incomplete checkpoint: missing {role_name} directory {role_path}")
        world_size = self._actor_world_size()
        for rank in range(world_size):
            for prefix in ('model', 'optim', 'extra_state'):
                self._require_checkpoint_file(
                    os.path.join(role_path, f'{prefix}_world_size_{world_size}_rank_{rank}.pt')
                )

    def _validate_checkpoint_files(self, global_step_folder: str):
        tracker_path = os.path.join(os.path.dirname(global_step_folder), 'latest_checkpointed_iteration.txt')
        if os.path.isfile(tracker_path):
            with open(tracker_path, 'r') as f:
                first_line = f.readline().strip()
            if first_line:
                int(first_line)
        actor_path = os.path.join(global_step_folder, 'actor')
        self._validate_worker_checkpoint_files(actor_path, 'actor')
        if self.use_critic:
            critic_path = os.path.join(global_step_folder, 'critic')
            self._validate_worker_checkpoint_files(critic_path, 'critic')
        self._require_checkpoint_file(os.path.join(global_step_folder, DATA_STATE_NAME))

    def _validate_loaded_trainer_state(self, state: dict, global_step_folder: str):
        expected_completed_steps = _parse_global_step_from_path(global_step_folder)
        checks = {
            'completed_steps': expected_completed_steps,
            'global_steps': expected_completed_steps,
            'steps_per_epoch': self.steps_per_epoch,
            'dataset_len': self.train_dataset_len,
            'data_seed': self.data_seed,
            'shuffle': bool(self.config.data.shuffle),
            'train_batch_size': self.train_batch_size,
            'rollout_n': int(self.config.actor_rollout_ref.rollout.n),
            'world_size': self._actor_world_size(),
        }
        for key, expected in checks.items():
            actual = state.get(key)
            if actual != expected:
                raise ValueError(
                    f"Cannot strictly resume {global_step_folder}: {key} mismatch "
                    f"(checkpoint={actual}, current={expected})"
                )

        checkpoint_total_steps = state.get('total_training_steps')
        if checkpoint_total_steps != self.total_training_steps:
            allow_extend = bool(self.config.trainer.get('resume_allow_extend_total_training_steps', False))
            try:
                checkpoint_total_steps_int = int(checkpoint_total_steps)
            except (TypeError, ValueError):
                checkpoint_total_steps_int = None
            if (
                not allow_extend
                or checkpoint_total_steps_int is None
                or checkpoint_total_steps_int > int(self.total_training_steps)
            ):
                raise ValueError(
                    f"Cannot strictly resume {global_step_folder}: total_training_steps mismatch "
                    f"(checkpoint={checkpoint_total_steps}, current={self.total_training_steps})"
                )
            print(
                f"Extending resumed run total_training_steps from "
                f"{checkpoint_total_steps} to {self.total_training_steps}"
            )

        expected_cursor = cursor_from_completed_steps(expected_completed_steps, self.steps_per_epoch)
        if int(state.get('epoch_idx', -1)) != expected_cursor.epoch_idx or \
                int(state.get('batch_idx_in_epoch', -1)) != expected_cursor.batch_idx_in_epoch:
            raise ValueError(
                f"Cannot strictly resume {global_step_folder}: cursor mismatch "
                f"(checkpoint epoch={state.get('epoch_idx')} batch={state.get('batch_idx_in_epoch')}, "
                f"expected epoch={expected_cursor.epoch_idx} batch={expected_cursor.batch_idx_in_epoch})"
            )
        rounds_state = state.get('rounds_scheduler')
        if not isinstance(rounds_state, dict):
            raise ValueError(f"Cannot strictly resume {global_step_folder}: missing rounds_scheduler state")
        if 'global_steps' in rounds_state and int(rounds_state['global_steps']) != expected_completed_steps + 1:
            raise ValueError(
                f"Cannot strictly resume {global_step_folder}: rounds scheduler next step mismatch "
                f"(checkpoint={rounds_state['global_steps']}, expected={expected_completed_steps + 1})"
            )

    def _load_trainer_state_or_infer(self, global_step_folder: str):
        trainer_state_path = os.path.join(global_step_folder, TRAINER_STATE_NAME)
        if os.path.isfile(trainer_state_path):
            state = _torch_load(trainer_state_path, map_location='cpu')
            self._validate_loaded_trainer_state(state, global_step_folder)
            self.completed_steps = int(state['completed_steps'])
            self.global_steps = int(state['global_steps'])
            self.epoch_idx = int(state['epoch_idx'])
            self.batch_idx_in_epoch = int(state['batch_idx_in_epoch'])
            self.rounds_scheduler.load_state_dict(state['rounds_scheduler'])
            self._load_driver_rng_state(state.get('driver_rng'))
            return state

        try:
            data_state = _torch_load(os.path.join(global_step_folder, DATA_STATE_NAME), map_location='cpu')
        except Exception:
            data_state = None
        if isinstance(data_state, dict) and data_state.get('version') == 1:
            raise FileNotFoundError(
                f"Incomplete checkpoint: missing {trainer_state_path} next to new-format {DATA_STATE_NAME}"
            )

        completed_steps = _parse_global_step_from_path(global_step_folder)
        self._set_cursor_from_completed_steps(completed_steps)
        self.rounds_scheduler.set_global_steps(completed_steps + 1)
        print(
            f"Checkpoint {global_step_folder} has no {TRAINER_STATE_NAME}; "
            f"inferring cursor from global_step_{completed_steps}."
        )
        return None

    def _save_checkpoint(self):
        if int(self.global_steps) != int(self.completed_steps):
            raise RuntimeError(
                f"Refusing to save ambiguous checkpoint: global_steps={self.global_steps} "
                f"completed_steps={self.completed_steps}"
            )
        if self.config.trainer.storage_mode == 'aistudio':
            from aistudio_checkpoint.aistudio_mnt_checkpointer import AistudioMntCheckpointer
            ckpter = AistudioMntCheckpointer()
            save_dir = ckpter.get_save_dir(step=self.global_steps)
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(save_dir,
                                                    f'global_step_{self.global_steps}')
        elif self.config.trainer.storage_mode == 'local':
            # path: given_path + `/global_step_{global_steps}` + `/actor`
            local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                    f'global_step_{self.global_steps}')
        else:
            raise NotImplementedError
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        data_state_path = os.path.join(local_global_step_folder, DATA_STATE_NAME)
        trainer_state_path = os.path.join(local_global_step_folder, TRAINER_STATE_NAME)
        _atomic_torch_save(self._checkpoint_data_state(), data_state_path)
        _atomic_torch_save(self._checkpoint_train_state(), trainer_state_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        if self.config.trainer.storage_mode == 'aistudio':
            tracker_text = str(self.global_steps) + "\n" + ckpter.commit(memo=self.config.trainer.experiment_name)
        elif self.config.trainer.storage_mode == 'local':
            tracker_text = str(self.global_steps)
        else:
            raise NotImplementedError
        _atomic_write_text(local_latest_checkpointed_iteration, tracker_text)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            self._set_cursor_from_completed_steps(0)
            self.rounds_scheduler.set_global_steps(1)
            print('Training from scratch because trainer.resume_mode=disable')
            return 0

        resume_mode = self.config.trainer.resume_mode
        global_step_folder = None
        if resume_mode == 'auto':
            # load from hdfs
            if self.config.trainer.default_hdfs_dir is not None:
                raise NotImplementedError('load from hdfs is not implemented yet')
            else:
                checkpoint_folder = self.config.trainer.default_local_dir
                if not os.path.isabs(checkpoint_folder):
                    working_dir = os.getcwd()
                    checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
                if self.config.trainer.storage_mode == 'aistudio':
                    global_step_folder = find_latest_ckpt_path_aistudio(checkpoint_folder)  # None if no latest
                elif self.config.trainer.storage_mode == 'local':
                    global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
                else:
                    raise NotImplementedError
            if global_step_folder is None:
                print('Training from scratch')
                self._set_cursor_from_completed_steps(0)
                self.rounds_scheduler.set_global_steps(1)
                return 0
        else:
            assert isinstance(resume_mode, str), "resume ckpt must be str type"
            assert 'global_step_' in resume_mode, "resume ckpt must specify the global_steps"
            global_step_folder = resume_mode
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)

        print(f'Load from checkpoint folder: {global_step_folder}')

        self._validate_checkpoint_files(global_step_folder)
        trainer_state = self._load_trainer_state_or_infer(global_step_folder)

        print(f'Resuming from {global_step_folder}')
        print(
            f"Restored global_step={self.completed_steps}, epoch_idx={self.epoch_idx}, "
            f"batch_idx_in_epoch={self.batch_idx_in_epoch}, next_step={self.completed_steps + 1}"
        )

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        if isinstance(self.train_dataset, RLHFDataset):
            self.train_dataset.resume_dataset_state()

        if trainer_state is None:
            self.rounds_scheduler.set_global_steps(self.completed_steps + 1)

        return self.completed_steps

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _run_eval_rollout(self):
        # === LOCAL CHANGE：使用当前 rollout 策略运行固定验证 prompt。 ===
        if not getattr(self, "val_fixed_batches", None):
            return {}

        metrics = {}
        rollout_n = self.config.actor_rollout_ref.rollout.n
        world_size = self.actor_rollout_wg.world_size
        for batch_idx, batch_dict in enumerate(self.val_fixed_batches):
            timing_raw = {}
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            gen_batch = batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=['item_id', 'raw_prompt'],
            )
            valid_prompt_count = len(gen_batch)
            # 原因：Ray DP dispatch 要求 batch 可整除 world_size；dummy 样本 rollout 后再裁掉。
            gen_batch, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
            dummy_flags = [False] * valid_prompt_count + [True] * pad_size
            gen_batch.non_tensor_batch['rollout_is_dummy'] = np.array(dummy_flags, dtype=object)
            gen_batch.meta_info['global_steps'] = f"eval_global{self.global_steps}_batch{batch_idx}"
            gen_batch.meta_info['max_rounds'] = self.rounds_scheduler.get_rounds()

            with _timer('eval_gen', timing_raw):
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            if pad_size:
                gen_batch_output = unpad_dataproto(gen_batch_output, pad_size * rollout_n)

            batch_prefix = "eval" if len(self.val_fixed_batches) == 1 else f"eval/batch{batch_idx}"
            metrics.update(compute_rollout_monitor_metrics(gen_batch_output, prefix=batch_prefix))
            drop_rollout_monitor(gen_batch_output)

            task_scores = gen_batch_output.batch["task_scores"].sum(-1)
            if "task_successes" in gen_batch_output.batch.keys():
                task_successes = gen_batch_output.batch["task_successes"].float()
            else:
                task_successes = (task_scores > 0).float()
            task_rounds = gen_batch_output.batch["task_rounds"]
            response_length = gen_batch_output.batch["response_mask"].sum(-1).float()
            metrics.update({
                f"{batch_prefix}/task_score/mean": torch.mean(task_scores).detach().item(),
                f"{batch_prefix}/task_score/max": torch.max(task_scores).detach().item(),
                f"{batch_prefix}/task_score/min": torch.min(task_scores).detach().item(),
                f"{batch_prefix}/task_success/mean": torch.mean(task_successes).detach().item(),
                f"{batch_prefix}/pass_at_1": torch.mean(task_successes).detach().item(),
                f"{batch_prefix}/score_positive_at_1": torch.mean((task_scores > 0).float()).detach().item(),
                f"{batch_prefix}/task_round/mean": torch.mean(task_rounds).detach().item(),
                f"{batch_prefix}/task_round/max": torch.max(task_rounds).detach().item(),
                f"{batch_prefix}/task_round/min": torch.min(task_rounds).detach().item(),
                f"{batch_prefix}/response_length/mean": torch.mean(response_length).detach().item(),
                f"{batch_prefix}/response_length/max": torch.max(response_length).detach().item(),
                f"{batch_prefix}/response_length/min": torch.min(response_length).detach().item(),
                f"{batch_prefix}/timing_s/gen": timing_raw.get('eval_gen', 0.0),
            })
        return metrics

    def _early_stop_config(self):
        return self.config.trainer.get('early_stop', {})

    def _early_stop_enabled(self):
        cfg = self._early_stop_config()
        return bool(cfg.get('enabled', False))

    @staticmethod
    def _finite_metric(metrics: dict, key: str):
        value = metrics.get(key, None)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    def _early_stop_train_guard(self, metrics: dict):
        if not self._early_stop_enabled():
            return None

        cfg = self._early_stop_config()

        for key in ('actor/kl_loss', 'actor/grad_norm', 'actor/entropy_loss', 'actor/pg_loss'):
            if key in metrics:
                try:
                    value = float(metrics[key])
                except (TypeError, ValueError):
                    return f'non-finite {key}'
                if not math.isfinite(value):
                    return f'non-finite {key}'

        min_steps = int(cfg.get('min_steps', 0))
        if self.global_steps < min_steps:
            return None

        min_valid_ratio = cfg.get('min_rollout_valid_ratio', None)
        if min_valid_ratio is not None:
            min_valid_ratio = float(min_valid_ratio)
            for key in ('rollout/candidate_string_valid_ratio', 'rollout/taken_string_valid_ratio'):
                value = self._finite_metric(metrics, key)
                if value is not None and value < min_valid_ratio:
                    return f'{key}={value:.4g} < {min_valid_ratio:.4g}'

        max_response_length = cfg.get('max_response_length_mean', None)
        if max_response_length is not None:
            value = self._finite_metric(metrics, 'response_length/mean')
            if value is not None and value > float(max_response_length):
                return f'response_length/mean={value:.4g} > {float(max_response_length):.4g}'

        max_kl_loss = cfg.get('max_kl_loss', None)
        if max_kl_loss is not None:
            value = self._finite_metric(metrics, 'actor/kl_loss')
            if value is not None and value > float(max_kl_loss):
                return f'actor/kl_loss={value:.4g} > {float(max_kl_loss):.4g}'

        max_grad_norm = cfg.get('max_grad_norm', None)
        if max_grad_norm is not None:
            value = self._finite_metric(metrics, 'actor/grad_norm')
            if value is not None and value > float(max_grad_norm):
                return f'actor/grad_norm={value:.4g} > {float(max_grad_norm):.4g}'

        return None

    def _early_stop_eval_guard(self, eval_metrics: dict):
        if not self._early_stop_enabled() or not eval_metrics:
            return None, {}

        cfg = self._early_stop_config()
        metric_key = str(cfg.get('metric', 'eval/task_score/mean'))
        current = self._finite_metric(eval_metrics, metric_key)
        if current is None:
            print(f'[early_stop] skip check: missing/non-finite metric {metric_key}')
            return None, {}

        mode = str(cfg.get('mode', 'max')).lower()
        if mode not in ('max', 'min'):
            raise ValueError(f"trainer.early_stop.mode must be 'max' or 'min', got {mode}")

        min_delta = float(cfg.get('min_delta', 0.0))
        patience = int(cfg.get('patience', 0))
        min_steps = int(cfg.get('min_steps', 0))

        if self.early_stop_best is None:
            self.early_stop_best = current
            self.early_stop_bad_checks = 0
            improved = True
        elif mode == 'max':
            improved = current > self.early_stop_best + min_delta
        else:
            improved = current < self.early_stop_best - min_delta

        if improved:
            self.early_stop_best = current
            self.early_stop_bad_checks = 0
        else:
            self.early_stop_bad_checks += 1

        log_metrics = {
            'early_stop/current_metric': current,
            'early_stop/best_metric': float(self.early_stop_best),
            'early_stop/bad_checks': float(self.early_stop_bad_checks),
            'early_stop/should_stop': 0.0,
        }

        hard_min_metric = cfg.get('hard_min_metric', None)
        if hard_min_metric is not None and self.global_steps >= min_steps:
            hard_min_metric = float(hard_min_metric)
            if mode == 'max' and current < hard_min_metric:
                log_metrics['early_stop/should_stop'] = 1.0
                return f'{metric_key}={current:.4g} < hard_min_metric={hard_min_metric:.4g}', log_metrics
            if mode == 'min' and current > hard_min_metric:
                log_metrics['early_stop/should_stop'] = 1.0
                return f'{metric_key}={current:.4g} > hard_min_metric={hard_min_metric:.4g}', log_metrics

        max_drop = cfg.get('max_drop', None)
        if max_drop is not None and self.global_steps >= min_steps:
            max_drop = float(max_drop)
            if mode == 'max' and float(self.early_stop_best) - current > max_drop:
                log_metrics['early_stop/should_stop'] = 1.0
                return f'{metric_key} dropped by {float(self.early_stop_best) - current:.4g} > {max_drop:.4g}', log_metrics
            if mode == 'min' and current - float(self.early_stop_best) > max_drop:
                log_metrics['early_stop/should_stop'] = 1.0
                return f'{metric_key} rose by {current - float(self.early_stop_best):.4g} > {max_drop:.4g}', log_metrics

        if patience > 0 and self.global_steps >= min_steps and self.early_stop_bad_checks >= patience:
            log_metrics['early_stop/should_stop'] = 1.0
            return (
                f'{metric_key} did not improve by min_delta={min_delta:.4g} '
                f'for {self.early_stop_bad_checks} eval checks'
            ), log_metrics

        return None, log_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # === LOCAL CHANGE：记录 rollout 有效性指标并周期性执行 eval rollout。 ===
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.completed_steps >= self.total_training_steps:
            print(
                f"Checkpoint already completed training: completed_steps={self.completed_steps}, "
                f"total_training_steps={self.total_training_steps}. Exiting."
            )
            return

        if self.config.trainer.storage_mode == 'aistudio' and self.completed_steps == 0:
            self._save_checkpoint()

        while self.completed_steps < self.total_training_steps:
            self.global_steps = self.completed_steps + 1
            self.rounds_scheduler.set_global_steps(self.global_steps)
            batch_dict = self._get_train_batch()

            metrics = {}
            timing_raw = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)

            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'], non_tensor_batch_keys=['item_id', 'raw_prompt'])
            gen_batch.meta_info['global_steps'] = self.global_steps
            gen_batch.meta_info['max_rounds'] = self.rounds_scheduler.get_rounds()
            metrics.update({
                'max_rounds': self.rounds_scheduler.get_rounds(),
            })

            with _timer('step', timing_raw):
                # generate a batch
                with _timer('gen', timing_raw):
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                if self.config.algorithm.adv_estimator == 'remax':
                    with _timer('gen_max', timing_raw):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info['do_sample'] = False
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        batch = batch.union(gen_baseline_output)
                        reward_baseline_tensor = batch.batch['rewards']
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        batch.batch['reward_baselines'] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                         dtype=object)
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)
                # 原因：动作格式/环境有效率只用于观测 rollout 质量，不参与 advantage 计算。
                metrics.update(compute_rollout_monitor_metrics(batch, prefix="rollout"))
                drop_rollout_monitor(batch)

                # balance the number of valid tokens on each dp rank.
                # Note that this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                # recompute old_log_probs
                with _timer('old_log_prob', timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer('ref', timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                g2rl_config = self.config.algorithm.g2rl
                g2rl_active = is_g2rl_active(g2rl_config, self.global_steps)
                if g2rl_active:
                    with _timer('g2rl_feature', timing_raw):
                        batch.meta_info['g2rl_feature_scope'] = str(g2rl_config.get('feature_scope', 'response'))
                        batch.meta_info['g2rl_feature_topk'] = int(g2rl_config.get('feature_topk', 256))
                        batch.meta_info['g2rl_token_chunk_size'] = int(g2rl_config.get('token_chunk_size', 256))
                        g2rl_features = self.actor_rollout_wg.compute_g2rl_features(batch)
                        batch = batch.union(g2rl_features)
                elif bool(g2rl_config.get('enabled', False)):
                    metrics.update({
                        'g2rl/active': 0.0,
                        'g2rl/skipped_after_stop': 1.0,
                        'g2rl/stop_after_steps': _g2rl_stop_after_steps_metric(g2rl_config),
                    })

                # compute values
                if self.use_critic:
                    with _timer('values', timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer('adv', timing_raw):
                    # we combine with rule-based rm
                    reward_tensor = batch.batch['scores']
                    batch.batch['token_level_scores'] = reward_tensor

                    # compute rewards. apply_kl_penalty if available
                    if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                        batch, kl_metrics = apply_kl_penalty(batch,
                                                             kl_ctrl=self.kl_ctrl,
                                                             kl_penalty=self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                    if g2rl_active:
                        batch, g2rl_metrics = apply_g2rl_reward_shaping(batch, g2rl_config)
                        metrics.update(g2rl_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(batch,
                                              adv_estimator=self.config.algorithm.adv_estimator,
                                              gamma=self.config.algorithm.gamma,
                                              lam=self.config.algorithm.lam,
                                              num_repeat=self.config.actor_rollout_ref.rollout.n)

                # update critic
                if self.use_critic:
                    with _timer('update_critic', timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer('update_actor', timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

            early_stop_reason = self._early_stop_train_guard(metrics)
            if early_stop_reason is not None:
                metrics['early_stop/should_stop'] = 1.0

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if early_stop_reason is None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                # 注意：eval 走固定 batch 和当前 rollout 配置，不保存训练梯度。
                eval_metrics = self._run_eval_rollout()
                if eval_metrics:
                    early_stop_reason, early_stop_metrics = self._early_stop_eval_guard(eval_metrics)
                    eval_metrics.update(early_stop_metrics)
                    logger.log(data=eval_metrics, step=self.global_steps)

            if early_stop_reason is not None:
                print(f'[early_stop] step={self.global_steps}: {early_stop_reason}')

            current_step = self.global_steps
            self.completed_steps = current_step
            self._advance_train_cursor()
            self.global_steps = self.completed_steps
            self.rounds_scheduler.set_global_steps(self.completed_steps + 1)

            should_save = self.config.trainer.save_freq > 0 and (
                self.completed_steps % self.config.trainer.save_freq == 0 or
                self.completed_steps >= self.total_training_steps
            )
            if early_stop_reason is not None and bool(self._early_stop_config().get('save_on_stop', True)):
                should_save = True
            if should_save:
                self._save_checkpoint()

            if early_stop_reason is not None:
                return

            if self.completed_steps >= self.total_training_steps:
                return
