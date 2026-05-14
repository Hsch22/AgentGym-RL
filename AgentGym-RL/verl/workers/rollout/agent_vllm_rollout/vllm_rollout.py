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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List
from omegaconf import DictConfig
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

import os
import sys
import json
import time
import requests
import threading
import numpy as np
from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.agentgym.client import init_env_client
from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs

import random
from collections import defaultdict
from verl.workers.rollout.agent_vllm_rollout import clustering as _clustering

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, rollout_config: DictConfig, agentgym_config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        # === LOCAL CHANGE：初始化聚类配置和独立聚类模型。 ===
        super().__init__()
        self.config = rollout_config
        self.agentgym_config = agentgym_config
        assert not (not rollout_config.enforce_eager and rollout_config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=rollout_config.dtype,
            enforce_eager=rollout_config.enforce_eager,
            gpu_memory_utilization=rollout_config.gpu_memory_utilization,
            max_model_len=rollout_config.max_model_len,
            skip_tokenizer_init=False,
            load_format=rollout_config.load_format,
            disable_log_stats=rollout_config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=rollout_config.enable_chunked_prefill,
        )

        # vLLM 权重初始化后先下 CPU，降低与 FSDP 参数同步叠加时的显存峰值。
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=rollout_config.max_tokens,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)
        kwargs["n"] = 1  # because we have repeated task n times

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer

        # 记录 rank 方便多进程 rollout 日志按 rank 排查。
        try:
            if torch.distributed.is_initialized():
                self._rank = torch.distributed.get_rank()
            else:
                self._rank = int(os.environ.get("RANK", "0"))
        except Exception:
            self._rank = int(os.environ.get("RANK", "0"))

        # 可选启用候选动作聚类，用更多采样换取动作多样性。
        self.clustering_config = getattr(rollout_config, 'clustering', None)
        self.clustering_enabled = (
            self.clustering_config is not None
            and getattr(self.clustering_config, 'enabled', False)
        )
        self._actor_module_ref = actor_module
        self._gradient_model = None
        if self.clustering_enabled:
            method = getattr(self.clustering_config, "method", None)
            round1_candidates = int(getattr(self.clustering_config, "round1_candidates"))
            round1_clusters = int(getattr(self.clustering_config, "round1_clusters"))
            later_candidates = int(getattr(self.clustering_config, "later_candidates"))
            later_clusters = int(getattr(self.clustering_config, "later_clusters"))
            later_cluster_every = int(getattr(self.clustering_config, "later_cluster_every", 1))
            later_cluster_start = int(getattr(self.clustering_config, "later_cluster_start", 1))
            later_cluster_until = int(getattr(self.clustering_config, "later_cluster_until", -1))
            later_cluster_horizon_min = float(getattr(self.clustering_config, "later_cluster_horizon_min", 0.0))
            if method not in ("gradient", "gradient_multiview", "semantic", "random_valid", "random_raw"):
                raise ValueError(f"unsupported clustering method: {method}")
            assert round1_candidates >= round1_clusters >= 1, (
                f"invalid round0 clustering config: {round1_candidates=} {round1_clusters=}"
            )
            assert later_candidates >= later_clusters >= 1, (
                f"invalid later-round clustering config: {later_candidates=} {later_clusters=}"
            )
            assert later_cluster_every >= 0, (
                f"later_cluster_every must be >= 0, got {later_cluster_every}"
            )
            assert later_cluster_start >= 1, (
                f"later_cluster_start is a later-round index and must be >= 1, got {later_cluster_start}"
            )
            assert 0.0 <= later_cluster_horizon_min <= 1.0, (
                f"later_cluster_horizon_min must be in [0, 1], got {later_cluster_horizon_min}"
            )
            assert round1_clusters == int(self.config.n), (
                f"round0 clustering must match rollout.n so each repeated trajectory gets one center: "
                f"{round1_clusters=} rollout.n={self.config.n}"
            )
            print(
                "[clustering-config] "
                f"method={method} "
                f"round0={round1_candidates}/{round1_clusters} "
                f"later={later_candidates}/{later_clusters} "
                f"later_schedule=every:{later_cluster_every},start:{later_cluster_start},"
                f"until:{later_cluster_until},horizon_min:{later_cluster_horizon_min} "
                f"feature_topk={int(getattr(self.clustering_config, 'feature_topk', 256))} "
                f"feature_chunk_size={int(getattr(self.clustering_config, 'feature_chunk_size', 4))}"
            )
        # gradient/semantic/multiview 聚类需要独立 HF 模型，避免直接调 FSDP actor 时因 rank 提前结束卡住 all_gather。
        if self.clustering_enabled and self.clustering_config.method in ("gradient", "gradient_multiview", "semantic"):
            from transformers import AutoModelForCausalLM
            gradient_model_path = self.clustering_config.gradient_model_path
            self._gradient_model = AutoModelForCausalLM.from_pretrained(
                gradient_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            # 聚类模型常驻 CPU，避免和 vLLM KV cache、FSDP all-gather 同时占显存。
            self._gradient_model.cpu()
            self._gradient_model.eval()
            # 只有旧 gradient CountSketch 聚类要对参数反传；multiview/semantic 都是 forward-only。
            requires_grad = self.clustering_config.method == "gradient"
            for p in self._gradient_model.parameters():
                p.requires_grad_(requires_grad)


    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _sync_gradient_model_from_actor(self):
        # === LOCAL CHANGE：将 FSDP actor 权重同步到非 FSDP 聚类模型。 ===
        if not self.clustering_enabled:
            return
        if self.clustering_config.method not in ("gradient", "gradient_multiview", "semantic"):
            return
        if self._gradient_model is None:
            return
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(self._actor_module_ref, StateDictType.FULL_STATE_DICT, cfg):
            full_state = self._actor_module_ref.state_dict()
        target_state = self._gradient_model.state_dict()
        with torch.no_grad():
            for k in list(full_state.keys()):
                cpu_tensor = full_state.pop(k)
                clean_k = k.replace("_fsdp_wrapped_module.", "")
                tgt = target_state.get(clean_k)
                if tgt is None:
                    del cpu_tensor
                    continue
                tgt.copy_(cpu_tensor, non_blocking=True)
                del cpu_tensor
        full_state.clear()
        torch.cuda.synchronize()
        del full_state, target_state
        # 同步完成后才搬到 GPU，后续候选选择共享这份独立模型。
        if next(self._gradient_model.parameters()).device.type == "cpu":
            self._gradient_model.cuda()
        torch.cuda.empty_cache()

    def _get_clustering_model(self):
        # === LOCAL CHANGE：集中处理聚类和随机基线的方法选择。 ===
        method = self.clustering_config.method
        if method in ("gradient", "gradient_multiview", "semantic"):
            # 聚类前向不走 FSDP，避免不同轨迹早停导致 collective 调用不一致。
            return self._gradient_model
        if method in ("random_valid", "random_raw"):
            return None
        raise ValueError(f"Unknown clustering method: {method}")

    def _should_cluster_later_round(self, round_idx, max_rounds):
        """Return whether a later environment round should use expensive candidate clustering."""
        every = int(getattr(self.clustering_config, "later_cluster_every", 1))
        if every <= 0:
            return False

        start = int(getattr(self.clustering_config, "later_cluster_start", 1))
        until = int(getattr(self.clustering_config, "later_cluster_until", -1))
        if round_idx < start:
            return False
        if until >= 0 and round_idx > until:
            return False
        if (round_idx - start) % every != 0:
            return False

        horizon_min = float(getattr(self.clustering_config, "later_cluster_horizon_min", 0.0))
        horizon_ratio = (max_rounds - round_idx - 1) / max(max_rounds - 1, 1)
        return horizon_ratio >= horizon_min

    def _select_response_indices(
        self,
        model,
        obs_token_ids,
        response_texts,
        k,
        *,
        round_idx=0,
        max_rounds=1,
        temperature=None,
        selection_stats=None,
    ):
        """Select candidate indices according to the configured rollout method."""
        # === LOCAL CHANGE：按 gradient/semantic/random 策略选择候选中心。 ===
        method = self.clustering_config.method
        if method in ("random_valid", "random_raw"):
            return random.sample(range(len(response_texts)), min(k, len(response_texts)))
        if method == "gradient":
            with torch.enable_grad():
                return _clustering.select_centers(
                    method="gradient",
                    model=model,
                    tokenizer=self.tokenizer,
                    obs_token_ids=obs_token_ids,
                    response_texts=response_texts,
                    k=k,
                    d_proj=self.clustering_config.gradient_d_proj,
                )
        if method == "semantic":
            return _clustering.select_centers(
                method="semantic",
                model=model,
                tokenizer=self.tokenizer,
                obs_token_ids=obs_token_ids,
                response_texts=response_texts,
                k=k,
            )
        if method == "gradient_multiview":
            with torch.no_grad():
                return _clustering.select_centers(
                    method="gradient_multiview",
                    model=model,
                    tokenizer=self.tokenizer,
                    obs_token_ids=obs_token_ids,
                    response_texts=response_texts,
                    k=k,
                    round_idx=round_idx,
                    max_rounds=max_rounds,
                    temperature=float(
                        self.config.get("temperature", 1.0)
                        if temperature is None
                        else temperature
                    ),
                    feature_topk=int(getattr(self.clustering_config, "feature_topk", 256)),
                    feature_chunk_size=int(getattr(self.clustering_config, "feature_chunk_size", 4)),
                    stats=selection_stats,
                )
        raise ValueError(f"Unknown clustering method: {method}")

    @staticmethod
    def _new_rollout_monitor():
        # === LOCAL CHANGE：创建每条轨迹的 rollout 指标累加器。 ===
        # monitor 随 DataProto 回传，训练侧才能聚合动作有效率等指标。
        return {}

    @staticmethod
    def _monitor_add(monitor, key, value, round_idx=None):
        # === LOCAL CHANGE：累加总计和按轮次分桶的 monitor 计数。 ===
        monitor[key] = monitor.get(key, 0) + value
        if round_idx is not None:
            bucket = "round0" if round_idx == 0 else "later"
            bucket_key = f"{bucket}_{key}"
            monitor[bucket_key] = monitor.get(bucket_key, 0) + value

    @staticmethod
    def _monitor_max(monitor, key, value, round_idx=None):
        # === LOCAL CHANGE：维护 monitor 字段中的最大值。 ===
        monitor[key] = max(monitor.get(key, 0), value)
        if round_idx is not None:
            bucket = "round0" if round_idx == 0 else "later"
            bucket_key = f"{bucket}_{key}"
            monitor[bucket_key] = max(monitor.get(bucket_key, 0), value)

    @staticmethod
    def _parse_action_for_monitor(raw_response):
        # === LOCAL CHANGE：解析 TextCraft 动作有效性用于 rollout 诊断。 ===
        normalized = _clustering.parse_valid_action(raw_response)
        return normalized is not None, normalized or ""

    def _record_candidate_monitor(self, monitor, round_idx, response_texts):
        # === LOCAL CHANGE：统计采样候选中格式有效的动作字符串。 ===
        valid_count = sum(
            1 for text in response_texts
            if _clustering.parse_valid_action(text) is not None
        )
        total = len(response_texts)
        self._monitor_add(monitor, "candidate_total", total, round_idx)
        self._monitor_add(monitor, "candidate_string_valid", valid_count, round_idx)
        return valid_count

    def _record_selection_monitor(self, monitor, round_idx, selected_texts, selection_stats=None):
        # === LOCAL CHANGE：统计候选中心里同 normalized Action 的重复与 tokenizer fallback。 ===
        normalized_actions = [
            action
            for action in (_clustering.parse_valid_action(text) for text in selected_texts)
            if action is not None
        ]
        duplicate_count = len(normalized_actions) - len(set(normalized_actions))
        self._monitor_add(monitor, "selected_total", len(selected_texts), round_idx)
        self._monitor_add(monitor, "selected_duplicate_action", duplicate_count, round_idx)
        if selection_stats:
            self._monitor_add(
                monitor,
                "offset_fallback_count",
                int(selection_stats.get("offset_fallback_count", 0)),
                round_idx,
            )
            self._monitor_add(
                monitor,
                "multiview_feature_candidates",
                int(selection_stats.get("multiview_feature_candidates", 0)),
                round_idx,
            )

    def _record_taken_action(
        self,
        *,
        rollout_monitor,
        action_records,
        action_records_lock,
        rollout_handler,
        trajectory_index,
        round_idx,
        raw_response,
        raw_candidate_index,
        candidate_count,
        candidate_string_valid_count,
        step_output=None,
        error=None,
    ):
        # === LOCAL CHANGE：记录被选动作、环境有效性、奖励和诊断信息。 ===
        string_valid, normalized_action = self._parse_action_for_monitor(raw_response)
        step_info = getattr(step_output, "info", {}) or {}
        env_valid = bool(step_info.get("env_action_valid", False))
        reward = float(getattr(step_output, "reward", 0.0) if step_output is not None else 0.0)
        done = bool(getattr(step_output, "done", True) if step_output is not None else True)
        observation = getattr(step_output, "state", "") if step_output is not None else str(error)
        raw_chars = len(raw_response or "")
        norm_chars = len(normalized_action or "")

        self._monitor_add(rollout_monitor, "taken_total", 1, round_idx)
        self._monitor_add(rollout_monitor, "taken_string_valid", int(string_valid), round_idx)
        self._monitor_add(rollout_monitor, "taken_env_valid", int(env_valid), round_idx)
        self._monitor_add(
            rollout_monitor,
            "string_valid_env_invalid",
            int(string_valid and not env_valid),
            round_idx,
        )
        self._monitor_add(rollout_monitor, "taken_action_raw_chars_sum", raw_chars, round_idx)
        self._monitor_add(rollout_monitor, "taken_action_norm_chars_sum", norm_chars, round_idx)
        self._monitor_max(rollout_monitor, "taken_action_raw_chars_max", raw_chars, round_idx)
        self._monitor_max(rollout_monitor, "taken_action_norm_chars_max", norm_chars, round_idx)

        record = {
            "rank": getattr(self, "_rank", int(os.environ.get("RANK", "0"))),
            "trajectory_index": trajectory_index,
            "item_id": rollout_handler.item_id,
            "round": round_idx,
            "raw_candidate_index": raw_candidate_index,
            "candidate_count": candidate_count,
            "candidate_string_valid_count": candidate_string_valid_count,
            "raw_response": raw_response,
            "normalized_action": normalized_action,
            "string_valid": string_valid,
            "env_valid": env_valid,
            "env_action_type": step_info.get("env_action_type", "unknown"),
            "env_action_error": step_info.get("env_action_error", "" if error is None else str(error)),
            "reward": reward,
            "done": done,
            "env_observation": observation,
            "raw_response_chars": raw_chars,
            "normalized_action_chars": norm_chars,
        }
        with action_records_lock:
            action_records.append(record)

    @staticmethod
    def _monitor_delta(monitors, key, start_value):
        # === LOCAL CHANGE：汇总每轮 monitor 增量。 ===
        return sum(m.get(key, 0) for m in monitors) - start_value

    def _round_valid_summary(self, monitors, start_counts):
        # === LOCAL CHANGE：格式化 rollout 有效性摘要用于 stderr 进度日志。 ===
        cand_total = self._monitor_delta(monitors, "candidate_total", start_counts["candidate_total"])
        cand_valid = self._monitor_delta(monitors, "candidate_string_valid", start_counts["candidate_string_valid"])
        taken_total = self._monitor_delta(monitors, "taken_total", start_counts["taken_total"])
        taken_string_valid = self._monitor_delta(monitors, "taken_string_valid", start_counts["taken_string_valid"])
        taken_env_valid = self._monitor_delta(monitors, "taken_env_valid", start_counts["taken_env_valid"])
        string_valid_env_invalid = self._monitor_delta(
            monitors,
            "string_valid_env_invalid",
            start_counts["string_valid_env_invalid"],
        )
        selected_total = self._monitor_delta(monitors, "selected_total", start_counts["selected_total"])
        selected_duplicate_action = self._monitor_delta(
            monitors,
            "selected_duplicate_action",
            start_counts["selected_duplicate_action"],
        )
        offset_fallback_count = self._monitor_delta(
            monitors,
            "offset_fallback_count",
            start_counts["offset_fallback_count"],
        )
        later_clustered_action = self._monitor_delta(
            monitors,
            "later_clustered_action",
            start_counts.get("later_clustered_action", 0),
        )
        later_skipped_action = self._monitor_delta(
            monitors,
            "later_skipped_action",
            start_counts.get("later_skipped_action", 0),
        )
        return (
            f"candidate_string_valid={cand_valid}/{cand_total} "
            f"selected_duplicate_action={selected_duplicate_action}/{selected_total} "
            f"taken_string_valid={taken_string_valid}/{taken_total} "
            f"taken_env_valid={taken_env_valid}/{taken_total} "
            f"string_valid_env_invalid={string_valid_env_invalid} "
            f"offset_fallback={offset_fallback_count} "
            f"later_clustered={later_clustered_action} "
            f"later_skipped={later_skipped_action}"
        )

    def _round0_clustering(
        self,
        rollout_handler_ls,
        env_clients,
        task_rounds,
        rounds,
        max_rounds,
        kwargs_sp,
        rollout_monitors,
        action_records,
        action_records_lock,
    ):
        # === LOCAL CHANGE：首轮对重复轨迹进行多候选聚类选择。 ===
        # 同一 prompt 先集中采样再选中心，减少重复轨迹，把 rollout.n 分配给更多样的动作。
        groups = defaultdict(list)
        for idx, handler in enumerate(rollout_handler_ls):
            if not handler.done:
                groups[handler.item_id].append(idx)

        n_candidates = self.clustering_config.round1_candidates
        n_clusters = self.clustering_config.round1_clusters
        clustering_model = self._get_clustering_model()

        for item_id, handler_idxs in groups.items():
            representative = rollout_handler_ls[handler_idxs[0]]
            gen_prompt = representative.get_generation_prompt(self.tokenizer)

            # 用 vLLM 原生 n=K 一次性取候选
            sp_kwargs = dict(kwargs_sp)
            sp_kwargs['n'] = n_candidates
            with self.update_sampling_params(**sp_kwargs):
                output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=[gen_prompt],
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
            # output[0] 是 padded 的候选 token 矩阵，后续统一 decode 后再做聚类。
            all_response_ids = output[0].tolist()
            all_response_texts = [
                self.tokenizer.decode(r, skip_special_tokens=True) for r in all_response_ids
            ]
            candidate_string_valid_count = self._record_candidate_monitor(
                rollout_monitors[handler_idxs[0]],
                rounds,
                all_response_texts,
            )

            if self.clustering_config.method == "random_raw":
                selection_stats = {}
                selected_idxs = self._select_response_indices(
                    model=clustering_model,
                    obs_token_ids=gen_prompt,
                    response_texts=all_response_texts,
                    k=n_clusters,
                    round_idx=rounds,
                    max_rounds=max_rounds,
                    temperature=sp_kwargs.get("temperature", self.config.get("temperature", 1.0)),
                    selection_stats=selection_stats,
                )
                if not selected_idxs:
                    selected_idxs = [0]
                selected_responses = [(i, all_response_texts[i]) for i in selected_idxs]
            else:
                # 只聚类格式可解析的动作；输入仍用原始 Thought+Action，保留语义差异。
                valid_indices = [
                    i for i, t in enumerate(all_response_texts)
                    if _clustering.parse_valid_action(t) is not None
                ]
                valid_texts = [all_response_texts[i] for i in valid_indices]

                # 候选全无效时仍走 env.step，让环境返回错误状态而不是中断 batch。
                if len(valid_texts) == 0:
                    selected_responses = [(0, all_response_texts[0])]
                    selection_stats = {}
                else:
                    k_eff = min(n_clusters, len(valid_texts))
                    selection_stats = {}
                    center_local_idxs = self._select_response_indices(
                        model=clustering_model,
                        obs_token_ids=gen_prompt,
                        response_texts=valid_texts,
                        k=k_eff,
                        round_idx=rounds,
                        max_rounds=max_rounds,
                        temperature=sp_kwargs.get("temperature", self.config.get("temperature", 1.0)),
                        selection_stats=selection_stats,
                    )
                    selected_responses = [
                        (valid_indices[i], valid_texts[i]) for i in center_local_idxs
                    ]
            self._record_selection_monitor(
                rollout_monitors[handler_idxs[0]],
                rounds,
                [text for _, text in selected_responses],
                selection_stats,
            )

            # 中心数少于重复轨迹时循环分配，保持 batch 形状与 rollout.n 一致。
            time.sleep(self.config.send_interval)
            for assign_i, hidx in enumerate(handler_idxs):
                raw_candidate_index, resp_text = selected_responses[assign_i % len(selected_responses)]
                handler = rollout_handler_ls[hidx]
                handler.add_assistant_message(self.tokenizer, resp_text)
                task_rounds[hidx] += 1
                try:
                    step_output = env_clients[hidx].step(resp_text)
                    self._record_taken_action(
                        rollout_monitor=rollout_monitors[hidx],
                        action_records=action_records,
                        action_records_lock=action_records_lock,
                        rollout_handler=handler,
                        trajectory_index=hidx,
                        round_idx=rounds,
                        raw_response=resp_text,
                        raw_candidate_index=raw_candidate_index,
                        candidate_count=len(all_response_texts),
                        candidate_string_valid_count=candidate_string_valid_count,
                        step_output=step_output,
                    )
                    handler.score = step_output.reward
                    handler.done = step_output.done
                    if not step_output.done:
                        handler.add_user_message(self.tokenizer, step_output.state)
                except Exception as e:
                    self._record_taken_action(
                        rollout_monitor=rollout_monitors[hidx],
                        action_records=action_records,
                        action_records_lock=action_records_lock,
                        rollout_handler=handler,
                        trajectory_index=hidx,
                        round_idx=rounds,
                        raw_response=resp_text,
                        raw_candidate_index=raw_candidate_index,
                        candidate_count=len(all_response_texts),
                        candidate_string_valid_count=candidate_string_valid_count,
                        error=e,
                    )
                    handler.score = 0
                    handler.done = True
                    print(f"Round 0 step Error: {e} item id = {handler.item_id}")

    def _later_round_clustering(
        self,
        rollout_handler_ls,
        env_clients,
        task_rounds,
        rounds,
        max_rounds,
        kwargs_sp,
        rollout_monitors,
        action_records,
        action_records_lock,
    ):
        """Round 1+: each active handler generates later_candidates, selects one response."""
        # === LOCAL CHANGE：后续轮次为每条活跃轨迹采样候选并选择中心。 ===
        n_candidates = self.clustering_config.later_candidates
        n_clusters = self.clustering_config.later_clusters
        clustering_model = self._get_clustering_model()

        not_done = [(idx, h) for idx, h in enumerate(rollout_handler_ls) if not h.done]
        if not not_done:
            return
        for handler_idx, _ in not_done:
            self._monitor_add(rollout_monitors[handler_idx], "later_clustered_action", 1, rounds)

        time.sleep(self.config.send_interval)
        for handler_idx, handler in not_done:
            gen_prompt = handler.get_generation_prompt(self.tokenizer)

            sp_kwargs = dict(kwargs_sp)
            sp_kwargs['n'] = n_candidates
            with self.update_sampling_params(**sp_kwargs):
                output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=[gen_prompt],
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
            all_response_ids = output[0].tolist()
            all_response_texts = [
                self.tokenizer.decode(r, skip_special_tokens=True) for r in all_response_ids
            ]
            candidate_string_valid_count = self._record_candidate_monitor(
                rollout_monitors[handler_idx],
                rounds,
                all_response_texts,
            )

            if self.clustering_config.method == "random_raw":
                selection_stats = {}
                selected_idxs = self._select_response_indices(
                    model=clustering_model,
                    obs_token_ids=gen_prompt,
                    response_texts=all_response_texts,
                    k=n_clusters,
                    round_idx=rounds,
                    max_rounds=max_rounds,
                    temperature=sp_kwargs.get("temperature", self.config.get("temperature", 1.0)),
                    selection_stats=selection_stats,
                )
                if not selected_idxs:
                    selected_idxs = [0]
                self._record_selection_monitor(
                    rollout_monitors[handler_idx],
                    rounds,
                    [all_response_texts[i] for i in selected_idxs],
                    selection_stats,
                )
                chosen_raw_idx = random.choice(selected_idxs)
                chosen_text = all_response_texts[chosen_raw_idx]
            else:
                valid_indices = [
                    i for i, t in enumerate(all_response_texts)
                    if _clustering.parse_valid_action(t) is not None
                ]
                valid_texts = [all_response_texts[i] for i in valid_indices]

                if len(valid_texts) == 0:
                    # 候选全无效时保留原始响应交给环境判错，避免丢失该轮轨迹。
                    chosen_raw_idx = 0
                    chosen_text = all_response_texts[0]
                    self._record_selection_monitor(
                        rollout_monitors[handler_idx],
                        rounds,
                        [chosen_text],
                        {},
                    )
                else:
                    k_eff = min(n_clusters, len(valid_texts))
                    selection_stats = {}
                    center_local_idxs = self._select_response_indices(
                        model=clustering_model,
                        obs_token_ids=gen_prompt,
                        response_texts=valid_texts,
                        k=k_eff,
                        round_idx=rounds,
                        max_rounds=max_rounds,
                        temperature=sp_kwargs.get("temperature", self.config.get("temperature", 1.0)),
                        selection_stats=selection_stats,
                    )
                    self._record_selection_monitor(
                        rollout_monitors[handler_idx],
                        rounds,
                        [valid_texts[i] for i in center_local_idxs],
                        selection_stats,
                    )
                    chosen_local = random.choice(center_local_idxs)
                    chosen_raw_idx = valid_indices[chosen_local]
                    chosen_text = valid_texts[chosen_local]

            handler.add_assistant_message(self.tokenizer, chosen_text)
            task_rounds[handler_idx] += 1
            try:
                step_output = env_clients[handler_idx].step(chosen_text)
                self._record_taken_action(
                    rollout_monitor=rollout_monitors[handler_idx],
                    action_records=action_records,
                    action_records_lock=action_records_lock,
                    rollout_handler=handler,
                    trajectory_index=handler_idx,
                    round_idx=rounds,
                    raw_response=chosen_text,
                    raw_candidate_index=chosen_raw_idx,
                    candidate_count=len(all_response_texts),
                    candidate_string_valid_count=candidate_string_valid_count,
                    step_output=step_output,
                )
                handler.score = step_output.reward
                handler.done = step_output.done
                if not step_output.done:
                    handler.add_user_message(self.tokenizer, step_output.state)
            except Exception as e:
                self._record_taken_action(
                    rollout_monitor=rollout_monitors[handler_idx],
                    action_records=action_records,
                    action_records_lock=action_records_lock,
                    rollout_handler=handler,
                    trajectory_index=handler_idx,
                    round_idx=rounds,
                    raw_response=chosen_text,
                    raw_candidate_index=chosen_raw_idx,
                    candidate_count=len(all_response_texts),
                    candidate_string_valid_count=candidate_string_valid_count,
                    error=e,
                )
                handler.score = 0
                handler.done = True
                print(f"Round {rounds} step Error: {e} item id = {handler.item_id}")

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for _ in range(n):
                # only keep not pad part
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][i])
                attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                handler = RolloutHandler(
                    messages=[
                        Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt
                    ],
                    task_name=prompts.non_tensor_batch["item_id"][i].split("_")[0],
                    item_id=int(prompts.non_tensor_batch["item_id"][i].split("_")[-1]),
                    score=0,
                    done=False,
                    input_ids=list(input_ids),
                    prompt_ids=list(input_ids),
                    response_ids=[],
                    attention_mask=list(attention_mask),
                    prompt_attention_mask=list(attention_mask),
                    response_attention_mask=[],
                    position_ids=list(position_ids),
                    prompt_position_ids=list(position_ids),
                    response_position_ids=[],
                    loss_mask=[0] * len(input_ids),
                    prompt_loss_mask=[0] * len(input_ids),
                    response_loss_mask=[],
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length)
                )
                assert len(handler.input_ids) == len(handler.attention_mask) == len(handler.position_ids) == len(handler.loss_mask), f"RolloutHandler has mismatched length: input_ids={len(handler.input_ids)}, attention_mask={len(handler.attention_mask)}, position_ids={len(handler.position_ids)}, loss_mask={len(handler.loss_mask)}"
                handler_list.append(handler)
        return handler_list


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # === LOCAL CHANGE：增加 dummy padding、聚类路径、动作日志和 monitor 返回。 ===
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        global_steps = prompts.meta_info.get('global_steps', None)
        max_rounds = prompts.meta_info.get('max_rounds', 10)
        cur_device = prompts.batch["input_ids"].device

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # repeat for self.config.n times to rollout
        batch_size = prompts.batch['input_ids'].size(0)
        batch_size *= self.config.n
        prompt_dummy_flags = prompts.non_tensor_batch.get("rollout_is_dummy", None)
        if prompt_dummy_flags is None:
            valid_trajectory_mask = [True] * batch_size
        else:
            valid_trajectory_mask = []
            for is_dummy in prompt_dummy_flags:
                valid_trajectory_mask.extend([not bool(is_dummy)] * self.config.n)
            if len(valid_trajectory_mask) != batch_size:
                valid_trajectory_mask = [True] * batch_size
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=self.config.n)
        env_clients = []
        rollout_bar = None
        try:
            for _ in range(batch_size):
                env_clients.append(init_env_client(self.agentgym_config))
            time.sleep(self.config.send_interval) # take a break before sendng request
            all_done_flag = False
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                try:
                    env_clients[idx].reset(rollout_handler.item_id)
                    task = env_clients[idx].observe()
                    rollout_handler.add_user_message(self.tokenizer, task)
                except TimeoutError:
                    print(f"Reset Timeout: Webarena Env Timeout. item id = {rollout_handler.item_id}")
                    rollout_handler.done = True
                    rollout_handler.score = 0

            rounds = 0
            task_rounds = [0] * batch_size
            rollout_monitors = [self._new_rollout_monitor() for _ in range(batch_size)]
            action_records = []
            action_records_lock = threading.Lock()
            rollout_bar = tqdm(total = max_rounds, desc="Running rounds", disable=torch.distributed.get_rank() != 0)
            _rollout_rank = getattr(self, "_rank", int(os.environ.get("RANK", "0")))
            _rollout_wall_start = time.time()
            print(
                f"[rollout] rank={_rollout_rank} START batch_size={batch_size} max_rounds={max_rounds} t={_rollout_wall_start:.1f}",
                file=sys.stderr, flush=True,
            )
            def standard_round_step(record_later_skip=False):
                generation_prompt_idxs = []
                not_done_idxs = []
                for idx, rollout_handler in enumerate(rollout_handler_ls):
                    if not rollout_handler.done:
                        generation_prompt_idxs.append(rollout_handler.get_generation_prompt(self.tokenizer))
                        not_done_idxs.append(idx)

                if record_later_skip:
                    for idx in not_done_idxs:
                        self._monitor_add(rollout_monitors[idx], "later_skipped_action", 1, rounds)

                rollout_bar.set_description(
                    f"Rounds {rounds + 1}/{max_rounds} | Active agents per gpu: {len(not_done_idxs)}"
                )
                if len(not_done_idxs) == 0:
                    return True

                with self.update_sampling_params(**kwargs):
                    output = self.inference_engine.generate(
                        prompts=None,
                        prompt_token_ids=generation_prompt_idxs,
                        sampling_params=self.sampling_params,
                        use_tqdm=False)
                response_ids_local = output[0].tolist()

                def direct_agent_step(i, idx):
                    content = self.tokenizer.decode(response_ids_local[i], skip_special_tokens=True)
                    candidate_string_valid_count = self._record_candidate_monitor(
                        rollout_monitors[idx],
                        rounds,
                        [content],
                    )
                    rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
                    task_rounds[idx] += 1
                    try:
                        step_output = env_clients[idx].step(content)
                        self._record_taken_action(
                            rollout_monitor=rollout_monitors[idx],
                            action_records=action_records,
                            action_records_lock=action_records_lock,
                            rollout_handler=rollout_handler_ls[idx],
                            trajectory_index=idx,
                            round_idx=rounds,
                            raw_response=content,
                            raw_candidate_index=0,
                            candidate_count=1,
                            candidate_string_valid_count=candidate_string_valid_count,
                            step_output=step_output,
                        )
                        state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                            step_output.state,
                            step_output.reward,
                            step_output.done,
                        )
                        rollout_handler_ls[idx].add_user_message(self.tokenizer, state)
                        return step_output.done
                    except Exception as e:
                        self._record_taken_action(
                            rollout_monitor=rollout_monitors[idx],
                            action_records=action_records,
                            action_records_lock=action_records_lock,
                            rollout_handler=rollout_handler_ls[idx],
                            trajectory_index=idx,
                            round_idx=rounds,
                            raw_response=content,
                            raw_candidate_index=0,
                            candidate_count=1,
                            candidate_string_valid_count=candidate_string_valid_count,
                            error=e,
                        )
                        rollout_handler_ls[idx].score = 0
                        rollout_handler_ls[idx].done = True
                        print(f"Rollou step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                        return True

                time.sleep(self.config.send_interval)
                with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                    step_dones = list(executor.map(
                        lambda args: direct_agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                    ))
                return all(step_dones)

            while rounds < max_rounds and not all_done_flag:
                _round_start_done = sum(1 for h in rollout_handler_ls if h.done)
                _round_start_active = batch_size - _round_start_done
                _round_start_counts = {
                    "candidate_total": sum(m.get("candidate_total", 0) for m in rollout_monitors),
                    "candidate_string_valid": sum(m.get("candidate_string_valid", 0) for m in rollout_monitors),
                    "taken_total": sum(m.get("taken_total", 0) for m in rollout_monitors),
                    "taken_string_valid": sum(m.get("taken_string_valid", 0) for m in rollout_monitors),
                    "taken_env_valid": sum(m.get("taken_env_valid", 0) for m in rollout_monitors),
                    "string_valid_env_invalid": sum(m.get("string_valid_env_invalid", 0) for m in rollout_monitors),
                    "selected_total": sum(m.get("selected_total", 0) for m in rollout_monitors),
                    "selected_duplicate_action": sum(m.get("selected_duplicate_action", 0) for m in rollout_monitors),
                    "offset_fallback_count": sum(m.get("offset_fallback_count", 0) for m in rollout_monitors),
                    "later_clustered_action": sum(m.get("later_clustered_action", 0) for m in rollout_monitors),
                    "later_skipped_action": sum(m.get("later_skipped_action", 0) for m in rollout_monitors),
                }
                print(
                    f"[rollout] rank={_rollout_rank} round={rounds}/{max_rounds} "
                    f"active={_round_start_active}/{batch_size} t={time.time():.1f}",
                    file=sys.stderr, flush=True,
                )
                if self.clustering_enabled:
                    # 仅 clustering_enabled 时分流到候选聚类；默认仍保持标准 vLLM rollout。
                    if rounds == 0:
                        self._sync_gradient_model_from_actor()
                        rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Round 0 clustering")
                        self._round0_clustering(
                            rollout_handler_ls,
                            env_clients,
                            task_rounds,
                            rounds,
                            max_rounds,
                            kwargs,
                            rollout_monitors,
                            action_records,
                            action_records_lock,
                        )
                    else:
                        active_cnt = sum(1 for h in rollout_handler_ls if not h.done)
                        if self._should_cluster_later_round(rounds, max_rounds):
                            rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active {active_cnt}")
                            self._later_round_clustering(
                                rollout_handler_ls,
                                env_clients,
                                task_rounds,
                                rounds,
                                max_rounds,
                                kwargs,
                                rollout_monitors,
                                action_records,
                                action_records_lock,
                            )
                        else:
                            rollout_bar.set_description(
                                f"Rounds {rounds + 1}/{max_rounds} | Active {active_cnt} | Direct"
                            )
                            all_done_flag = standard_round_step(record_later_skip=True)
                            rounds += 1
                            rollout_bar.update(1)
                            print(
                                f"[rollout] rank={_rollout_rank} round={rounds - 1} "
                                f"done_in_round={sum(1 for h in rollout_handler_ls if h.done) - _round_start_done} "
                                f"total_done={sum(1 for h in rollout_handler_ls if h.done)}/{batch_size} "
                                f"{self._round_valid_summary(rollout_monitors, _round_start_counts)} "
                                f"t={time.time():.1f}",
                                file=sys.stderr, flush=True,
                            )
                            continue
                    all_done_flag = all(h.done for h in rollout_handler_ls)
                else:
                    all_done_flag = standard_round_step()
                _round_end_done = sum(1 for h in rollout_handler_ls if h.done)
                _newly_done = _round_end_done - _round_start_done
                print(
                    f"[rollout] rank={_rollout_rank} round={rounds} "
                    f"done_in_round={_newly_done} total_done={_round_end_done}/{batch_size} "
                    f"{self._round_valid_summary(rollout_monitors, _round_start_counts)} "
                    f"t={time.time():.1f}",
                    file=sys.stderr, flush=True,
                )
                rounds += 1
                rollout_bar.update(1)
            _final_done = sum(1 for h in rollout_handler_ls if h.done)
            _elapsed = time.time() - _rollout_wall_start
            print(
                f"[rollout] rank={_rollout_rank} FINISHED total_rounds={rounds} "
                f"total_done={_final_done}/{batch_size} wall={_elapsed:.1f}s",
                file=sys.stderr, flush=True,
            )

            # rollout 后立刻下放聚类模型，给 compute_log_prob/update_actor 的 FSDP 峰值让显存。
            if (
                self._gradient_model is not None
                and next(self._gradient_model.parameters()).device.type == "cuda"
            ):
                self._gradient_model.cpu()
                torch.cuda.empty_cache()

            # process ids
            rollout_bar.close()
            rollout_bar = None
            response_ids, response_attention_mask, response_position_ids, response_loss_mask = [], [], [], []
            scores, messages = [], []
            
            for rollout_handler in rollout_handler_ls:
                # check length
                rollout_handler.truncate_output_ids()
                assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=}, 
                {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}"""
                assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

                response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
                response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
                response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
                response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
                scores.append(rollout_handler.score)
                messages.append(rollout_handler.messages)
            
            # pad to length
            response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
            if response_ids.shape[1] < self.config.response_length:
                response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
            response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
            if response_attention_mask.shape[1] < self.config.response_length:
                response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
            response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
            if response_loss_mask.shape[1] < self.config.response_length:
                response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
            response_length = response_ids.size(1)
            delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
            delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
            input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
            prompt_length = input_ids.size(-1)
            # left-padded attention_mask
            attention_mask = prompts.batch['attention_mask']
            position_ids = prompts.batch['position_ids']
            input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            response_position_ids = position_ids[:, -1:] + delta_position_ids

            seq = torch.cat((input_ids, response_ids), dim=-1)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
            position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
            response_mask = response_loss_mask

            reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32) # (bs, response_length)
            valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
            for i in range(len(scores)):
                reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if global_steps and self.config.rollout_log_dir:
                try:
                    # actions.jsonl 记录候选与 env 校验结果，便于定位无效动作来源。
                    step_log_dir = os.path.join(self.config.rollout_log_dir, f"step{global_steps}")
                    os.makedirs(step_log_dir, exist_ok=True)
                    rank = torch.distributed.get_rank()
                    with open(os.path.join(step_log_dir, f"{rank}.actions.jsonl"), "w") as f:
                        for record in sorted(
                            action_records,
                            key=lambda r: (r["trajectory_index"], r["round"]),
                        ):
                            if not valid_trajectory_mask[record["trajectory_index"]]:
                                continue
                            f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    with open(os.path.join(step_log_dir, f"{rank}.json"), "w") as f:
                        json_msg = []
                        for idx, msgs in enumerate(messages):
                            if not valid_trajectory_mask[idx]:
                                continue
                            records = {
                                "item_id": rollout_handler_ls[idx].item_id,
                                "conversations": [msg.to_dict() for msg in msgs],
                                "reward": scores[idx]
                            }
                            json_msg.append(records)
                        json.dump(json_msg, f, ensure_ascii=True, indent=4)
                except Exception as e:
                    print(e)
        finally:
            if rollout_bar is not None:
                rollout_bar.close()
            # 环境 client 是外部服务连接，异常退出也要关闭，避免服务端连接泄漏。
            for client in env_clients:
                try:
                    client.close()
                except Exception as e:
                    print(f"Error during closing env: {e}")

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                'scores': reward_tensor,
                'task_rounds': torch.tensor(task_rounds, dtype=torch.float32).to(input_ids.device),
                'task_scores': reward_tensor
            },
            batch_size=batch_size)
        
        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(
            batch=batch,
            # 额外返回 rollout_monitor，训练侧聚合后会删除，不进入 actor 更新。
            non_tensor_batch={"rollout_monitor": np.array(rollout_monitors, dtype=object)},
        )
