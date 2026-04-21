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

        # Offload vllm model to reduce peak memory usage
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

        # ==== rank (for per-rank rollout logging) ====
        try:
            if torch.distributed.is_initialized():
                self._rank = torch.distributed.get_rank()
            else:
                self._rank = int(os.environ.get("RANK", "0"))
        except Exception:
            self._rank = int(os.environ.get("RANK", "0"))

        # ==== clustering setup ====
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
            if method not in ("gradient", "semantic"):
                raise ValueError(f"unsupported clustering method: {method}")
            assert round1_candidates >= round1_clusters >= 1, (
                f"invalid round0 clustering config: {round1_candidates=} {round1_clusters=}"
            )
            assert later_candidates >= later_clusters >= 1, (
                f"invalid later-round clustering config: {later_candidates=} {later_clusters=}"
            )
            assert round1_clusters == int(self.config.n), (
                f"round0 clustering must match rollout.n so each repeated trajectory gets one center: "
                f"{round1_clusters=} rollout.n={self.config.n}"
            )
            print(
                "[clustering-config] "
                f"method={method} "
                f"round0={round1_candidates}/{round1_clusters} "
                f"later={later_candidates}/{later_clusters}"
            )
        # Both gradient and semantic clustering need a standalone (non-FSDP) HF copy.
        # Semantic used to call model(...) on the FSDP actor directly; that triggers
        # an all_gather collective under divergent control flow (ranks that finish
        # their rollout early never enter the call) and deadlocks.
        if self.clustering_enabled and self.clustering_config.method in ("gradient", "semantic"):
            from transformers import AutoModelForCausalLM
            gradient_model_path = self.clustering_config.gradient_model_path
            self._gradient_model = AutoModelForCausalLM.from_pretrained(
                gradient_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            # Keep on CPU initially to avoid competing with vLLM KV cache + FSDP
            # state_dict gather during compute_log_prob / update_actor. Moved to
            # GPU at sync time (round 0) and offloaded back at end of rollout.
            self._gradient_model.cpu()
            self._gradient_model.eval()
            # Gradient method needs backward through params; explicit for safety.
            for p in self._gradient_model.parameters():
                p.requires_grad_(True)


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
        """Sync FSDP actor_module weights to the standalone clustering model.

        Called at the start of each rollout (rounds==0), after PPO has updated
        the actor. Runs on all ranks in lockstep — safe because every rank enters
        generate_sequences together and hits round 0 before any divergence.
        Covers both gradient and semantic methods.
        """
        if not self.clustering_enabled:
            return
        if self.clustering_config.method not in ("gradient", "semantic"):
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
        # Move clustering model to GPU for the upcoming round-0 + later-round calls.
        if next(self._gradient_model.parameters()).device.type == "cpu":
            self._gradient_model.cuda()
        torch.cuda.empty_cache()

    def _get_clustering_model(self):
        method = self.clustering_config.method
        if method in ("gradient", "semantic"):
            # Both routes use the standalone non-FSDP copy to avoid collective
            # deadlock in divergent-control-flow paths (some ranks exit rollout
            # early while others still call model(...)).
            return self._gradient_model
        raise ValueError(f"Unknown clustering method: {method}")

    def _select_centers_with_grad(self, model, obs_token_ids, response_texts, k):
        """Wrap clustering.select_centers with appropriate grad context."""
        method = self.clustering_config.method
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
        return _clustering.select_centers(
            method="semantic",
            model=model,
            tokenizer=self.tokenizer,
            obs_token_ids=obs_token_ids,
            response_texts=response_texts,
            k=k,
        )

    def _round0_clustering(self, rollout_handler_ls, env_clients, task_rounds, rounds, kwargs_sp):
        """Round 0: generate round1_candidates per unique prompt, cluster to round1_clusters."""
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

            # sample n_candidates via native vLLM n=K
            sp_kwargs = dict(kwargs_sp)
            sp_kwargs['n'] = n_candidates
            with self.update_sampling_params(**sp_kwargs):
                output = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=[gen_prompt],
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
            # output[0] is a padded tensor of shape (n_candidates, max_len)
            all_response_ids = output[0].tolist()
            all_response_texts = [
                self.tokenizer.decode(r, skip_special_tokens=True) for r in all_response_ids
            ]

            # Filter to valid normalized actions only.
            # Note: clustering input = raw response text (full Thought+Action), not
            # the normalized action — Thought adds semantic signal for diversity;
            # env.step still receives the raw text downstream.
            valid_indices = [
                i for i, t in enumerate(all_response_texts)
                if _clustering.parse_valid_action(t) is not None
            ]
            valid_texts = [all_response_texts[i] for i in valid_indices]

            # Cluster to min(n_clusters, len(valid_texts)); fall back to first raw
            # text cycled across handlers when all candidates parse-fail.
            if len(valid_texts) == 0:
                selected_response_texts = [all_response_texts[0]]
            else:
                k_eff = min(n_clusters, len(valid_texts))
                center_local_idxs = self._select_centers_with_grad(
                    model=clustering_model,
                    obs_token_ids=gen_prompt,
                    response_texts=valid_texts,
                    k=k_eff,
                )
                selected_response_texts = [valid_texts[i] for i in center_local_idxs]

            # Assign one center to each handler and interact. Cycle with modulo when
            # len(handler_idxs) > len(selected_response_texts) (e.g., rollout.n > 1
            # or collapsed candidate pool).
            time.sleep(self.config.send_interval)
            for assign_i, hidx in enumerate(handler_idxs):
                resp_text = selected_response_texts[assign_i % len(selected_response_texts)]
                handler = rollout_handler_ls[hidx]
                handler.add_assistant_message(self.tokenizer, resp_text)
                task_rounds[hidx] += 1
                try:
                    step_output = env_clients[hidx].step(resp_text)
                    handler.score = step_output.reward
                    handler.done = step_output.done
                    if not step_output.done:
                        handler.add_user_message(self.tokenizer, step_output.state)
                except Exception as e:
                    handler.score = 0
                    handler.done = True
                    print(f"Round 0 step Error: {e} item id = {handler.item_id}")

    def _later_round_clustering(self, rollout_handler_ls, env_clients, task_rounds, rounds, kwargs_sp):
        """Round 1+: each not-done handler generates later_candidates, cluster later_clusters, pick 1 random center."""
        n_candidates = self.clustering_config.later_candidates
        n_clusters = self.clustering_config.later_clusters
        clustering_model = self._get_clustering_model()

        not_done = [(idx, h) for idx, h in enumerate(rollout_handler_ls) if not h.done]
        if not not_done:
            return

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

            valid_indices = [
                i for i, t in enumerate(all_response_texts)
                if _clustering.parse_valid_action(t) is not None
            ]
            valid_texts = [all_response_texts[i] for i in valid_indices]

            if len(valid_texts) == 0:
                # All invalid; fallback to first raw response (env will return error state)
                chosen_text = all_response_texts[0]
            else:
                k_eff = min(n_clusters, len(valid_texts))
                center_local_idxs = self._select_centers_with_grad(
                    model=clustering_model,
                    obs_token_ids=gen_prompt,
                    response_texts=valid_texts,
                    k=k_eff,
                )
                chosen_local = random.choice(center_local_idxs)
                chosen_text = valid_texts[chosen_local]

            handler.add_assistant_message(self.tokenizer, chosen_text)
            task_rounds[handler_idx] += 1
            try:
                step_output = env_clients[handler_idx].step(chosen_text)
                handler.score = step_output.reward
                handler.done = step_output.done
                if not step_output.done:
                    handler.add_user_message(self.tokenizer, step_output.state)
            except Exception as e:
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
            rollout_bar = tqdm(total = max_rounds, desc="Running rounds", disable=torch.distributed.get_rank() != 0)
            _rollout_rank = getattr(self, "_rank", int(os.environ.get("RANK", "0")))
            _rollout_wall_start = time.time()
            print(
                f"[rollout] rank={_rollout_rank} START batch_size={batch_size} max_rounds={max_rounds} t={_rollout_wall_start:.1f}",
                file=sys.stderr, flush=True,
            )
            def agent_step(i, idx):
                content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)
                rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
                task_rounds[idx] += 1
                try:
                    step_output = env_clients[idx].step(content)
                    state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                        step_output.state,
                        step_output.reward,
                        step_output.done,
                    )
                    rollout_handler_ls[idx].add_user_message(self.tokenizer, state)
                    return step_output.done
                except Exception as e:
                    rollout_handler_ls[idx].score = 0
                    rollout_handler_ls[idx].done = True
                    print(f"Rollou step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                    return True
            while rounds < max_rounds and not all_done_flag:
                _round_start_done = sum(1 for h in rollout_handler_ls if h.done)
                _round_start_active = batch_size - _round_start_done
                print(
                    f"[rollout] rank={_rollout_rank} round={rounds}/{max_rounds} "
                    f"active={_round_start_active}/{batch_size} t={time.time():.1f}",
                    file=sys.stderr, flush=True,
                )
                if self.clustering_enabled:
                    if rounds == 0:
                        self._sync_gradient_model_from_actor()
                        rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Round 0 clustering")
                        self._round0_clustering(rollout_handler_ls, env_clients, task_rounds, rounds, kwargs)
                    else:
                        active_cnt = sum(1 for h in rollout_handler_ls if not h.done)
                        rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active {active_cnt}")
                        self._later_round_clustering(rollout_handler_ls, env_clients, task_rounds, rounds, kwargs)
                    all_done_flag = all(h.done for h in rollout_handler_ls)
                else:
                    # get generation prompt
                    generation_prompt_idxs = []
                    not_done_idxs = []
                    for idx, rollout_handler in enumerate(rollout_handler_ls):
                        if not rollout_handler.done:
                            generation_prompt_idxs.append(rollout_handler.get_generation_prompt(self.tokenizer))
                            not_done_idxs.append(idx)

                    rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active agents per gpu: {len(not_done_idxs)}")
                    # users can customize different sampling_params at different run
                    with self.update_sampling_params(**kwargs):
                        output = self.inference_engine.generate(
                            prompts=None,
                            prompt_token_ids=generation_prompt_idxs,
                            sampling_params=self.sampling_params,
                            use_tqdm=False)
                    response_ids = output[0].tolist()
                    all_done_flag = True
                    time.sleep(self.config.send_interval) # take a break before sendng request
                    if len(not_done_idxs) > 0:
                        with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                            step_dones = list(executor.map(
                                lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                            ))
                            all_done_flag = all(step_dones)
                _round_end_done = sum(1 for h in rollout_handler_ls if h.done)
                _newly_done = _round_end_done - _round_start_done
                print(
                    f"[rollout] rank={_rollout_rank} round={rounds} "
                    f"done_in_round={_newly_done} total_done={_round_end_done}/{batch_size} "
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

            # Offload clustering model to CPU so compute_log_prob / update_actor
            # don't fight its ~6GB against FSDP all-gather peaks. Re-uploaded at
            # next round-0 inside _sync_gradient_model_from_actor.
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

            if global_steps:
                try:
                    os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                    with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                        json_msg = []
                        for idx, msgs in enumerate(messages):
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

        return DataProto(batch=batch)
