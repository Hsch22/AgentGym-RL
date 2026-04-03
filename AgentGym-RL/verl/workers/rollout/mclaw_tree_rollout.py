"""
MClaw Tree Rollout — drop-in replacement for vLLMRollout.

Uses MClaw's tree-search algorithm (MCTS-like) instead of the flat multi-round
agent rollout.  Integrates with the existing Ray/FSDP framework by conforming
to the same ``generate_sequences(DataProto) -> DataProto`` contract.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, List

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM
from vllm import SamplingParams

from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length

# MClaw imports
from mclaw.core import BranchSelector, TreeRollout
from mclaw.core.contracts import TreeRolloutOutput
from mclaw.clustering import HiddenStateClusterer, LogProbClusterer
from mclaw.critic import QCritic, QHead
from mclaw.adapters import (
    AgentEnvClientAdapter,
    DataProtoAdapter,
    VerlInferenceEngine,
)
from mclaw.config import TreeRolloutConfig, QCriticConfig

logger = logging.getLogger(__name__)


def _to_tree_rollout_config(mclaw_cfg: DictConfig) -> TreeRolloutConfig:
    """Convert OmegaConf MClaw config to MClaw's TreeRolloutConfig dataclass."""
    tr_cfg = mclaw_cfg.get("tree_rollout", {})
    return TreeRolloutConfig(
        root_budget=int(tr_cfg.get("root_budget", 256)),
        n_envs=int(tr_cfg.get("n_envs", 16)),
        root_clusters=int(tr_cfg.get("root_clusters", 16)),
        branch_budget=int(tr_cfg.get("branch_budget", 16)),
        intra_branch_clusters=int(tr_cfg.get("intra_branch_clusters", 4)),
        max_rounds=int(tr_cfg.get("max_rounds", 30)),
    )


def _to_qcritic_config(mclaw_cfg: DictConfig) -> QCriticConfig:
    """Convert OmegaConf MClaw config to MClaw's QCriticConfig dataclass."""
    qc_cfg = mclaw_cfg.get("q_critic", {})
    return QCriticConfig(
        hidden_dim=int(qc_cfg.get("hidden_dim", 3584)),
        intermediate_dim=int(qc_cfg.get("intermediate_dim", 1024)),
        lr=float(qc_cfg.get("lr", 1e-4)),
        gamma=float(qc_cfg.get("gamma", 0.99)),
        update_freq=int(qc_cfg.get("update_freq", 1)),
        grad_clip_norm=float(qc_cfg.get("grad_clip_norm", 1.0)),
        micro_batch_size=int(qc_cfg.get("micro_batch_size", 32)),
    )


def _build_clusterer(mclaw_cfg: DictConfig) -> Any:
    """Build the clustering strategy from config."""
    from mclaw.clustering import (
        HiddenStateClusterer,
        LogProbClusterer,
        LogitDistributionClusterer,
        OutputGradClusterer,
    )
    from mclaw.config import ClusteringConfig

    cluster_cfg = mclaw_cfg.get("clustering", {})
    method = str(cluster_cfg.get("method", "hidden_state")).strip().lower()

    # Build a ClusteringConfig from the DictConfig
    clustering_config = ClusteringConfig(
        method=method,
        pca_dim=int(cluster_cfg.get("pca_dim", 128)),
    )

    if method == "hidden_state":
        return HiddenStateClusterer(clustering_config)
    if method == "logprob":
        return LogProbClusterer(clustering_config)
    if method == "logit_distribution":
        return LogitDistributionClusterer(clustering_config)
    if method == "output_grad":
        return OutputGradClusterer(clustering_config)
    raise ValueError(f"unsupported clustering method: {method}")


def _build_env_client_factory(agentgym_config: DictConfig):
    """Build a factory that creates AgentEnvClientAdapter instances."""
    task_name = agentgym_config.get("task_name", "textcraft")
    env_addr = agentgym_config.get("env_addr", "http://127.0.0.1:36006")
    max_retries = int(agentgym_config.get("max_retries", 3))

    args = SimpleNamespace(
        task_name=task_name,
        env_addr=env_addr,
        max_retries=max_retries,
    )

    def _factory() -> AgentEnvClientAdapter:
        return AgentEnvClientAdapter.from_agentgym_args(args)

    return _factory


class MClawTreeRollout(BaseRollout):
    """
    MClaw tree-search rollout that conforms to the verl BaseRollout interface.

    Uses MClaw's TreeRollout internally for MCTS-like exploration, then converts
    the output to DataProto for the existing PPO training pipeline.
    """

    def __init__(
        self,
        actor_module: nn.Module,
        actor_module_fsdp: nn.Module,
        rollout_config: DictConfig,
        agentgym_config: DictConfig,
        mclaw_config: DictConfig,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        super().__init__()
        self.config = rollout_config
        self.agentgym_config = agentgym_config
        self.mclaw_config = mclaw_config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        # ── Build vLLM inference engine (same pattern as vLLMRollout) ────────
        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        self.inference_engine_raw = LLM(
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
            disable_log_stats=rollout_config.get("disable_log_stats", True),
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=rollout_config.get("enable_chunked_prefill", True),
        )
        self.inference_engine_raw.offload_model_weights()

        # Build MClaw's VerlInferenceEngine wrapper
        sampling_kwargs = {
            "max_tokens": int(rollout_config.get("max_tokens", 512)),
            "temperature": float(rollout_config.get("temperature", 1.0)),
            "n": 1,
            "logprobs": int(rollout_config.get("logprobs", 1)),
        }
        for key in ("top_p", "top_k"):
            if key in rollout_config:
                sampling_kwargs[key] = rollout_config.get(key)

        self.verl_inference_engine = VerlInferenceEngine(
            llm=self.inference_engine_raw,
            sampling_kwargs=sampling_kwargs,
        )

        # ── Build Q-Critic ───────────────────────────────────────────────────
        qcritic_config = _to_qcritic_config(mclaw_config)

        # Auto-detect hidden_dim from model config
        actor_config = getattr(actor_module_fsdp, "config", None)
        actor_hidden_dim = getattr(actor_config, "hidden_size", None)
        if actor_hidden_dim is not None:
            qcritic_config.hidden_dim = int(actor_hidden_dim)

        device = next(actor_module_fsdp.parameters()).device
        q_head = QHead(
            hidden_dim=qcritic_config.hidden_dim,
            intermediate_dim=qcritic_config.intermediate_dim,
        ).to(device)

        self.q_critic = QCritic(
            actor_module_fsdp=actor_module_fsdp,
            q_head=q_head,
            tokenizer=tokenizer,
            config=qcritic_config,
        )

        # ── Build TreeRollout components ─────────────────────────────────────
        tree_config = _to_tree_rollout_config(mclaw_config)
        clusterer = _build_clusterer(mclaw_config)
        branch_selector = BranchSelector()
        env_client_factory = _build_env_client_factory(agentgym_config)

        self.tree_rollout = TreeRollout(
            inference_engine=self.verl_inference_engine,
            actor_module_fsdp=actor_module_fsdp,
            q_critic=self.q_critic,
            clusterer=clusterer,
            tokenizer=tokenizer,
            config=tree_config,
            env_client_factory=env_client_factory,
            branch_selector=branch_selector,
        )

        # ── Build DataProtoAdapter ───────────────────────────────────────────
        self.dataproto_adapter = DataProtoAdapter(
            pad_token_id=self.pad_token_id or 0,
        )

        logger.info(
            "MClawTreeRollout initialized: root_budget=%d, n_envs=%d, "
            "root_clusters=%d, branch_budget=%d, max_rounds=%d",
            tree_config.root_budget,
            tree_config.n_envs,
            tree_config.root_clusters,
            tree_config.branch_budget,
            tree_config.max_rounds,
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using MClaw's tree-search algorithm.

        Converts the incoming DataProto to MClaw prompt items, runs tree rollout,
        then converts back to DataProto matching the vLLMRollout output contract.
        """
        if self.config.get("free_cache_engine", False):
            self.inference_engine_raw.init_cache_engine()

        cur_device = prompts.batch["input_ids"].device
        batch_size = prompts.batch["input_ids"].size(0)
        max_rounds = prompts.meta_info.get("max_rounds", 30)

        # Override max_rounds if provided
        if max_rounds != self.tree_rollout.config.max_rounds:
            self.tree_rollout.config.max_rounds = max_rounds

        # ── Convert DataProto -> MClaw prompt items ──────────────────────────
        prompt_items = self._dataproto_to_prompt_items(prompts)

        # ── Run tree rollout ─────────────────────────────────────────────────
        rollout_output: TreeRolloutOutput = self.tree_rollout.generate_tree_rollout(
            prompt_items
        )

        # ── Update Q-head with critic data (TD learning) ────────────────────
        q_metrics = {}
        if rollout_output.critic_data and rollout_output.critic_data.samples:
            q_metrics = self.q_critic.update(rollout_output.critic_data)

        # ── Convert actor_data -> DataProto ──────────────────────────────────
        actor_data = rollout_output.actor_data
        if not actor_data.trajectories:
            # No valid trajectories — return empty batch matching expected shape
            return self._build_empty_output(prompts, cur_device, batch_size)

        # Stash auxiliary batch in actor_data metadata for VerlActorBackend
        if rollout_output.aux_actor_data and rollout_output.aux_actor_data.samples:
            actor_data.metadata["auxiliary_batch"] = rollout_output.aux_actor_data

        dataproto = self.dataproto_adapter.to_dataproto(actor_data)

        # ── Augment DataProto to match vLLMRollout output contract ───────────
        output = self._augment_dataproto(
            dataproto, actor_data, prompts, cur_device, q_metrics
        )

        if self.config.get("free_cache_engine", False):
            self.inference_engine_raw.free_cache_engine()

        return output

    def _dataproto_to_prompt_items(self, prompts: DataProto) -> list[dict]:
        """Convert DataProto batch to list of MClaw prompt item dicts."""
        batch_size = prompts.batch["input_ids"].size(0)
        prompt_items = []

        for i in range(batch_size):
            input_ids = prompts.batch["input_ids"][i]
            attention_mask = prompts.batch["attention_mask"][i]

            # Remove padding (left-padded)
            valid_mask = attention_mask.bool()
            valid_ids = input_ids[valid_mask].tolist()

            item = {
                "input_ids": valid_ids,
                "state_tokens": valid_ids,
            }

            # Extract item_id if available
            if "item_id" in prompts.non_tensor_batch:
                item["item_id"] = prompts.non_tensor_batch["item_id"][i]

            # Extract raw prompt messages if available
            if "raw_prompt" in prompts.non_tensor_batch:
                raw_prompt = prompts.non_tensor_batch["raw_prompt"][i]
                messages = []
                for msg in raw_prompt:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                item["messages"] = messages

            prompt_items.append(item)

        return prompt_items

    def _augment_dataproto(
        self,
        dataproto: DataProto,
        actor_data: Any,
        original_prompts: DataProto,
        cur_device: torch.device,
        q_metrics: dict,
    ) -> DataProto:
        """
        Augment the MClaw-generated DataProto to match vLLMRollout's output.

        Adds: prompts, scores, task_rounds, task_scores, returns tensors.
        """
        actual_batch_size = dataproto.batch["input_ids"].size(0)
        full_seq_len = dataproto.batch["input_ids"].size(1)

        # Determine prompt/response split
        response_portion = dataproto.batch.get("responses")
        if response_portion is not None:
            response_length = response_portion.size(1)
        else:
            # Infer from response_mask
            response_mask = dataproto.batch["response_mask"]
            response_length = response_mask.size(1)

        prompt_length = full_seq_len - response_length
        prompts_tensor = dataproto.batch["input_ids"][:, :prompt_length]

        # Build scores tensor (reward at last valid response token)
        scores = torch.zeros(
            actual_batch_size, response_length, dtype=torch.float32, device=cur_device
        )
        response_mask = dataproto.batch["response_mask"]
        if response_mask.size(1) == response_length:
            valid_response_lengths = response_mask.sum(dim=-1)
        else:
            valid_response_lengths = response_mask[:, prompt_length:].sum(dim=-1)

        for i, trajectory in enumerate(actor_data.trajectories):
            if i >= actual_batch_size:
                break
            total_reward = sum(
                float(step.reward) for step in trajectory.steps
            )
            idx = max(int(valid_response_lengths[i].item()) - 1, 0)
            scores[i, idx] = total_reward

        # Build task_rounds
        task_rounds_list = []
        for trajectory in actor_data.trajectories:
            task_rounds_list.append(float(len(trajectory.steps)))
        task_rounds = torch.tensor(
            task_rounds_list[:actual_batch_size],
            dtype=torch.float32,
            device=cur_device,
        )

        # Build returns from advantages (MClaw tree advantage = td_target based)
        advantages = dataproto.batch.get("advantages")
        if advantages is not None:
            returns = advantages.clone()
        else:
            returns = torch.zeros(
                actual_batch_size, response_length, dtype=torch.float32, device=cur_device
            )

        # Assemble output batch
        output_batch = TensorDict(
            {
                "prompts": prompts_tensor.to(cur_device),
                "responses": dataproto.batch["input_ids"][:, prompt_length:].to(cur_device),
                "input_ids": dataproto.batch["input_ids"].to(cur_device),
                "attention_mask": dataproto.batch["attention_mask"].to(cur_device),
                "position_ids": dataproto.batch["position_ids"].to(cur_device),
                "response_mask": response_mask.to(cur_device)
                if response_mask.size(1) == response_length
                else response_mask[:, prompt_length:].to(cur_device),
                "scores": scores,
                "task_rounds": task_rounds,
                "task_scores": scores.clone(),
                "advantages": advantages.to(cur_device) if advantages is not None else returns,
                "returns": returns.to(cur_device),
            },
            batch_size=actual_batch_size,
        )

        output = DataProto(batch=output_batch)

        # Carry over meta_info
        output.meta_info.update(dataproto.meta_info)
        output.meta_info["q_critic_metrics"] = q_metrics

        # Stash auxiliary data for actor update
        if "auxiliary_batch" in actor_data.metadata:
            output.meta_info["mclaw_auxiliary_batch"] = actor_data.metadata[
                "auxiliary_batch"
            ]

        # Carry over non-tensor batch
        if hasattr(dataproto, "non_tensor_batch") and dataproto.non_tensor_batch:
            output.non_tensor_batch.update(dataproto.non_tensor_batch)

        return output

    def _build_empty_output(
        self, prompts: DataProto, cur_device: torch.device, batch_size: int
    ) -> DataProto:
        """Build an empty DataProto when tree rollout produces no trajectories."""
        prompt_length = prompts.batch["input_ids"].size(1)
        response_length = int(self.config.get("response_length", self.config.get("max_tokens", 512)))

        empty_responses = torch.full(
            (batch_size, response_length),
            fill_value=self.pad_token_id or 0,
            dtype=torch.long,
            device=cur_device,
        )
        empty_mask = torch.zeros(
            batch_size, response_length, dtype=torch.long, device=cur_device
        )
        empty_scores = torch.zeros(
            batch_size, response_length, dtype=torch.float32, device=cur_device
        )

        input_ids = torch.cat(
            [prompts.batch["input_ids"], empty_responses], dim=-1
        )
        attention_mask = torch.cat(
            [prompts.batch["attention_mask"], empty_mask], dim=-1
        )
        delta_pos = torch.arange(1, response_length + 1, device=cur_device)
        delta_pos = delta_pos.unsqueeze(0).repeat(batch_size, 1)
        position_ids = torch.cat(
            [prompts.batch["position_ids"], prompts.batch["position_ids"][:, -1:] + delta_pos],
            dim=-1,
        )

        batch = TensorDict(
            {
                "prompts": prompts.batch["input_ids"],
                "responses": empty_responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": empty_mask,
                "scores": empty_scores,
                "task_rounds": torch.zeros(batch_size, dtype=torch.float32, device=cur_device),
                "task_scores": empty_scores.clone(),
                "advantages": torch.zeros(
                    batch_size, response_length, dtype=torch.float32, device=cur_device
                ),
                "returns": torch.zeros(
                    batch_size, response_length, dtype=torch.float32, device=cur_device
                ),
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch)
