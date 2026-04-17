"""
clustering.py — Self-contained clustering utilities for AgentGym-RL rollout.

Provides gradient-based and semantic-based cluster-center selection over
candidate LLM responses. All logic is copied/adapted from RLclaw
(rlclaw.analysis.kcenter and rlclaw.analysis.gradient_action_selection) so
that this module has NO dependency on the rlclaw package.

Public API
----------
parse_valid_action(text)                          -> Optional[str]
select_centers_gradient(model, tokenizer, ...)   -> List[int]
select_centers_semantic(model, tokenizer, ...)   -> List[int]
select_centers(method, model, tokenizer, ...)    -> List[int]
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KCenter greedy clustering (verbatim copy from rlclaw/analysis/kcenter.py)
# ---------------------------------------------------------------------------

_CENTER_TIE_BREAK_EPS = 1e-4


@dataclass(frozen=True)
class KCenterGreedyResult:
    center_indices: List[int]
    nearest_center_ranks: List[int]
    nearest_distances: List[float]


def kcenter_greedy(
    embeddings: torch.Tensor,
    k: int,
    *,
    initial_center_index: int = 0,
) -> KCenterGreedyResult:
    """Greedy k-center clustering on L2 distance.

    Parameters
    ----------
    embeddings:
        2-D float tensor of shape ``[N, D]``.
    k:
        Number of cluster centers to select.
    initial_center_index:
        Index of the first center (default 0).

    Returns
    -------
    KCenterGreedyResult
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected 2D embeddings, got shape={tuple(embeddings.shape)}")

    num_points = embeddings.shape[0]
    if num_points == 0:
        raise ValueError("cannot cluster empty embeddings")
    if k < 1 or k > num_points:
        raise ValueError(f"expected 1 <= k <= {num_points}, got k={k}")
    if initial_center_index < 0 or initial_center_index >= num_points:
        raise ValueError(
            f"initial_center_index must be in [0, {num_points}), got {initial_center_index}"
        )

    points = embeddings.to(dtype=torch.float32)
    device = points.device
    selected_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
    nearest_distances = torch.full(
        (num_points,), float("inf"), dtype=torch.float32, device=device
    )
    nearest_center_ranks = torch.zeros(num_points, dtype=torch.long, device=device)

    center_indices: List[int] = []
    next_center_index = int(initial_center_index)
    for center_rank in range(k):
        center_indices.append(next_center_index)
        selected_mask[next_center_index] = True

        center_vector = points[next_center_index : next_center_index + 1]
        distances = torch.cdist(points, center_vector, p=2).squeeze(1)
        update_mask = distances < nearest_distances
        nearest_distances = torch.minimum(nearest_distances, distances)
        nearest_center_ranks = torch.where(
            update_mask,
            torch.full_like(nearest_center_ranks, center_rank),
            nearest_center_ranks,
        )

        if center_rank == k - 1:
            break

        candidate_distances = nearest_distances.clone()
        candidate_distances[selected_mask] = -1.0
        max_distance = torch.max(candidate_distances)
        tie_mask = torch.logical_and(
            torch.logical_not(selected_mask),
            candidate_distances >= (max_distance - _CENTER_TIE_BREAK_EPS),
        )
        if torch.any(tie_mask):
            next_center_index = int(torch.nonzero(tie_mask, as_tuple=False)[0].item())
        else:
            next_center_index = int(torch.argmax(candidate_distances).item())

    return KCenterGreedyResult(
        center_indices=center_indices,
        nearest_center_ranks=nearest_center_ranks.cpu().tolist(),
        nearest_distances=nearest_distances.cpu().tolist(),
    )


# ---------------------------------------------------------------------------
# CountSketch + gradient helpers
# (adapted from rlclaw/analysis/gradient_action_selection.py)
# ---------------------------------------------------------------------------

def _compute_action_log_prob(
    logits: torch.Tensor,
    *,
    obs_length: int,
    action_token_ids: List[int],
) -> torch.Tensor:
    """Compute sum of per-token log-probs for the action tokens.

    Parameters
    ----------
    logits:
        Shape ``[1, seq_len, vocab_size]`` — output of a causal LM forward pass,
        WITH gradients enabled.
    obs_length:
        Number of prompt (observation) tokens.
    action_token_ids:
        Token IDs of the action.

    Returns
    -------
    Scalar tensor (grad-enabled).
    """
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(
            f"expected logits shape [1, seq, vocab], got {tuple(logits.shape)}"
        )
    if obs_length < 1:
        raise ValueError("obs_length must be >= 1")
    if not action_token_ids:
        raise ValueError("action_token_ids must be non-empty")

    sequence_length = logits.shape[1]
    full_length = obs_length + len(action_token_ids)
    if sequence_length < full_length:
        raise ValueError(
            f"logits sequence_length={sequence_length} shorter than "
            f"required full_length={full_length}"
        )

    # The token at position i predicts position i+1.
    # Action starts at index obs_length, so predictions come from positions
    # obs_length-1 .. obs_length+len(action_token_ids)-2.
    prediction_positions = torch.arange(
        obs_length - 1,
        obs_length + len(action_token_ids) - 1,
        device=logits.device,
        dtype=torch.long,
    )
    target_ids = torch.tensor(action_token_ids, device=logits.device, dtype=torch.long)
    next_token_logits = logits[0, prediction_positions, :]
    token_log_probs = F.log_softmax(next_token_logits.float(), dim=-1)
    return token_log_probs.gather(1, target_ids.unsqueeze(1)).sum()


def _count_sketch_gradients(
    model: torch.nn.Module,
    *,
    d_proj: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply CountSketch projection to all parameter gradients.

    Uses SHA-256 of the parameter name as a deterministic seed so that the
    same parameter always maps to the same hash/sign pair across calls.

    Parameters
    ----------
    model:
        The model whose ``.grad`` attributes will be read.
    d_proj:
        Output sketch dimension.
    device:
        Device on which to accumulate the sketch (typically the GPU).

    Returns
    -------
    ``[d_proj]`` float32 tensor on *device*.
    """
    sketch = torch.zeros(d_proj, dtype=torch.float32, device=device)

    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue

        grad_flat = grad.detach().flatten().float()
        if grad_flat.numel() == 0:
            continue

        # Deterministic seed from parameter name
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 31)
        generator = torch.Generator(device=grad.device)
        generator.manual_seed(seed)

        buckets = torch.randint(
            low=0,
            high=d_proj,
            size=(grad_flat.numel(),),
            generator=generator,
            device=grad.device,
            dtype=torch.long,
        )
        signs = torch.randint(
            low=0,
            high=2,
            size=(grad_flat.numel(),),
            generator=generator,
            device=grad.device,
            dtype=torch.long,
        )
        signed_grad = grad_flat * (signs.to(torch.float32).mul_(2.0).sub_(1.0))
        sketch.scatter_add_(0, buckets, signed_grad)

    return sketch


def _select_sketch_centers(
    sketches: Sequence[torch.Tensor],
    *,
    k: int,
) -> Tuple[List[int], List[int], List[float]]:
    """L2-normalize sketch vectors, then run greedy k-center with a random initial center.

    Parameters
    ----------
    sketches:
        Sequence of 1-D float tensors (one per candidate), each of length ``d_proj``.
    k:
        Number of centers to select.

    Returns
    -------
    (center_indices, nearest_center_ranks, nearest_distances)
    """
    if not sketches:
        raise ValueError("sketches must be non-empty")

    sketch_matrix = torch.stack(list(sketches), dim=0)
    normalized_sketches = F.normalize(sketch_matrix, p=2, dim=-1, eps=1e-12)
    first_center_index = int(
        torch.randint(low=0, high=normalized_sketches.shape[0], size=(1,)).item()
    )
    cluster_result = kcenter_greedy(
        normalized_sketches,
        k,
        initial_center_index=first_center_index,
    )
    center_vectors = normalized_sketches[cluster_result.center_indices]
    center_distances = 1.0 - torch.matmul(normalized_sketches, center_vectors.T)
    nearest_distances_t, nearest_ranks_t = torch.min(center_distances, dim=1)
    return (
        cluster_result.center_indices,
        nearest_ranks_t.cpu().tolist(),
        nearest_distances_t.cpu().tolist(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_valid_action(text: str) -> Optional[str]:
    """Extract the normalized action from a raw LLM response.

    Logic matches ``AgentGym/agentenv/agentenv/envs/textcraft.py`` ``step()``:

    1. Apply ``re.findall(r"Action:\\s*(.*?)(?=\\n|$)", text)``.
    2. Require exactly **one** match; otherwise return ``None``.
    3. Normalize: ``re.sub(r"[^A-Za-z0-9, ]+", "", raw)``, then
       ``" ".join(normalized.split()).strip()``.
    4. Return ``None`` if the normalized string is empty.

    Parameters
    ----------
    text:
        Raw response text from the LLM (may contain thought + action).

    Returns
    -------
    Normalized action string, or ``None`` if the response is invalid.
    """
    matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", text)
    if len(matches) != 1:
        return None
    raw = matches[0]
    normalized = re.sub(r"[^A-Za-z0-9, ]+", "", raw)
    normalized = " ".join(normalized.split()).strip()
    return normalized if normalized else None


def select_centers_gradient(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
    d_proj: int = 512,
) -> List[int]:
    """Select k cluster-center indices using gradient-based (CountSketch) clustering.

    For each candidate response the function computes the gradient of the
    action log-probability w.r.t. all model parameters, projects it into a
    low-dimensional sketch via CountSketch, and then runs greedy k-center on
    the normalized sketches.

    Parameters
    ----------
    model:
        HF CausalLM actor module.  ``backward()`` will be called; the caller
        is responsible for restoring train/eval state if needed.
    tokenizer:
        Tokenizer compatible with *model*.
    obs_token_ids:
        Prompt token IDs (shared across all candidates).
    response_texts:
        List of N raw response strings.
    k:
        Number of centers to return.
    d_proj:
        CountSketch projection dimension (default 512).

    Returns
    -------
    List of ``min(k, N)`` indices into *response_texts*.
    """
    device = next(model.parameters()).device
    obs_ids = [int(x) for x in obs_token_ids]

    # Tokenize all response texts up front
    action_ids_list: List[List[int]] = []
    for text in response_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        action_ids_list.append([int(x) for x in ids])

    model.zero_grad(set_to_none=True)
    sketches: List[torch.Tensor] = []
    try:
        for action_ids in action_ids_list:
            input_ids = torch.tensor(
                [obs_ids + action_ids], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids)
            model.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            log_prob = _compute_action_log_prob(
                outputs.logits,
                obs_length=len(obs_ids),
                action_token_ids=action_ids,
            )
            log_prob.backward()

            sketch = _count_sketch_gradients(model, d_proj=d_proj, device=device).cpu()
            sketches.append(sketch)

            del outputs, log_prob, attention_mask, input_ids
            model.zero_grad(set_to_none=True)

        effective_k = min(k, len(sketches))
        center_indices, _, _ = _select_sketch_centers(sketches, k=effective_k)
    finally:
        model.zero_grad(set_to_none=True)

    return center_indices


def select_centers_semantic(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
) -> List[int]:
    """Select k cluster-center indices using semantic (hidden-state) clustering.

    Builds a padded batch of ``[prompt + response]`` sequences, runs a
    no-grad forward pass to get the last hidden layer, mean-pools the
    response token embeddings, L2-normalises, then runs greedy k-center.

    Parameters
    ----------
    model:
        HF CausalLM actor module (used for ``output_hidden_states=True``).
    tokenizer:
        Tokenizer compatible with *model*.
    obs_token_ids:
        Prompt token IDs (shared across all candidates).
    response_texts:
        List of N raw response strings.
    k:
        Number of centers to return.

    Returns
    -------
    List of ``min(k, N)`` indices into *response_texts*.
    """
    device = next(model.parameters()).device
    obs_ids = [int(x) for x in obs_token_ids]
    obs_len = len(obs_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    full_ids_list: List[List[int]] = []
    resp_lens: List[int] = []
    for text in response_texts:
        resp_ids = tokenizer.encode(text, add_special_tokens=False)
        full_ids_list.append(obs_ids + [int(x) for x in resp_ids])
        resp_lens.append(len(resp_ids))

    max_len = max(len(x) for x in full_ids_list)
    N = len(full_ids_list)
    input_ids = torch.full((N, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(N, max_len, dtype=torch.long, device=device)
    spans: List[Tuple[int, int]] = []
    for i, ids in enumerate(full_ids_list):
        L = len(ids)
        pad_len = max_len - L
        input_ids[i, pad_len:] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[i, pad_len:] = 1
        resp_start = pad_len + obs_len
        spans.append((resp_start, resp_start + resp_lens[i]))

    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

    last_hidden = outputs.hidden_states[-1]  # [N, seq_len, hidden_dim]

    pooled_list = []
    for i, (start, end) in enumerate(spans):
        pooled_list.append(last_hidden[i, start:end, :].mean(dim=0))
    pooled = torch.stack(pooled_list, dim=0).float()  # [N, hidden_dim]
    pooled = F.normalize(pooled, p=2, dim=-1)

    effective_k = min(k, N)
    result = kcenter_greedy(pooled, effective_k, initial_center_index=0)
    return result.center_indices


def select_centers(
    method: str,
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
    d_proj: int = 512,
) -> List[int]:
    """Dispatcher: select cluster centers by *method*.

    Parameters
    ----------
    method:
        ``"gradient"`` or ``"semantic"``.
    model:
        HF CausalLM actor module.
    tokenizer:
        Tokenizer compatible with *model*.
    obs_token_ids:
        Prompt token IDs.
    response_texts:
        Candidate response strings.
    k:
        Number of centers.
    d_proj:
        CountSketch projection dim (used only for gradient method).

    Returns
    -------
    List of indices into *response_texts*.
    """
    if method == "gradient":
        return select_centers_gradient(
            model, tokenizer, obs_token_ids, response_texts, k, d_proj=d_proj
        )
    if method == "semantic":
        return select_centers_semantic(
            model, tokenizer, obs_token_ids, response_texts, k
        )
    raise ValueError(f"Unknown clustering method: {method!r}")
