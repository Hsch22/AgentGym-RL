"""
clustering.py — Self-contained clustering utilities for AgentGym-RL rollout.

Provides gradient-based, semantic-based, and TextCraft multi-view gradient
cluster-center selection over candidate LLM responses. Most legacy logic is
copied/adapted from RLclaw
(rlclaw.analysis.kcenter and rlclaw.analysis.gradient_action_selection) so
that this module has NO dependency on the rlclaw package.

Public API
----------
parse_valid_action(text)                          -> Optional[str]
select_centers_gradient(model, tokenizer, ...)   -> List[int]
select_centers_semantic(model, tokenizer, ...)   -> List[int]
select_centers_gradient_multiview(...)           -> List[int]
select_centers_g2rl_action_gradient(...)         -> List[int]
select_centers_g2rl_normalized_action_gradient(...) -> List[int]
select_centers_quality_unique_action(...)        -> List[int]
select_centers(method, model, tokenizer, ...)    -> List[int]
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import List, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KCenter greedy clustering (verbatim copy from rlclaw/analysis/kcenter.py)
# ---------------------------------------------------------------------------

_CENTER_TIE_BREAK_EPS = 1e-4
_DEFAULT_SEMANTIC_CHUNK_SIZE = 4
_SEMANTIC_CHUNK_ENV = "VERL_SEMANTIC_CLUSTER_CHUNK_SIZE"
_DEFAULT_MULTIVIEW_TOPK = 256
_DEFAULT_MULTIVIEW_CHUNK_SIZE = 4
_MULTIVIEW_TOKEN_BATCH_SIZE = 16
_MULTIVIEW_EPS = 1e-8
_TEXTCRAFT_ACTION_RE = re.compile(r"Action:\s*(.*?)(?=\n|$)")
_ACTION_NORMALIZER_ALIASES = {
    "": "textcraft",
    "default": "textcraft",
    "textcraft": "textcraft",
    "sciworld": "sciworld",
    "scienceworld": "sciworld",
}


@dataclass(frozen=True)
class KCenterGreedyResult:
    center_indices: List[int]
    nearest_center_ranks: List[int]
    nearest_distances: List[float]


@dataclass(frozen=True)
class TextCraftCandidate:
    raw_index: int
    raw_text: str
    thought_text: str
    action_text: str
    normalized_action: str


@dataclass
class _MultiviewCandidateFeature:
    raw_index: int
    normalized_action: str
    action_feature: torch.Tensor
    thought_feature: torch.Tensor
    thought_residual: torch.Tensor
    mean_logprob: float


@dataclass
class _G2RLCandidateFeature:
    raw_index: int
    normalized_action: str
    feature: torch.Tensor


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

def _normalize_textcraft_action(raw: str) -> Optional[str]:
    normalized = re.sub(r"[^A-Za-z0-9, ]+", "", raw)
    normalized = " ".join(normalized.split()).strip()
    return normalized if normalized else None


def resolve_action_normalizer(action_normalizer: Optional[str] = None) -> str:
    normalizer = str(action_normalizer or "textcraft").strip().lower()
    normalizer = _ACTION_NORMALIZER_ALIASES.get(normalizer, normalizer)
    if normalizer not in {"textcraft", "sciworld"}:
        raise ValueError(
            f"Unsupported action_normalizer={action_normalizer!r}; "
            "expected 'textcraft' or 'sciworld'"
        )
    return normalizer


def _normalize_sciworld_action(raw: str) -> Optional[str]:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.splitlines()[0].strip()
    if text.endswith("</s>"):
        text = text[:-4].strip()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = " ".join(text.lower().split()).strip()
    if not text:
        return None

    text = re.sub(r"^(?:i will|i should|i need to|let me|now i will)\s+", "", text)
    if text.isdigit():
        return text
    choose_match = re.fullmatch(r"choose\s+([0-9]+)", text)
    if choose_match:
        return choose_match.group(1)
    if text in {"wait1", "wait 1", "wait one"}:
        return "wait1"
    if text in {"wait", "wait10", "wait 10", "wait ten"}:
        return "wait"

    aliases = (
        ("look around", "look around"),
        ("lookaround", "look around"),
        ("look at", "look at"),
        ("lookat", "look at"),
        ("look in", "look in"),
        ("lookin", "look in"),
        ("pick up", "pick up"),
        ("pickup", "pick up"),
        ("put down", "drop"),
        ("putdown", "drop"),
        ("go to", "go to"),
        ("goto", "go to"),
        ("focus on", "focus on"),
        ("focus", "focus on"),
        ("deactivate", "deactivate"),
        ("activate", "activate"),
        ("disconnect", "disconnect"),
        ("connect", "connect"),
        ("inventory", "inventory"),
        ("examine", "examine"),
        ("close", "close"),
        ("open", "open"),
        ("read", "read"),
        ("move", "move"),
        ("drop", "drop"),
        ("pour", "pour"),
        ("dunk", "dunk"),
        ("mix", "mix"),
        ("use", "use"),
        ("eat", "eat"),
        ("flush", "flush"),
        ("task", "task"),
    )
    for alias, canonical in aliases:
        if text == alias:
            return canonical
        if text.startswith(alias + " "):
            suffix = text[len(alias):].strip()
            return f"{canonical} {suffix}".strip()
    return text


def _parse_action_candidate_with_spans(
    raw_index: int,
    text: str,
    *,
    action_normalizer: Optional[str] = None,
) -> Optional[Tuple[TextCraftCandidate, Tuple[int, int], Tuple[int, int]]]:
    normalizer = resolve_action_normalizer(action_normalizer)
    matches = list(_TEXTCRAFT_ACTION_RE.finditer(text))
    if normalizer == "textcraft":
        if len(matches) != 1:
            return None
        match = matches[0]
        normalized = _normalize_textcraft_action(match.group(1))
    else:
        if not matches:
            return None
        match = matches[-1]
        normalized = _normalize_sciworld_action(match.group(1))

    if normalized is None:
        return None

    raw_action = match.group(1)
    thought_span = (0, match.start())
    action_span = match.span(1)
    candidate = TextCraftCandidate(
        raw_index=raw_index,
        raw_text=text,
        thought_text=text[thought_span[0]:thought_span[1]].strip(),
        action_text=raw_action.strip(),
        normalized_action=normalized,
    )
    return candidate, thought_span, action_span


def _parse_textcraft_candidate_with_spans(
    raw_index: int,
    text: str,
) -> Optional[Tuple[TextCraftCandidate, Tuple[int, int], Tuple[int, int]]]:
    return _parse_action_candidate_with_spans(
        raw_index,
        text,
        action_normalizer="textcraft",
    )


def parse_textcraft_candidate(raw_index: int, text: str) -> Optional[TextCraftCandidate]:
    parsed = _parse_textcraft_candidate_with_spans(raw_index, text)
    if parsed is None:
        return None
    candidate, _, _ = parsed
    return candidate


def parse_valid_action(text: str, action_normalizer: Optional[str] = None) -> Optional[str]:
    """Extract the normalized action from a raw LLM response.

    ``action_normalizer='textcraft'`` preserves the TextCraft rule used in
    the original ablation: require exactly one ``Action:`` line and strip
    punctuation except commas. ``action_normalizer='sciworld'`` uses the last
    ``Action:`` line, lowercases it, collapses common SciWorld command aliases
    such as ``pickup``/``pick up`` and ``put down``/``drop``, and keeps the
    executable action intent as the compact feature text.

    Parameters
    ----------
    text:
        Raw response text from the LLM (may contain thought + action).

    Returns
    -------
    Normalized action string, or ``None`` if the response is invalid.
    """
    parsed = _parse_action_candidate_with_spans(
        0,
        text,
        action_normalizer=action_normalizer,
    )
    if parsed is None:
        return None
    candidate, _, _ = parsed
    return None if candidate is None else candidate.normalized_action


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

    Builds padded ``[prompt + response]`` sequences, runs chunked no-grad
    forward passes to get the last hidden layer, mean-pools the response token
    embeddings, L2-normalises, then runs greedy k-center.

    The chunking is intentionally conservative: semantic clustering is an
    auxiliary path, and stability matters more than throughput. If a chunk
    still OOMs, this function recursively halves it until it succeeds or only a
    single candidate remains.

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

    N = len(full_ids_list)

    # Skip candidates whose response tokenized to empty — mean over an empty
    # span returns NaN and poisons k-center. Keep a map back to the original
    # response_texts index so callers can index into their own list.
    pooled_list: List[torch.Tensor] = []
    original_indices: List[int] = []

    def _empty_cuda_cache() -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_semantic_chunk_size(num_candidates: int) -> int:
        raw = os.environ.get(_SEMANTIC_CHUNK_ENV)
        if raw is not None:
            try:
                return max(1, min(num_candidates, int(raw)))
            except ValueError:
                pass
        return min(num_candidates, _DEFAULT_SEMANTIC_CHUNK_SIZE)

    def _process_slice(start_idx: int, end_idx: int) -> Tuple[List[torch.Tensor], List[int]]:
        chunk_size = end_idx - start_idx
        chunk_full_ids = full_ids_list[start_idx:end_idx]
        chunk_resp_lens = resp_lens[start_idx:end_idx]
        max_len = max(len(ids) for ids in chunk_full_ids)

        input_ids = torch.full((chunk_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((chunk_size, max_len), dtype=torch.long, device=device)
        spans: List[Tuple[int, int]] = []
        for local_i, ids in enumerate(chunk_full_ids):
            seq_len = len(ids)
            pad_len = max_len - seq_len
            input_ids[local_i, pad_len:] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[local_i, pad_len:] = 1
            resp_start = pad_len + obs_len
            spans.append((resp_start, resp_start + chunk_resp_lens[local_i]))

        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )
            last_hidden = outputs.hidden_states[-1]  # [chunk, seq_len, hidden_dim]
        except torch.OutOfMemoryError:
            del input_ids, attention_mask, position_ids
            _empty_cuda_cache()
            if chunk_size == 1:
                raise
            mid = start_idx + chunk_size // 2
            left_pooled, left_indices = _process_slice(start_idx, mid)
            right_pooled, right_indices = _process_slice(mid, end_idx)
            return left_pooled + right_pooled, left_indices + right_indices

        pooled_chunk: List[torch.Tensor] = []
        original_indices_chunk: List[int] = []
        for local_i, (start, end) in enumerate(spans):
            if end <= start:
                continue
            pooled_chunk.append(last_hidden[local_i, start:end, :].mean(dim=0).float().cpu())
            original_indices_chunk.append(start_idx + local_i)

        del outputs, last_hidden, input_ids, attention_mask, position_ids
        _empty_cuda_cache()
        return pooled_chunk, original_indices_chunk

    semantic_chunk_size = _get_semantic_chunk_size(N)
    for start_idx in range(0, N, semantic_chunk_size):
        end_idx = min(start_idx + semantic_chunk_size, N)
        chunk_pooled, chunk_indices = _process_slice(start_idx, end_idx)
        pooled_list.extend(chunk_pooled)
        original_indices.extend(chunk_indices)

    if len(pooled_list) == 0:
        # Degenerate: every response was empty. Return first min(k, N) raw
        # indices as best-effort — caller will feed them to env.step, which
        # typically rejects empty actions.
        return list(range(min(k, N)))

    pooled = torch.stack(pooled_list, dim=0).float()  # [M, hidden_dim], M <= N
    pooled = F.normalize(pooled, p=2, dim=-1)

    effective_k = min(k, len(pooled_list))
    result = kcenter_greedy(pooled, effective_k, initial_center_index=0)
    return [original_indices[i] for i in result.center_indices]


def _as_flat_int_list(values) -> List[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(x) for x in values]


def _as_flat_offsets(values) -> List[Tuple[int, int]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list) and values[0] and isinstance(values[0][0], (list, tuple)):
        values = values[0]
    return [(int(start), int(end)) for start, end in values]


def _token_indices_from_offsets(
    offsets: Sequence[Tuple[int, int]],
    span: Tuple[int, int],
) -> List[int]:
    span_start, span_end = span
    if span_end <= span_start:
        return []
    token_indices: List[int] = []
    for token_idx, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if token_end > span_start and token_start < span_end:
            token_indices.append(token_idx)
    return token_indices


def _token_indices_from_prefix_lengths(
    tokenizer,
    text: str,
    span: Tuple[int, int],
    *,
    num_tokens: int,
) -> List[int]:
    span_start, span_end = span
    if span_end <= span_start:
        return []
    start_idx = len(tokenizer.encode(text[:span_start], add_special_tokens=False))
    end_idx = len(tokenizer.encode(text[:span_end], add_special_tokens=False))
    start_idx = max(0, min(num_tokens, start_idx))
    end_idx = max(start_idx, min(num_tokens, end_idx))
    return list(range(start_idx, end_idx))


def _candidate_token_indices(
    tokenizer,
    text: str,
    thought_span: Tuple[int, int],
    action_span: Tuple[int, int],
) -> Tuple[List[int], List[int], List[int], bool]:
    """Tokenize a response and map Thought/Action character spans to token indices.

    Fast tokenizers use offset mappings. Slow tokenizers fall back to prefix
    token lengths, which is deterministic and sufficient for candidate-local
    response spans.
    """
    used_fallback = False
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        response_ids = _as_flat_int_list(encoded["input_ids"])
        offsets = _as_flat_offsets(encoded["offset_mapping"])
        if len(response_ids) != len(offsets):
            raise ValueError("offset mapping length mismatch")

        thought_indices = _token_indices_from_offsets(offsets, thought_span)
        action_indices = _token_indices_from_offsets(offsets, action_span)
        if action_span[1] > action_span[0] and not action_indices:
            raise ValueError("empty action span from offsets")
        return response_ids, thought_indices, action_indices, used_fallback
    except Exception:
        used_fallback = True
        response_ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        thought_indices = _token_indices_from_prefix_lengths(
            tokenizer,
            text,
            thought_span,
            num_tokens=len(response_ids),
        )
        action_indices = _token_indices_from_prefix_lengths(
            tokenizer,
            text,
            action_span,
            num_tokens=len(response_ids),
        )
        return response_ids, thought_indices, action_indices, used_fallback


def _get_output_embedding_weight(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "get_output_embeddings"):
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            return output_embeddings.weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    raise ValueError("model must expose lm_head.weight or get_output_embeddings().weight")


def _temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits
    return logits / float(temperature)


def _mean_logprob_from_logits(
    next_token_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    temperature: float,
) -> float:
    if target_ids.numel() == 0:
        return float("-inf")

    logprob_sum = 0.0
    token_count = 0
    for start in range(0, target_ids.numel(), _MULTIVIEW_TOKEN_BATCH_SIZE):
        end = min(start + _MULTIVIEW_TOKEN_BATCH_SIZE, target_ids.numel())
        logits_slice = _temperature_scale(next_token_logits[start:end].float(), temperature)
        log_probs = F.log_softmax(logits_slice, dim=-1)
        gathered = log_probs.gather(1, target_ids[start:end].unsqueeze(1)).squeeze(1)
        logprob_sum += float(gathered.sum().item())
        token_count += int(gathered.numel())
        del logits_slice, log_probs, gathered
    return logprob_sum / max(token_count, 1)


def _mean_sensitivity_feature(
    next_token_logits: torch.Tensor,
    target_ids: torch.Tensor,
    token_indices: Sequence[int],
    lm_head_weight: torch.Tensor,
    *,
    feature_topk: int,
    temperature: float,
) -> torch.Tensor:
    hidden_dim = int(lm_head_weight.shape[1])
    if not token_indices:
        return torch.zeros(hidden_dim, dtype=torch.float32)

    device = next_token_logits.device
    index_tensor = torch.tensor(token_indices, dtype=torch.long, device=device)
    selected_logits = next_token_logits.index_select(0, index_tensor)
    selected_targets = target_ids.index_select(0, index_tensor)

    feature_sum = torch.zeros(hidden_dim, dtype=torch.float32, device=device)
    token_count = 0
    topk = max(1, min(int(feature_topk), int(selected_logits.shape[-1])))
    weight = lm_head_weight.to(device)

    for start in range(0, selected_targets.numel(), _MULTIVIEW_TOKEN_BATCH_SIZE):
        end = min(start + _MULTIVIEW_TOKEN_BATCH_SIZE, selected_targets.numel())
        logits_slice = _temperature_scale(selected_logits[start:end].float(), temperature)
        topk_logits, topk_ids = torch.topk(logits_slice, k=topk, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1).float()
        topk_weight = weight.index_select(0, topk_ids.reshape(-1)).reshape(
            topk_ids.shape[0],
            topk_ids.shape[1],
            hidden_dim,
        ).float()
        expected_weight = torch.bmm(topk_probs.unsqueeze(1), topk_weight).squeeze(1)
        target_weight = weight.index_select(0, selected_targets[start:end]).float()
        feature_sum += (target_weight - expected_weight).sum(dim=0)
        token_count += int(end - start)
        del logits_slice, topk_logits, topk_ids, topk_probs, topk_weight
        del expected_weight, target_weight

    del index_tensor, selected_logits, selected_targets
    return (feature_sum / max(token_count, 1)).detach().cpu()


def _thought_residual(thought_feature: torch.Tensor, action_feature: torch.Tensor) -> torch.Tensor:
    action_norm_sq = torch.dot(action_feature, action_feature)
    if float(action_norm_sq.item()) <= _MULTIVIEW_EPS:
        residual = thought_feature.clone()
    else:
        projection_scale = torch.dot(thought_feature, action_feature) / action_norm_sq
        residual = thought_feature - projection_scale * action_feature
    if float(torch.linalg.vector_norm(residual).item()) <= _MULTIVIEW_EPS:
        return torch.zeros_like(residual)
    return residual


def _cosine_novelty(vector: torch.Tensor, selected_vectors: Sequence[torch.Tensor]) -> float:
    if float(torch.linalg.vector_norm(vector).item()) <= _MULTIVIEW_EPS:
        return 0.0

    nonzero_selected = [
        selected
        for selected in selected_vectors
        if float(torch.linalg.vector_norm(selected).item()) > _MULTIVIEW_EPS
    ]
    if not nonzero_selected:
        return 1.0

    selected_matrix = torch.stack(nonzero_selected, dim=0).float()
    vector = vector.float()
    similarities = F.cosine_similarity(selected_matrix, vector.unsqueeze(0), dim=-1)
    novelty = 1.0 - float(torch.max(similarities).item())
    return max(0.0, novelty)


def _quality_ranks(mean_logprobs: Sequence[float]) -> List[float]:
    num_items = len(mean_logprobs)
    if num_items == 0:
        return []
    if num_items == 1:
        return [1.0]

    sorted_indices = sorted(range(num_items), key=lambda i: (-mean_logprobs[i], i))
    ranks = [0.0] * num_items
    denom = float(num_items - 1)
    for rank, feature_idx in enumerate(sorted_indices):
        ranks[feature_idx] = (num_items - 1 - rank) / denom
    return ranks


def _horizon_ratio(round_idx: int, max_rounds: int) -> float:
    return max(0.0, (int(max_rounds) - int(round_idx) - 1) / max(int(max_rounds) - 1, 1))


def _select_centers_multiview_from_features(
    features: Sequence[_MultiviewCandidateFeature],
    k: int,
    *,
    round_idx: int,
    max_rounds: int,
) -> List[int]:
    if not features:
        return []

    effective_k = min(int(k), len(features))
    if effective_k <= 0:
        return []

    mean_logprobs = [feature.mean_logprob for feature in features]
    quality_rank = _quality_ranks(mean_logprobs)
    selected_feature_indices: List[int] = []
    selected_actions = set()
    selected_action_features: List[torch.Tensor] = []
    selected_thought_residuals: List[torch.Tensor] = []

    first_idx = max(range(len(features)), key=lambda i: (mean_logprobs[i], -features[i].raw_index))
    horizon = _horizon_ratio(round_idx, max_rounds)

    def _add(feature_idx: int) -> None:
        feature = features[feature_idx]
        selected_feature_indices.append(feature_idx)
        selected_actions.add(feature.normalized_action)
        selected_action_features.append(feature.action_feature)
        selected_thought_residuals.append(feature.thought_residual)

    _add(first_idx)

    while len(selected_feature_indices) < effective_k:
        best_idx: Optional[int] = None
        best_key: Optional[Tuple[float, float, float, int]] = None
        selected_set = set(selected_feature_indices)
        for feature_idx, feature in enumerate(features):
            if feature_idx in selected_set:
                continue
            if feature.normalized_action in selected_actions:
                action_gain = 0.0
            else:
                action_gain = _cosine_novelty(feature.action_feature, selected_action_features)
            thought_gain = _cosine_novelty(feature.thought_residual, selected_thought_residuals)
            score = action_gain + horizon * quality_rank[feature_idx] * thought_gain
            tie_key = (
                score,
                quality_rank[feature_idx],
                feature.mean_logprob,
                -feature.raw_index,
            )
            if best_key is None or tie_key > best_key:
                best_key = tie_key
                best_idx = feature_idx

        if best_idx is None:
            break
        _add(best_idx)

    return [features[feature_idx].raw_index for feature_idx in selected_feature_indices]


def _empty_cuda_cache_for(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _model_forward_no_grad(model: torch.nn.Module, **kwargs):
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("position_ids", None)
        return model(**kwargs)


def _compute_multiview_features(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    parsed_candidates: Sequence[Tuple[TextCraftCandidate, Tuple[int, int], Tuple[int, int]]],
    *,
    feature_topk: int,
    feature_chunk_size: int,
    temperature: float,
    stats: Optional[MutableMapping[str, int]] = None,
) -> List[_MultiviewCandidateFeature]:
    device = next(model.parameters()).device
    obs_ids = [int(x) for x in obs_token_ids]
    if len(obs_ids) < 1:
        raise ValueError("obs_token_ids must be non-empty for causal token scoring")

    lm_head_weight = _get_output_embedding_weight(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    chunk_size = max(1, int(feature_chunk_size))
    feature_topk = max(1, int(feature_topk))

    token_infos = []
    fallback_count = 0
    for candidate, thought_span, action_span in parsed_candidates:
        response_ids, thought_indices, action_indices, used_fallback = _candidate_token_indices(
            tokenizer,
            candidate.raw_text,
            thought_span,
            action_span,
        )
        fallback_count += int(used_fallback)
        if not response_ids or not action_indices:
            continue
        token_infos.append(
            (candidate, thought_span, action_span, response_ids, thought_indices, action_indices)
        )

    if stats is not None:
        stats["offset_fallback_count"] = stats.get("offset_fallback_count", 0) + fallback_count
        stats["multiview_feature_candidates"] = stats.get("multiview_feature_candidates", 0) + len(token_infos)

    features: List[_MultiviewCandidateFeature] = []
    model_was_training = bool(model.training)
    model.eval()

    try:
        for start_idx in range(0, len(token_infos), chunk_size):
            end_idx = min(start_idx + chunk_size, len(token_infos))
            chunk_infos = token_infos[start_idx:end_idx]
            full_ids_list = [
                obs_ids + response_ids
                for _, _, _, response_ids, _, _ in chunk_infos
            ]
            max_len = max(len(ids) for ids in full_ids_list)

            input_ids = torch.full(
                (len(chunk_infos), max_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(chunk_infos), max_len),
                dtype=torch.long,
                device=device,
            )
            pad_lengths: List[int] = []
            for local_i, ids in enumerate(full_ids_list):
                seq_len = len(ids)
                pad_len = max_len - seq_len
                pad_lengths.append(pad_len)
                input_ids[local_i, pad_len:] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[local_i, pad_len:] = 1

            position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

            try:
                with torch.no_grad():
                    outputs = _model_forward_no_grad(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    )
                logits = outputs.logits
            except torch.OutOfMemoryError:
                del input_ids, attention_mask, position_ids
                _empty_cuda_cache_for(device)
                if len(chunk_infos) == 1:
                    raise
                local_mid = max(1, len(chunk_infos) // 2)
                left_candidates = [
                    (candidate, thought_span, action_span)
                    for candidate, thought_span, action_span, _, _, _ in chunk_infos[:local_mid]
                ]
                right_candidates = [
                    (candidate, thought_span, action_span)
                    for candidate, thought_span, action_span, _, _, _ in chunk_infos[local_mid:]
                ]
                left = _compute_multiview_features(
                    model,
                    tokenizer,
                    obs_token_ids,
                    left_candidates,
                    feature_topk=feature_topk,
                    feature_chunk_size=1,
                    temperature=temperature,
                    stats=None,
                )
                right = _compute_multiview_features(
                    model,
                    tokenizer,
                    obs_token_ids,
                    right_candidates,
                    feature_topk=feature_topk,
                    feature_chunk_size=1,
                    temperature=temperature,
                    stats=None,
                )
                features.extend(left)
                features.extend(right)
                continue

            for local_i, (
                candidate,
                _thought_span,
                _action_span,
                response_ids,
                thought_indices,
                action_indices,
            ) in enumerate(chunk_infos):
                response_len = len(response_ids)
                target_ids = torch.tensor(response_ids, dtype=torch.long, device=device)
                prediction_start = pad_lengths[local_i] + len(obs_ids) - 1
                prediction_positions = torch.arange(
                    prediction_start,
                    prediction_start + response_len,
                    dtype=torch.long,
                    device=device,
                )
                next_token_logits = logits[local_i].index_select(0, prediction_positions)
                mean_logprob = _mean_logprob_from_logits(
                    next_token_logits,
                    target_ids,
                    temperature=temperature,
                )
                action_feature = _mean_sensitivity_feature(
                    next_token_logits,
                    target_ids,
                    action_indices,
                    lm_head_weight,
                    feature_topk=feature_topk,
                    temperature=temperature,
                )
                thought_feature = _mean_sensitivity_feature(
                    next_token_logits,
                    target_ids,
                    thought_indices,
                    lm_head_weight,
                    feature_topk=feature_topk,
                    temperature=temperature,
                )
                thought_residual = _thought_residual(thought_feature, action_feature)
                features.append(
                    _MultiviewCandidateFeature(
                        raw_index=candidate.raw_index,
                        normalized_action=candidate.normalized_action,
                        action_feature=action_feature,
                        thought_feature=thought_feature,
                        thought_residual=thought_residual,
                        mean_logprob=mean_logprob,
                    )
                )
                del target_ids, prediction_positions, next_token_logits

            del outputs, logits, input_ids, attention_mask, position_ids
            _empty_cuda_cache_for(device)
    finally:
        if model_was_training:
            model.train()

    return features


def _compute_candidate_mean_logprobs(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    parsed_candidates: Sequence[Tuple[TextCraftCandidate, Tuple[int, int], Tuple[int, int]]],
    *,
    feature_chunk_size: int,
    temperature: float,
    stats: Optional[MutableMapping[str, int]] = None,
) -> List[Tuple[TextCraftCandidate, float]]:
    """Score candidate responses by mean token logprob under the current actor."""
    device = next(model.parameters()).device
    obs_ids = [int(x) for x in obs_token_ids]
    if len(obs_ids) < 1:
        raise ValueError("obs_token_ids must be non-empty for causal token scoring")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    chunk_size = max(1, int(feature_chunk_size))

    token_infos = []
    fallback_count = 0
    for candidate, thought_span, action_span in parsed_candidates:
        response_ids, _, _, used_fallback = _candidate_token_indices(
            tokenizer,
            candidate.raw_text,
            thought_span,
            action_span,
        )
        fallback_count += int(used_fallback)
        if not response_ids:
            continue
        token_infos.append((candidate, thought_span, action_span, response_ids))

    if stats is not None:
        stats["offset_fallback_count"] = stats.get("offset_fallback_count", 0) + fallback_count
        stats["quality_unique_scored_candidates"] = (
            stats.get("quality_unique_scored_candidates", 0) + len(token_infos)
        )

    scored: List[Tuple[TextCraftCandidate, float]] = []
    model_was_training = bool(model.training)
    model.eval()

    try:
        for start_idx in range(0, len(token_infos), chunk_size):
            end_idx = min(start_idx + chunk_size, len(token_infos))
            chunk_infos = token_infos[start_idx:end_idx]
            full_ids_list = [
                obs_ids + response_ids
                for _, _, _, response_ids in chunk_infos
            ]
            max_len = max(len(ids) for ids in full_ids_list)

            input_ids = torch.full(
                (len(chunk_infos), max_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(chunk_infos), max_len),
                dtype=torch.long,
                device=device,
            )
            pad_lengths: List[int] = []
            for local_i, ids in enumerate(full_ids_list):
                seq_len = len(ids)
                pad_len = max_len - seq_len
                pad_lengths.append(pad_len)
                input_ids[local_i, pad_len:] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[local_i, pad_len:] = 1

            position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

            try:
                with torch.no_grad():
                    outputs = _model_forward_no_grad(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    )
                logits = outputs.logits
            except torch.OutOfMemoryError:
                del input_ids, attention_mask, position_ids
                _empty_cuda_cache_for(device)
                if len(chunk_infos) == 1:
                    raise
                local_mid = max(1, len(chunk_infos) // 2)
                left_candidates = [
                    (candidate, thought_span, action_span)
                    for candidate, thought_span, action_span, _ in chunk_infos[:local_mid]
                ]
                right_candidates = [
                    (candidate, thought_span, action_span)
                    for candidate, thought_span, action_span, _ in chunk_infos[local_mid:]
                ]
                left = _compute_candidate_mean_logprobs(
                    model,
                    tokenizer,
                    obs_token_ids,
                    left_candidates,
                    feature_chunk_size=1,
                    temperature=temperature,
                    stats=None,
                )
                right = _compute_candidate_mean_logprobs(
                    model,
                    tokenizer,
                    obs_token_ids,
                    right_candidates,
                    feature_chunk_size=1,
                    temperature=temperature,
                    stats=None,
                )
                scored.extend(left)
                scored.extend(right)
                continue

            for local_i, (candidate, _thought_span, _action_span, response_ids) in enumerate(chunk_infos):
                response_len = len(response_ids)
                target_ids = torch.tensor(response_ids, dtype=torch.long, device=device)
                prediction_start = pad_lengths[local_i] + len(obs_ids) - 1
                prediction_positions = torch.arange(
                    prediction_start,
                    prediction_start + response_len,
                    dtype=torch.long,
                    device=device,
                )
                next_token_logits = logits[local_i].index_select(0, prediction_positions)
                mean_logprob = _mean_logprob_from_logits(
                    next_token_logits,
                    target_ids,
                    temperature=temperature,
                )
                scored.append((candidate, mean_logprob))
                del target_ids, prediction_positions, next_token_logits

            del outputs, logits, input_ids, attention_mask, position_ids
            _empty_cuda_cache_for(device)
    finally:
        if model_was_training:
            model.train()

    return scored


def _compute_g2rl_text_features(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    candidate_texts: Sequence[Tuple[TextCraftCandidate, str]],
    *,
    feature_topk: int,
    feature_chunk_size: int,
    temperature: float,
    stats: Optional[MutableMapping[str, int]] = None,
    stats_prefix: str = "g2rl",
) -> List[_G2RLCandidateFeature]:
    """Compute the G2RL token-gradient feature for compact candidate texts.

    This mirrors ``DataParallelPPOActor._accumulate_g2rl_features``: for each
    token in the selected feature text, use ``realized lm_head weight -
    expected lm_head weight`` and average across tokens.
    """
    device = next(model.parameters()).device
    obs_ids = [int(x) for x in obs_token_ids]
    if len(obs_ids) < 1:
        raise ValueError("obs_token_ids must be non-empty for causal token scoring")

    lm_head_weight = _get_output_embedding_weight(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    chunk_size = max(1, int(feature_chunk_size))
    feature_topk = max(1, int(feature_topk))

    token_infos = []
    for candidate, feature_text in candidate_texts:
        text = str(feature_text or "").strip() or "no action"
        response_ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if not response_ids:
            response_ids = [pad_id]
        token_infos.append((candidate, response_ids))

    if stats is not None:
        key = f"{stats_prefix}_feature_candidates"
        stats[key] = stats.get(key, 0) + len(token_infos)

    features: List[_G2RLCandidateFeature] = []
    model_was_training = bool(model.training)
    model.eval()

    try:
        for start_idx in range(0, len(token_infos), chunk_size):
            end_idx = min(start_idx + chunk_size, len(token_infos))
            chunk_infos = token_infos[start_idx:end_idx]
            full_ids_list = [obs_ids + response_ids for _, response_ids in chunk_infos]
            max_len = max(len(ids) for ids in full_ids_list)

            input_ids = torch.full(
                (len(chunk_infos), max_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(chunk_infos), max_len),
                dtype=torch.long,
                device=device,
            )
            pad_lengths: List[int] = []
            for local_i, ids in enumerate(full_ids_list):
                seq_len = len(ids)
                pad_len = max_len - seq_len
                pad_lengths.append(pad_len)
                input_ids[local_i, pad_len:] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[local_i, pad_len:] = 1

            position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

            try:
                with torch.no_grad():
                    outputs = _model_forward_no_grad(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                    )
                logits = outputs.logits
            except torch.OutOfMemoryError:
                del input_ids, attention_mask, position_ids
                _empty_cuda_cache_for(device)
                if len(chunk_infos) == 1:
                    raise
                local_mid = max(1, len(chunk_infos) // 2)
                left = _compute_g2rl_text_features(
                    model,
                    tokenizer,
                    obs_token_ids,
                    [
                        (candidate, tokenizer.decode(response_ids, skip_special_tokens=True))
                        for candidate, response_ids in chunk_infos[:local_mid]
                    ],
                    feature_topk=feature_topk,
                    feature_chunk_size=1,
                    temperature=temperature,
                    stats=None,
                    stats_prefix=stats_prefix,
                )
                right = _compute_g2rl_text_features(
                    model,
                    tokenizer,
                    obs_token_ids,
                    [
                        (candidate, tokenizer.decode(response_ids, skip_special_tokens=True))
                        for candidate, response_ids in chunk_infos[local_mid:]
                    ],
                    feature_topk=feature_topk,
                    feature_chunk_size=1,
                    temperature=temperature,
                    stats=None,
                    stats_prefix=stats_prefix,
                )
                features.extend(left)
                features.extend(right)
                continue

            for local_i, (candidate, response_ids) in enumerate(chunk_infos):
                response_len = len(response_ids)
                target_ids = torch.tensor(response_ids, dtype=torch.long, device=device)
                prediction_start = pad_lengths[local_i] + len(obs_ids) - 1
                prediction_positions = torch.arange(
                    prediction_start,
                    prediction_start + response_len,
                    dtype=torch.long,
                    device=device,
                )
                next_token_logits = logits[local_i].index_select(0, prediction_positions)
                feature = _mean_sensitivity_feature(
                    next_token_logits,
                    target_ids,
                    list(range(response_len)),
                    lm_head_weight,
                    feature_topk=feature_topk,
                    temperature=temperature,
                )
                features.append(
                    _G2RLCandidateFeature(
                        raw_index=candidate.raw_index,
                        normalized_action=candidate.normalized_action,
                        feature=feature,
                    )
                )
                del target_ids, prediction_positions, next_token_logits

            del outputs, logits, input_ids, attention_mask, position_ids
            _empty_cuda_cache_for(device)
    finally:
        if model_was_training:
            model.train()

    return features


def _select_g2rl_feature_kcenters(features: Sequence[_G2RLCandidateFeature], k: int) -> List[int]:
    if not features:
        return []
    feature_matrix = torch.stack([feature.feature for feature in features], dim=0).float()
    normalized = F.normalize(feature_matrix, p=2, dim=-1, eps=1e-12)
    effective_k = min(int(k), int(normalized.shape[0]))
    if effective_k <= 0:
        return []
    result = kcenter_greedy(normalized, effective_k, initial_center_index=0)
    return [features[feature_idx].raw_index for feature_idx in result.center_indices]


def _select_unique_actions_by_quality(
    scored_candidates: Sequence[Tuple[TextCraftCandidate, float]],
    k: int,
) -> List[int]:
    effective_k = min(int(k), len(scored_candidates))
    if effective_k <= 0:
        return []

    def sort_key(local_idx: int) -> Tuple[float, int]:
        candidate, score = scored_candidates[local_idx]
        quality = float(score)
        if not math.isfinite(quality):
            quality = float("-inf")
        return (-quality, candidate.raw_index)

    ordered = sorted(range(len(scored_candidates)), key=sort_key)
    selected_local: List[int] = []
    selected_set = set()
    seen_actions = set()

    for local_idx in ordered:
        candidate, _score = scored_candidates[local_idx]
        if candidate.normalized_action in seen_actions:
            continue
        selected_local.append(local_idx)
        selected_set.add(local_idx)
        seen_actions.add(candidate.normalized_action)
        if len(selected_local) >= effective_k:
            return [scored_candidates[i][0].raw_index for i in selected_local]

    for local_idx in ordered:
        if local_idx in selected_set:
            continue
        selected_local.append(local_idx)
        if len(selected_local) >= effective_k:
            break

    return [scored_candidates[i][0].raw_index for i in selected_local]


def select_centers_quality_unique_action(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
    *,
    temperature: float = 1.0,
    feature_chunk_size: int = _DEFAULT_MULTIVIEW_CHUNK_SIZE,
    stats: Optional[MutableMapping[str, int]] = None,
) -> List[int]:
    """Select high-confidence candidates while forcing unique normalized actions first.

    This is a quality-constrained diversity baseline for TextCraft: it keeps the
    round0 expansion budget but avoids low-logprob tail candidates unless they
    are needed to fill the requested number of centers.
    """
    parsed_candidates: List[Tuple[TextCraftCandidate, Tuple[int, int], Tuple[int, int]]] = []
    invalid_count = 0
    for raw_index, text in enumerate(response_texts):
        parsed = _parse_textcraft_candidate_with_spans(raw_index, text)
        if parsed is None:
            invalid_count += 1
            continue
        parsed_candidates.append(parsed)

    if stats is not None:
        stats["quality_unique_invalid_candidates"] = (
            stats.get("quality_unique_invalid_candidates", 0) + invalid_count
        )

    if not parsed_candidates:
        return list(range(min(k, len(response_texts))))

    scored_candidates = _compute_candidate_mean_logprobs(
        model,
        tokenizer,
        obs_token_ids,
        parsed_candidates,
        feature_chunk_size=feature_chunk_size,
        temperature=temperature,
        stats=stats,
    )
    if not scored_candidates:
        return [
            candidate.raw_index
            for candidate, _, _ in parsed_candidates[: min(k, len(parsed_candidates))]
        ]

    selected = _select_unique_actions_by_quality(scored_candidates, k)
    if stats is not None:
        unique_actions = {candidate.normalized_action for candidate, _ in scored_candidates}
        selected_action_counts = {}
        for raw_index in selected:
            candidate = next(
                (
                    scored_candidate
                    for scored_candidate, _score in scored_candidates
                    if scored_candidate.raw_index == raw_index
                ),
                None,
            )
            if candidate is None:
                continue
            selected_action_counts[candidate.normalized_action] = (
                selected_action_counts.get(candidate.normalized_action, 0) + 1
            )
        duplicate_fill_count = sum(
            max(0, count - 1)
            for count in selected_action_counts.values()
        )
        stats["quality_unique_action_count"] = (
            stats.get("quality_unique_action_count", 0) + len(unique_actions)
        )
        stats["quality_unique_duplicate_fill_count"] = (
            stats.get("quality_unique_duplicate_fill_count", 0) + duplicate_fill_count
        )
    return selected


def select_centers_g2rl_action_gradient(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
    *,
    temperature: float = 1.0,
    feature_topk: int = _DEFAULT_MULTIVIEW_TOPK,
    feature_chunk_size: int = _DEFAULT_MULTIVIEW_CHUNK_SIZE,
    stats: Optional[MutableMapping[str, int]] = None,
) -> List[int]:
    """Select candidates by k-center over the same action-gradient feature used by G2RL.

    The per-candidate embedding is the token-averaged ``realized lm_head weight
    - expected lm_head weight`` feature for the parsed Action span. This matches
    the runtime G2RL feature formula for action-scope tokens; the selector uses
    it only as a rollout-time clustering criterion and does not change rewards.
    """
    parsed_candidates: List[Tuple[TextCraftCandidate, Tuple[int, int], Tuple[int, int]]] = []
    invalid_count = 0
    for raw_index, text in enumerate(response_texts):
        parsed = _parse_textcraft_candidate_with_spans(raw_index, text)
        if parsed is None:
            invalid_count += 1
            continue
        parsed_candidates.append(parsed)

    if stats is not None:
        stats["g2rl_action_invalid_candidates"] = (
            stats.get("g2rl_action_invalid_candidates", 0) + invalid_count
        )

    if not parsed_candidates:
        return list(range(min(k, len(response_texts))))

    features = _compute_multiview_features(
        model,
        tokenizer,
        obs_token_ids,
        parsed_candidates,
        feature_topk=feature_topk,
        feature_chunk_size=feature_chunk_size,
        temperature=temperature,
        stats=stats,
    )
    if not features:
        return [
            candidate.raw_index
            for candidate, _, _ in parsed_candidates[: min(k, len(parsed_candidates))]
        ]

    if stats is not None:
        stats["g2rl_action_feature_candidates"] = (
            stats.get("g2rl_action_feature_candidates", 0) + len(features)
        )

    feature_matrix = torch.stack([feature.action_feature for feature in features], dim=0).float()
    normalized = F.normalize(feature_matrix, p=2, dim=-1, eps=1e-12)
    effective_k = min(int(k), int(normalized.shape[0]))
    if effective_k <= 0:
        return []

    result = kcenter_greedy(normalized, effective_k, initial_center_index=0)
    return [features[feature_idx].raw_index for feature_idx in result.center_indices]


def select_centers_g2rl_normalized_action_gradient(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
    *,
    temperature: float = 1.0,
    feature_topk: int = _DEFAULT_MULTIVIEW_TOPK,
    feature_chunk_size: int = _DEFAULT_MULTIVIEW_CHUNK_SIZE,
    stats: Optional[MutableMapping[str, int]] = None,
    action_normalizer: Optional[str] = None,
) -> List[int]:
    """Select candidates by k-center over G2RL normalized-action features.

    This matches the main TextCraft G2RL feature scope used in the training
    experiments: tokenize the parser-normalized action text as the compact
    response, compute the G2RL token-gradient feature, and cluster those
    features at rollout time without enabling reward shaping.
    """
    parsed_candidates: List[TextCraftCandidate] = []
    invalid_count = 0
    for raw_index, text in enumerate(response_texts):
        parsed = _parse_action_candidate_with_spans(
            raw_index,
            text,
            action_normalizer=action_normalizer,
        )
        if parsed is None:
            invalid_count += 1
            continue
        candidate, _, _ = parsed
        parsed_candidates.append(candidate)

    if stats is not None:
        stats["g2rl_normalized_action_invalid_candidates"] = (
            stats.get("g2rl_normalized_action_invalid_candidates", 0) + invalid_count
        )

    if not parsed_candidates:
        return list(range(min(k, len(response_texts))))

    features = _compute_g2rl_text_features(
        model,
        tokenizer,
        obs_token_ids,
        [(candidate, candidate.normalized_action) for candidate in parsed_candidates],
        feature_topk=feature_topk,
        feature_chunk_size=feature_chunk_size,
        temperature=temperature,
        stats=stats,
        stats_prefix="g2rl_normalized_action",
    )
    if not features:
        return [candidate.raw_index for candidate in parsed_candidates[: min(k, len(parsed_candidates))]]

    return _select_g2rl_feature_kcenters(features, k)


def select_centers_gradient_multiview(
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
    *,
    round_idx: int = 0,
    max_rounds: int = 1,
    temperature: float = 1.0,
    feature_topk: int = _DEFAULT_MULTIVIEW_TOPK,
    feature_chunk_size: int = _DEFAULT_MULTIVIEW_CHUNK_SIZE,
    stats: Optional[MutableMapping[str, int]] = None,
) -> List[int]:
    """Select TextCraft candidates with forward-only Action/Thought G2RL features."""
    parsed_candidates: List[Tuple[TextCraftCandidate, Tuple[int, int], Tuple[int, int]]] = []
    invalid_count = 0
    for raw_index, text in enumerate(response_texts):
        parsed = _parse_textcraft_candidate_with_spans(raw_index, text)
        if parsed is None:
            invalid_count += 1
            continue
        parsed_candidates.append(parsed)

    if stats is not None:
        stats["multiview_invalid_candidates"] = stats.get("multiview_invalid_candidates", 0) + invalid_count

    if not parsed_candidates:
        return list(range(min(k, len(response_texts))))

    features = _compute_multiview_features(
        model,
        tokenizer,
        obs_token_ids,
        parsed_candidates,
        feature_topk=feature_topk,
        feature_chunk_size=feature_chunk_size,
        temperature=temperature,
        stats=stats,
    )
    if not features:
        return [candidate.raw_index for candidate, _, _ in parsed_candidates[: min(k, len(parsed_candidates))]]

    return _select_centers_multiview_from_features(
        features,
        k,
        round_idx=round_idx,
        max_rounds=max_rounds,
    )


def select_centers(
    method: str,
    model: torch.nn.Module,
    tokenizer,
    obs_token_ids: List[int],
    response_texts: List[str],
    k: int,
    d_proj: int = 512,
    round_idx: int = 0,
    max_rounds: int = 1,
    temperature: float = 1.0,
    feature_topk: int = _DEFAULT_MULTIVIEW_TOPK,
    feature_chunk_size: int = _DEFAULT_MULTIVIEW_CHUNK_SIZE,
    stats: Optional[MutableMapping[str, int]] = None,
    action_normalizer: Optional[str] = None,
) -> List[int]:
    """Dispatcher: select cluster centers by *method*.

    Parameters
    ----------
    method:
        ``"gradient"``, ``"gradient_multiview"``, ``"g2rl_action_gradient"``,
        ``"g2rl_normalized_action_gradient"``, ``"quality_unique_action"``, or
        ``"semantic"``.
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
    if method == "gradient_multiview":
        return select_centers_gradient_multiview(
            model,
            tokenizer,
            obs_token_ids,
            response_texts,
            k,
            round_idx=round_idx,
            max_rounds=max_rounds,
            temperature=temperature,
            feature_topk=feature_topk,
            feature_chunk_size=feature_chunk_size,
            stats=stats,
        )
    if method == "g2rl_action_gradient":
        return select_centers_g2rl_action_gradient(
            model,
            tokenizer,
            obs_token_ids,
            response_texts,
            k,
            temperature=temperature,
            feature_topk=feature_topk,
            feature_chunk_size=feature_chunk_size,
            stats=stats,
        )
    if method == "g2rl_normalized_action_gradient":
        return select_centers_g2rl_normalized_action_gradient(
            model,
            tokenizer,
            obs_token_ids,
            response_texts,
            k,
            temperature=temperature,
            feature_topk=feature_topk,
            feature_chunk_size=feature_chunk_size,
            stats=stats,
            action_normalizer=action_normalizer,
        )
    if method == "quality_unique_action":
        return select_centers_quality_unique_action(
            model,
            tokenizer,
            obs_token_ids,
            response_texts,
            k,
            temperature=temperature,
            feature_chunk_size=feature_chunk_size,
            stats=stats,
        )
    raise ValueError(f"Unknown clustering method: {method!r}")
