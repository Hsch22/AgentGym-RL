import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import torch


_CLUSTERING_PATH = (
    Path(__file__).resolve().parents[1]
    / "verl"
    / "workers"
    / "rollout"
    / "agent_vllm_rollout"
    / "clustering.py"
)
_SPEC = importlib.util.spec_from_file_location("agent_vllm_clustering", _CLUSTERING_PATH)
clustering = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = clustering
_SPEC.loader.exec_module(clustering)


def _feature(raw_index, action, action_feature, thought_residual, mean_logprob):
    action_tensor = torch.tensor(action_feature, dtype=torch.float32)
    thought_tensor = torch.tensor(thought_residual, dtype=torch.float32)
    return clustering._MultiviewCandidateFeature(
        raw_index=raw_index,
        normalized_action=action,
        action_feature=action_tensor,
        thought_feature=thought_tensor,
        thought_residual=thought_tensor,
        mean_logprob=mean_logprob,
    )


def test_same_action_can_win_with_distinct_residual_thought():
    features = [
        _feature(0, "get 1 oak log", [1.0, 0.0], [1.0, 0.0], -0.1),
        _feature(1, "get 1 oak log", [1.0, 0.0], [0.0, 1.0], -0.2),
        _feature(2, "craft 1 stick", [1.0, 0.0], [1.0, 0.0], -0.3),
    ]

    selected = clustering._select_centers_multiview_from_features(
        features,
        k=2,
        round_idx=0,
        max_rounds=3,
    )

    assert selected == [0, 1]


def test_low_quality_thought_noise_does_not_beat_better_candidate():
    features = [
        _feature(0, "get 1 oak log", [1.0, 0.0], [1.0, 0.0], -0.1),
        _feature(1, "craft 1 stick", [0.98, 0.2], [1.0, 0.0], -0.2),
        _feature(2, "craft 1 plank", [1.0, 0.0], [0.0, 1.0], -10.0),
    ]

    selected = clustering._select_centers_multiview_from_features(
        features,
        k=2,
        round_idx=0,
        max_rounds=3,
    )

    assert selected == [0, 1]


def test_last_round_ignores_thought_gain():
    features = [
        _feature(0, "get 1 oak log", [1.0, 0.0], [1.0, 0.0], -0.1),
        _feature(1, "get 1 oak log", [1.0, 0.0], [0.0, 1.0], -0.2),
        _feature(2, "craft 1 stick", [0.0, 1.0], [0.0, 0.0], -10.0),
    ]

    selected = clustering._select_centers_multiview_from_features(
        features,
        k=2,
        round_idx=2,
        max_rounds=3,
    )

    assert selected == [0, 2]


def test_all_invalid_candidates_fall_back_to_raw_indices():
    selected = clustering.select_centers_gradient_multiview(
        model=None,
        tokenizer=None,
        obs_token_ids=[],
        response_texts=["Thought only", "Action: !!!"],
        k=2,
    )

    assert selected == [0, 1]


def test_sciworld_normalized_action_canonicalizes_common_aliases():
    assert clustering.parse_valid_action(
        "Thought:\ninspect\n\nAction:\nPick-up \"red apple\".",
        action_normalizer="sciworld",
    ) == "pick up red apple"
    assert clustering.parse_valid_action(
        "Thought:\ncheck room\n\nAction:\nlookaround",
        action_normalizer="sciworld",
    ) == "look around"
    assert clustering.parse_valid_action(
        "Thought:\nfree hand\n\nAction:\nput down beaker",
        action_normalizer="sciworld",
    ) == "drop beaker"
    assert clustering.parse_valid_action(
        "Thought:\nwrong\n\nAction:\nlook around\nThought:\ntry inventory\n\nAction:\ninventory",
        action_normalizer="sciworld",
    ) == "inventory"
    assert clustering.parse_valid_action(
        "Thought:\nwrong\n\nAction:\nlook around\nThought:\ntry inventory\n\nAction:\ninventory",
    ) is None


def test_tokenizer_without_offsets_uses_stable_prefix_fallback():
    class PrefixTokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("offset mapping unavailable")

    text = "Thought:\ncollect wood first\n\nAction:\nget 1 oak log\n"
    parsed = clustering._parse_textcraft_candidate_with_spans(0, text)
    assert parsed is not None
    _candidate, thought_span, action_span = parsed

    response_ids, thought_indices, action_indices, used_fallback = clustering._candidate_token_indices(
        PrefixTokenizer(),
        text,
        thought_span,
        action_span,
    )

    assert used_fallback is True
    assert len(response_ids) == len(text)
    assert thought_indices == list(range(thought_span[0], thought_span[1]))
    assert action_indices == list(range(action_span[0], action_span[1]))


def test_forward_multiview_selector_records_stats_without_backward():
    class CharTokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [(ord(ch) % 63) + 1 for ch in text]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            encoded = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
            if return_offsets_mapping:
                encoded["offset_mapping"] = [(idx, idx + 1) for idx in range(len(text))]
            return encoded

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(65, 8)
            self.lm_head = torch.nn.Linear(8, 65, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, attention_mask=None, position_ids=None, return_dict=True):
            hidden = self.embed(input_ids)
            return SimpleNamespace(logits=self.lm_head(hidden))

    model = TinyLM()
    stats = {}
    selected = clustering.select_centers_gradient_multiview(
        model=model,
        tokenizer=CharTokenizer(),
        obs_token_ids=[1, 2, 3],
        response_texts=[
            "Thought:\ncollect logs\n\nAction:\nget 1 oak log",
            "Thought:\nmake planks next\n\nAction:\nget 1 oak log",
            "Thought:\ncraft output\n\nAction:\ncraft 1 stick",
        ],
        k=2,
        round_idx=0,
        max_rounds=3,
        feature_topk=8,
        feature_chunk_size=2,
        stats=stats,
    )

    assert len(selected) == 2
    assert stats["offset_fallback_count"] == 0
    assert stats["multiview_feature_candidates"] == 3
    assert all(parameter.grad is None for parameter in model.parameters())
