import torch

from verl.workers.agent_fsdp_workers import (
    _build_textcraft_action_token_selection_fast,
    build_textcraft_action_feature_mask,
    build_textcraft_action_feature_mask_fast,
    build_textcraft_action_feature_mask_slow,
    build_textcraft_normalized_action_feature_inputs,
    find_textcraft_action_spans,
)


class CharTokenizer:

    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(token_id) for token_id in ids if token_id != self.pad_token_id)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        output = {'input_ids': self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            output['offset_mapping'] = [(idx, idx + 1) for idx in range(len(text))]
        return output


def _batch_texts(tokenizer, texts):
    ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    max_len = max(len(row) for row in ids)
    responses = torch.full((len(ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros_like(responses, dtype=torch.bool)
    for row_idx, row in enumerate(ids):
        responses[row_idx, :len(row)] = torch.tensor(row)
        response_mask[row_idx, :len(row)] = True
    return responses, response_mask


def _selected_texts(tokenizer, responses, feature_mask):
    return [
        tokenizer.decode(responses[row_idx, feature_mask[row_idx]].tolist())
        for row_idx in range(responses.size(0))
    ]


def _fast_selection_for_text(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return _build_textcraft_action_token_selection_fast(token_ids, tokenizer)


def test_textcraft_action_spans_exclude_thought_and_observation():
    text = 'Thought: inspect\nAction: inventory\nObservation: ok\nThought: craft\nAction: craft plank'

    spans = [text[start:end] for start, end in find_textcraft_action_spans(text)]

    assert spans == ['inventory', 'craft plank']


def test_action_feature_mask_keeps_multiple_actions_and_falls_back_without_marker():
    tokenizer = CharTokenizer()
    texts = [
        'Thought: inspect\nAction: inventory\nObservation: ok\nThought: craft\nAction: craft plank',
        'plain invalid response',
    ]
    responses, response_mask = _batch_texts(tokenizer, texts)

    feature_mask = build_textcraft_action_feature_mask(responses, response_mask, tokenizer)
    selected_first = tokenizer.decode(responses[0, feature_mask[0]].tolist())
    selected_second = tokenizer.decode(responses[1, feature_mask[1]].tolist())

    assert selected_first == 'inventorycraft plank'
    assert selected_second == texts[1]


def test_fast_action_feature_mask_matches_slow_for_canonical_textcraft_rows():
    tokenizer = CharTokenizer()
    texts = [
        'Action: inventory',
        ' Thought: inspect\n Action: inventory\n Observation: ok',
        'Thought: inspect\nAction: inventory\nObservation: ok\nThought: craft\nAction: craft plank',
        'Thought: inspect\nAction: inventory\nAction: craft plank\nObservation: ok',
    ]
    responses, response_mask = _batch_texts(tokenizer, texts)

    fast_mask = build_textcraft_action_feature_mask_fast(responses, response_mask, tokenizer)
    slow_mask = build_textcraft_action_feature_mask_slow(responses, response_mask, tokenizer)
    default_mask = build_textcraft_action_feature_mask(responses, response_mask, tokenizer)

    assert torch.equal(fast_mask, slow_mask)
    assert torch.equal(default_mask, slow_mask)
    assert _selected_texts(tokenizer, responses, fast_mask) == [
        'inventory',
        'inventory',
        'inventorycraft plank',
        'inventorycraft plank',
    ]
    for text in texts:
        assert _fast_selection_for_text(tokenizer, text) is not None


def test_fast_action_feature_mask_uses_slow_fallback_for_no_marker_and_unsupported_formats():
    tokenizer = CharTokenizer()
    texts = [
        'plain invalid response',
        'Thought: inspect\nAction : inventory\nObservation: ok',
        'Action: inventory\nObservation : ok',
        'thought: inspect\naction: inventory\nobservation: ok',
    ]
    responses, response_mask = _batch_texts(tokenizer, texts)

    fast_mask = build_textcraft_action_feature_mask_fast(responses, response_mask, tokenizer)
    slow_mask = build_textcraft_action_feature_mask_slow(responses, response_mask, tokenizer)

    assert torch.equal(fast_mask, slow_mask)
    assert _selected_texts(tokenizer, responses, fast_mask) == [
        texts[0],
        'inventory',
        'inventory',
        'inventory',
    ]
    for text in texts:
        assert _fast_selection_for_text(tokenizer, text) is None


def test_normalized_action_feature_inputs_replace_response_tokens():
    tokenizer = CharTokenizer()
    prompts, prompt_attention_mask = _batch_texts(tokenizer, ['Goal: one', 'Goal: two'])
    prompt_position_ids = torch.arange(prompts.size(1), dtype=torch.long).unsqueeze(0).repeat(2, 1)

    feature_inputs = build_textcraft_normalized_action_feature_inputs(
        normalized_action_texts=['get 1 oak log\ncraft 4 oak planks', ''],
        prompts=prompts,
        prompt_attention_mask=prompt_attention_mask.to(dtype=torch.long),
        prompt_position_ids=prompt_position_ids,
        tokenizer=tokenizer,
    )

    selected_texts = _selected_texts(
        tokenizer,
        feature_inputs['responses'],
        feature_inputs['response_mask'],
    )

    assert selected_texts == ['get 1 oak log\ncraft 4 oak planks', 'no action']
    assert feature_inputs['input_ids'].shape[0] == 2
    assert feature_inputs['input_ids'].shape[1] == prompts.size(1) + feature_inputs['responses'].size(1)
    assert torch.equal(feature_inputs['attention_mask'][:, :prompts.size(1)], prompt_attention_mask.to(dtype=torch.long))
