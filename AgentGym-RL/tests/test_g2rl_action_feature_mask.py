import torch

from verl.workers.agent_fsdp_workers import build_textcraft_action_feature_mask, find_textcraft_action_spans


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
