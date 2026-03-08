from __future__ import annotations

import unittest

from qwen3_server_manager.paro_serve import (
    CallableTokenizerProxy,
    CompatibleQwenVLMProcessor,
    _RESPONSE_STORE,
    _delete_response,
    _normalize_responses_input,
    _response_store_path,
    _save_response_store,
    _load_response_store,
    _stored_messages_from_response,
)


class _FakeImageProcessor:
    merge_size = 2


class ParoServeTests(unittest.TestCase):
    def test_expand_image_tokens_matches_grid_size(self) -> None:
        processor = CompatibleQwenVLMProcessor.__new__(CompatibleQwenVLMProcessor)
        processor.image_processor = _FakeImageProcessor()
        processor.image_token = '<|image_pad|>'

        expanded = processor._expand_image_tokens(
            '<|vision_start|><|image_pad|><|vision_end|>请描述图片',
            [[1, 64, 96]],
        )

        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].count('<|image_pad|>'), 1536)
        self.assertIn('<|vision_start|>', expanded[0])
        self.assertIn('<|vision_end|>', expanded[0])


if __name__ == '__main__':
    unittest.main()


class _FakeHFTokenizer:
    def __call__(self, text, **kwargs):
        return {'text': text, 'kwargs': kwargs}


class _FakeTokenizerWrapper:
    def __init__(self):
        self._tokenizer = _FakeHFTokenizer()
        self.detokenizer = object()
        self.eos_token_id = 1

    def convert_tokens_to_ids(self, token):
        return 42


class TokenizerProxyTests(unittest.TestCase):
    def test_callable_tokenizer_proxy_delegates_calls(self) -> None:
        proxy = CallableTokenizerProxy(_FakeTokenizerWrapper())
        result = proxy('hello', add_special_tokens=False)
        self.assertEqual(result['text'], 'hello')
        self.assertEqual(result['kwargs']['add_special_tokens'], False)
        self.assertIsNotNone(proxy.detokenizer)


class ResponsesCompatTests(unittest.TestCase):
    def test_normalize_responses_input_supports_text_and_image(self) -> None:
        messages, images = _normalize_responses_input([
            {
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'Describe this image.'},
                    {'type': 'input_image', 'image_url': 'https://example.com/a.png'},
                ],
            }
        ])
        self.assertEqual(messages, [{'role': 'user', 'content': 'Describe this image.'}])
        self.assertEqual(images, ['https://example.com/a.png'])

    def test_normalize_responses_input_supports_output_text_history(self) -> None:
        messages, images = _normalize_responses_input([
            {
                'role': 'assistant',
                'content': [
                    {'type': 'output_text', 'text': 'Previous answer.'},
                ],
            }
        ])
        self.assertEqual(messages, [{'role': 'assistant', 'content': 'Previous answer.'}])
        self.assertEqual(images, [])

    def test_stored_messages_from_response_round_trip(self) -> None:
        _RESPONSE_STORE['resp_test'] = {
            'messages': [
                {'role': 'user', 'content': 'Hi'},
                {'role': 'assistant', 'content': 'Hello'},
            ]
        }
        try:
            messages = _stored_messages_from_response('resp_test')
        finally:
            _RESPONSE_STORE.pop('resp_test', None)
        self.assertEqual(messages[0]['content'], 'Hi')
        self.assertEqual(messages[1]['content'], 'Hello')


class ResponseStoreTests(unittest.TestCase):
    def test_delete_response_removes_entry(self) -> None:
        _RESPONSE_STORE['resp_delete'] = {'messages': []}
        self.assertTrue(_delete_response('resp_delete'))
        self.assertNotIn('resp_delete', _RESPONSE_STORE)

    def test_response_store_persists_to_disk(self) -> None:
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            old = os.environ.get('QWEN3_RUNTIME_DIR')
            os.environ['QWEN3_RUNTIME_DIR'] = tmpdir
            _RESPONSE_STORE.clear()
            _RESPONSE_STORE['resp_disk'] = {'response': {'id': 'resp_disk'}, 'messages': [], 'created_at': 1}
            _save_response_store()
            _RESPONSE_STORE.clear()
            _load_response_store()
            try:
                self.assertIn('resp_disk', _RESPONSE_STORE)
                self.assertTrue(_response_store_path().exists())
            finally:
                if old is None:
                    os.environ.pop('QWEN3_RUNTIME_DIR', None)
                else:
                    os.environ['QWEN3_RUNTIME_DIR'] = old
