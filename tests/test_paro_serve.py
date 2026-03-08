from __future__ import annotations

import unittest

from qwen3_server_manager.paro_serve import CallableTokenizerProxy, CompatibleQwenVLMProcessor


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
