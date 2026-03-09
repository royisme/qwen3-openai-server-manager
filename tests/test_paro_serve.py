from __future__ import annotations

import unittest

from qwen3_server_manager.paro_serve import (
    CallableTokenizerProxy,
    CompatibleQwenVLMProcessor,
    _RESPONSE_STORE,
    _apply_reasoning_aliases,
    _build_response_output_items,
    _delete_response,
    _extract_response_include,
    _normalize_responses_input,
    _parse_response_tool_result,
    _response_function_call_item,
    _response_store_path,
    _save_response_store,
    _sanitize_output_text,
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

    def test_normalize_responses_input_supports_itemized_message_and_tool_output(self) -> None:
        messages, images = _normalize_responses_input([
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'Look at this.'},
                    {'type': 'input_image', 'image_url': {'url': 'https://example.com/b.png'}},
                ],
            },
            {
                'type': 'function_call_output',
                'call_id': 'call_123',
                'output': {'result': 'ok'},
            },
        ])
        self.assertEqual(messages[0], {'role': 'user', 'content': 'Look at this.'})
        self.assertEqual(messages[1], {'role': 'tool', 'content': '{"result": "ok"}'})
        self.assertEqual(images, ['https://example.com/b.png'])

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


class ReasoningAliasTests(unittest.TestCase):
    def test_enable_thinking_false_passthrough(self) -> None:
        kwargs = {}
        _apply_reasoning_aliases({'enable_thinking': False}, kwargs)
        self.assertEqual(kwargs['enable_thinking'], False)

    def test_reasoning_enabled_false_maps_to_no_thinking(self) -> None:
        kwargs = {}
        _apply_reasoning_aliases({'reasoning': {'enabled': False}}, kwargs)
        self.assertEqual(kwargs['enable_thinking'], False)

    def test_no_thinking_true_maps_to_no_thinking(self) -> None:
        kwargs = {}
        _apply_reasoning_aliases({'no_thinking': True}, kwargs)
        self.assertEqual(kwargs['enable_thinking'], False)


class OutputSanitizerTests(unittest.TestCase):
    def test_sanitize_output_text_removes_special_tokens(self) -> None:
        text = 'OK<|im_end|>\n<|endoftext|>'
        self.assertEqual(_sanitize_output_text(text), 'OK\n')


class ResponsesOutputShapeTests(unittest.TestCase):
    def test_response_function_call_item_maps_chat_tool_call_shape(self) -> None:
        item = _response_function_call_item({
            'id': 'call_123',
            'function': {'name': 'lookup_weather', 'arguments': '{"city":"Toronto"}'},
        })
        self.assertEqual(item['type'], 'function_call')
        self.assertEqual(item['call_id'], 'call_123')
        self.assertEqual(item['name'], 'lookup_weather')

    def test_build_response_output_items_supports_function_calls(self) -> None:
        items, output_text = _build_response_output_items(
            text='Done',
            tool_calls=[{'id': 'call_1', 'function': {'name': 'lookup', 'arguments': '{}'}}],
            message_id='msg_1',
        )
        self.assertEqual(output_text, 'Done')
        self.assertEqual(items[0]['type'], 'message')
        self.assertEqual(items[1]['type'], 'function_call')

    def test_parse_response_tool_result_sanitizes_remaining_text(self) -> None:
        class _FakeServerModule:
            @staticmethod
            def process_tool_calls(model_output, tool_module, tools):
                return {'calls': [{'id': 'call_1', 'function': {'name': 'lookup', 'arguments': '{}'}}], 'remaining_text': 'OK<|im_end|>'}

        parsed = _parse_response_tool_result(_FakeServerModule, 'ignored', object(), [])
        self.assertEqual(parsed['remaining_text'], 'OK')

    def test_extract_response_include_validates_type(self) -> None:
        self.assertEqual(_extract_response_include(['output[0].content[0].text']), ['output[0].content[0].text'])
        with self.assertRaises(ValueError):
            _extract_response_include('output_text')
