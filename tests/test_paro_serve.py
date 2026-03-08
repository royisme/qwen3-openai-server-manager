from __future__ import annotations

import unittest

from qwen3_server_manager.paro_serve import CompatibleQwenVLMProcessor


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
