from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


def _resolve_model_dir(model_path: str) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = Path(model_path)
    if local_dir.is_dir():
        return local_dir
    return Path(snapshot_download(model_path))


def _strip_arg(argv: list[str], name: str) -> list[str]:
    result = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == name:
            skip_next = True
            continue
        if item.startswith(name + '='):
            continue
        result.append(item)
    return result


def _is_vlm_model(model_path: str) -> bool:
    from mlx_vlm.utils import load_config

    config = load_config(_resolve_model_dir(model_path))
    return 'vision_config' in config


class ImageProcessorProxy:
    def __init__(self, image_processor):
        self._image_processor = image_processor

    def __call__(self, *args, **kwargs):
        return self._image_processor(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        return self._image_processor.preprocess(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._image_processor, name)


class CallableTokenizerProxy:
    def __init__(self, tokenizer_wrapper):
        self._wrapper = tokenizer_wrapper
        self._tokenizer = tokenizer_wrapper._tokenizer

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapper, name)


class CompatibleQwenVLMProcessor:
    def __init__(self, tokenizer_wrapper, image_processor):
        from mlx_vlm.utils import StoppingCriteria

        self.tokenizer = CallableTokenizerProxy(tokenizer_wrapper)
        self.detokenizer = self.tokenizer.detokenizer
        self.image_processor = ImageProcessorProxy(image_processor)
        self.image_token = getattr(self.tokenizer, 'image_token', '<|image_pad|>')
        self.image_token_id = getattr(self.tokenizer, 'image_token_id', None) or self.tokenizer.convert_tokens_to_ids(
            self.image_token
        )
        self.stopping_criteria = StoppingCriteria(
            getattr(self.tokenizer, 'eos_token_id', None), tokenizer=self.tokenizer._tokenizer
        )
        self.tokenizer.stopping_criteria = self.stopping_criteria

    def __call__(self, text, **kwargs):
        return self.tokenizer._tokenizer(text, add_special_tokens=False, **kwargs)

    def _expand_image_tokens(self, text, image_grid_thw) -> list[str]:
        if isinstance(text, str):
            text_items = [text]
        else:
            text_items = list(text)

        merge_length = self.image_processor.merge_size**2
        image_index = 0
        for idx, item in enumerate(text_items):
            while self.image_token in item and image_index < len(image_grid_thw):
                num_image_tokens = int(math.prod(image_grid_thw[image_index]) // merge_length)
                item = item.replace(self.image_token, '<|placeholder|>' * num_image_tokens, 1)
                image_index += 1
            text_items[idx] = item.replace('<|placeholder|>', self.image_token)
        return text_items

    def process(
        self,
        text,
        images=None,
        padding=True,
        return_tensors='np',
        padding_side='left',
        add_special_tokens=False,
        **kwargs,
    ):
        image_features = None
        if images:
            image_features = self.image_processor(images=images, return_tensors='np')
            image_grid_thw = image_features.get('image_grid_thw')
            if image_grid_thw is not None:
                text = self._expand_image_tokens(text, image_grid_thw)

        hf_tokenizer = self.tokenizer._tokenizer
        original_padding_side = getattr(hf_tokenizer, 'padding_side', 'left')
        hf_tokenizer.padding_side = padding_side
        try:
            encoded = hf_tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                add_special_tokens=add_special_tokens,
            )
        finally:
            hf_tokenizer.padding_side = original_padding_side

        result = dict(encoded)
        if image_features is not None:
            result['pixel_values'] = image_features['pixel_values']
            if 'image_grid_thw' in image_features:
                result['image_grid_thw'] = image_features['image_grid_thw']
        return result

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def _build_vlm_processor(local_dir: Path):
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from transformers import AutoImageProcessor

    tokenizer_wrapper = load_tokenizer(local_dir)
    image_processor = AutoImageProcessor.from_pretrained(local_dir, trust_remote_code=True)
    return CompatibleQwenVLMProcessor(tokenizer_wrapper, image_processor)


def _serve_vllm():
    from paroquant.cli.serve import _serve_vllm as _upstream_serve_vllm

    _upstream_serve_vllm()


def _serve_mlx_text():
    from paroquant.cli.serve import _serve_mlx as _upstream_serve_mlx

    _upstream_serve_mlx()


def _serve_mlx_vlm():
    import uvicorn
    import mlx_vlm.server

    from paroquant.inference.backends.mlx.load import load as paro_load

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--prefill-step-size', type=int, default=None)
    parser.add_argument('--kv-bits', type=int, default=None)
    parser.add_argument('--kv-group-size', type=int, default=None)
    parser.add_argument('--max-kv-size', type=int, default=None)
    parser.add_argument('--quantized-kv-start', type=int, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])

    if args.trust_remote_code:
        os.environ['MLX_TRUST_REMOTE_CODE'] = 'true'
    if args.prefill_step_size is not None:
        os.environ['PREFILL_STEP_SIZE'] = str(args.prefill_step_size)
    if args.kv_bits is not None:
        os.environ['KV_BITS'] = str(args.kv_bits)
    if args.kv_group_size is not None:
        os.environ['KV_GROUP_SIZE'] = str(args.kv_group_size)
    if args.max_kv_size is not None:
        os.environ['MAX_KV_SIZE'] = str(args.max_kv_size)
    if args.quantized_kv_start is not None:
        os.environ['QUANTIZED_KV_START'] = str(args.quantized_kv_start)

    def _patched_load(model_path, adapter_path=None, trust_remote_code=False, **kwargs):
        local_dir = _resolve_model_dir(model_path)
        model, _, is_vlm = paro_load(str(local_dir), force_text=False)
        if not is_vlm:
            raise RuntimeError(f'Model {model_path} was not loaded as VLM')
        weight = model.vision_tower.patch_embed.proj.weight
        if len(weight.shape) == 5 and weight.shape[1] == 3:
            model.vision_tower.patch_embed.proj.weight = weight.transpose(0, 2, 3, 4, 1)
        processor = _build_vlm_processor(local_dir)
        return model, processor

    mlx_vlm.server.load = _patched_load
    uvicorn.run(mlx_vlm.server.app, host=args.host, port=args.port, workers=1, reload=False)


def main():
    from paroquant.inference.base import detect_backend

    backend = detect_backend()
    if backend in ('vllm', 'transformers'):
        _serve_vllm()
        return
    if backend != 'mlx':
        raise RuntimeError(f'Serve requires vllm or mlx. Detected: {backend}')

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.model and _is_vlm_model(args.model):
        sys.argv = [sys.argv[0]] + _strip_arg(sys.argv[1:], '--model')
        _serve_mlx_vlm()
    else:
        _serve_mlx_text()


if __name__ == '__main__':
    main()
