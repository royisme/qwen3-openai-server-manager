from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
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


_RESPONSE_STORE: dict[str, dict] = {}


def _normalize_role(role: str | None) -> str:
    if role in {'system', 'developer'}:
        return 'system'
    if role in {'assistant', 'tool'}:
        return role
    return 'user'


def _extract_image_url(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get('url') or value.get('image_url')
    return None


def _normalize_responses_input(input_value):
    messages: list[dict] = []
    images: list[str] = []

    if input_value is None:
        return messages, images

    if isinstance(input_value, str):
        return [{'role': 'user', 'content': input_value}], []

    if not isinstance(input_value, list):
        raise ValueError('input must be a string or a list of messages')

    for item in input_value:
        if isinstance(item, str):
            messages.append({'role': 'user', 'content': item})
            continue
        if not isinstance(item, dict):
            raise ValueError('each input item must be a string or message object')

        role = _normalize_role(item.get('role'))
        content = item.get('content')
        if content is None:
            messages.append({'role': role, 'content': ''})
            continue
        if isinstance(content, str):
            messages.append({'role': role, 'content': content})
            continue

        if not isinstance(content, list):
            raise ValueError('message content must be a string or a list')

        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get('type')
            if part_type in {'input_text', 'text', 'output_text'}:
                value = part.get('text') or part.get('content') or ''
                if value:
                    text_parts.append(value)
            elif part_type in {'input_image', 'image_url'}:
                image_url = _extract_image_url(part.get('image_url', part))
                if image_url:
                    images.append(image_url)

        messages.append({'role': role, 'content': ' '.join(text_parts).strip()})

    return messages, images


def _assistant_output_message(text: str, message_id: str) -> dict:
    return {
        'id': message_id,
        'type': 'message',
        'status': 'completed',
        'role': 'assistant',
        'content': [
            {
                'type': 'output_text',
                'text': text,
                'annotations': [],
            }
        ],
    }


def _stored_messages_from_response(response_id: str) -> list[dict]:
    stored = _RESPONSE_STORE.get(response_id)
    if not stored:
        raise KeyError(response_id)
    return [dict(message) for message in stored.get('messages', [])]


def _patch_responses_routes(server_module) -> None:
    from fastapi import Body, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse

    app = server_module.app
    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, 'path', None) not in {'/responses', '/v1/responses', '/responses/{response_id}', '/v1/responses/{response_id}'}
    ]

    @app.post('/responses')
    @app.post('/v1/responses', include_in_schema=False)
    async def patched_responses_endpoint(body: dict = Body(...)):
        model_name = body.get('model')
        if not model_name:
            raise HTTPException(status_code=400, detail='model is required')

        previous_response_id = body.get('previous_response_id')
        store = body.get('store', True)
        instructions = body.get('instructions')
        stream = bool(body.get('stream', False))
        max_output_tokens = int(body.get('max_output_tokens', server_module.DEFAULT_MAX_TOKENS))
        temperature = float(body.get('temperature', server_module.DEFAULT_TEMPERATURE))
        top_p = float(body.get('top_p', server_module.DEFAULT_TOP_P))

        try:
            model, processor, config = server_module.get_cached_model(model_name)
            current_messages, images = _normalize_responses_input(body.get('input'))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"previous_response_id not found: {previous_response_id}")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if previous_response_id:
            try:
                prior_messages = _stored_messages_from_response(previous_response_id)
            except KeyError:
                raise HTTPException(status_code=404, detail=f'previous_response_id not found: {previous_response_id}')
        else:
            prior_messages = []

        chat_messages = prior_messages + current_messages
        if instructions:
            chat_messages = [{'role': 'system', 'content': instructions}] + chat_messages

        template_kwargs = {
            key: value
            for key, value in body.items()
            if key in server_module.ALLOWED_TEMPLATE_KWARGS
        }
        formatted_prompt = server_module.apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            **template_kwargs,
        )

        kwargs = dict(template_kwargs)
        generated_at = int(time.time())
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if stream:
            async def stream_generator():
                full_text = ''
                usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                base_response = {
                    'id': response_id,
                    'object': 'response',
                    'created_at': generated_at,
                    'status': 'in_progress',
                    'error': None,
                    'instructions': instructions,
                    'max_output_tokens': max_output_tokens,
                    'model': model_name,
                    'output': [],
                    'output_text': '',
                    'temperature': temperature,
                    'top_p': top_p,
                    'truncation': 'disabled',
                    'usage': usage,
                    'user': body.get('user'),
                }
                yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': base_response}, ensure_ascii=False)}\n\n"
                yield f"event: response.in_progress\ndata: {json.dumps({'type': 'response.in_progress', 'response': base_response}, ensure_ascii=False)}\n\n"
                item = {'id': message_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []}
                yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': item}, ensure_ascii=False)}\n\n"
                part = {'type': 'output_text', 'text': '', 'annotations': []}
                yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': part}, ensure_ascii=False)}\n\n"

                try:
                    token_iterator = server_module.stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                        top_p=top_p,
                        prefill_step_size=server_module.get_prefill_step_size(),
                        kv_bits=server_module.get_quantized_kv_bits(model_name),
                        kv_group_size=server_module.get_kv_group_size(),
                        max_kv_size=server_module.get_max_kv_size(model_name),
                        quantized_kv_start=server_module.get_quantized_kv_start(),
                        **kwargs,
                    )
                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, 'text'):
                            continue
                        full_text += chunk.text
                        usage = {
                            'input_tokens': chunk.prompt_tokens,
                            'output_tokens': chunk.generation_tokens,
                            'total_tokens': chunk.total_tokens,
                        }
                        yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'delta': chunk.text}, ensure_ascii=False)}\n\n"

                    final_part = {'type': 'output_text', 'text': full_text, 'annotations': []}
                    final_item = {'id': message_id, 'type': 'message', 'status': 'completed', 'role': 'assistant', 'content': [final_part]}
                    completed = dict(base_response)
                    completed.update({'status': 'completed', 'output': [final_item], 'output_text': full_text, 'usage': usage})
                    if store:
                        _RESPONSE_STORE[response_id] = {
                            'response': completed,
                            'messages': chat_messages + [{'role': 'assistant', 'content': full_text}],
                            'created_at': generated_at,
                        }
                    yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'text': full_text}, ensure_ascii=False)}\n\n"
                    yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': final_part}, ensure_ascii=False)}\n\n"
                    yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': final_item}, ensure_ascii=False)}\n\n"
                    yield f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': completed}, ensure_ascii=False)}\n\n"
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f'Generation failed: {exc}')
                finally:
                    server_module.mx.clear_cache()
                    server_module.gc.collect()

            return StreamingResponse(
                stream_generator(),
                media_type='text/event-stream',
                headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Accel-Buffering': 'no'},
            )

        try:
            result = server_module.generate(
                model=model,
                processor=processor,
                prompt=formatted_prompt,
                image=images,
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=top_p,
                prefill_step_size=server_module.get_prefill_step_size(),
                kv_bits=server_module.get_quantized_kv_bits(model_name),
                kv_group_size=server_module.get_kv_group_size(),
                max_kv_size=server_module.get_max_kv_size(model_name),
                quantized_kv_start=server_module.get_quantized_kv_start(),
                verbose=False,
                **kwargs,
            )
            response = {
                'id': response_id,
                'object': 'response',
                'created_at': generated_at,
                'status': 'completed',
                'error': None,
                'instructions': instructions,
                'max_output_tokens': max_output_tokens,
                'model': model_name,
                'output': [_assistant_output_message(result.text, message_id)],
                'output_text': result.text,
                'temperature': temperature,
                'top_p': top_p,
                'truncation': 'disabled',
                'usage': {
                    'input_tokens': result.prompt_tokens,
                    'output_tokens': result.generation_tokens,
                    'total_tokens': result.total_tokens,
                },
                'user': body.get('user'),
            }
            if store:
                _RESPONSE_STORE[response_id] = {
                    'response': response,
                    'messages': chat_messages + [{'role': 'assistant', 'content': result.text}],
                    'created_at': generated_at,
                }
            server_module.mx.clear_cache()
            server_module.gc.collect()
            return JSONResponse(response)
        except HTTPException:
            raise
        except Exception as exc:
            server_module.mx.clear_cache()
            server_module.gc.collect()
            raise HTTPException(status_code=500, detail=f'Generation failed: {exc}')

    @app.get('/responses/{response_id}')
    @app.get('/v1/responses/{response_id}', include_in_schema=False)
    async def patched_get_response(response_id: str):
        stored = _RESPONSE_STORE.get(response_id)
        if not stored:
            raise HTTPException(status_code=404, detail=f'response not found: {response_id}')
        return JSONResponse(stored['response'])


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
    _patch_responses_routes(mlx_vlm.server)
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
