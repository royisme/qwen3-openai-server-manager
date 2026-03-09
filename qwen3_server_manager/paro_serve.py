from __future__ import annotations

import argparse
import json
import importlib
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


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

_SPECIAL_OUTPUT_TOKENS = (
    '<|im_end|>',
    '<|endoftext|>',
)


def _response_store_path() -> Path:
    runtime_dir = Path(os.environ.get('QWEN3_RUNTIME_DIR', '.runtime/qwen3-server-manager'))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir / 'responses_store.json'


def _load_response_store() -> None:
    path = _response_store_path()
    if not path.exists():
        return
    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        loaded = {}
    _RESPONSE_STORE.clear()
    _RESPONSE_STORE.update(loaded)


def _save_response_store() -> None:
    path = _response_store_path()
    path.write_text(json.dumps(_RESPONSE_STORE, ensure_ascii=False, indent=2), encoding='utf-8')


def _store_response(response_id: str, response: dict[str, Any], messages: list[dict], created_at: int) -> None:
    _RESPONSE_STORE[response_id] = {
        'response': response,
        'messages': messages,
        'created_at': created_at,
    }
    _save_response_store()


def _delete_response(response_id: str) -> bool:
    removed = _RESPONSE_STORE.pop(response_id, None) is not None
    if removed:
        _save_response_store()
    return removed


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


def _apply_reasoning_aliases(body: dict, kwargs: dict) -> None:
    if body.get('enable_thinking') is False:
        kwargs['enable_thinking'] = False
        return
    reasoning = body.get('reasoning')
    no_thinking = body.get('no_thinking')
    if no_thinking is True:
        kwargs['enable_thinking'] = False
        return
    if isinstance(reasoning, dict):
        if reasoning.get('enabled') is False:
            kwargs['enable_thinking'] = False
        elif reasoning.get('effort') in {'none', 'minimal'}:
            kwargs['enable_thinking'] = False


def _sanitize_output_text(text: str) -> str:
    sanitized = text
    for token in _SPECIAL_OUTPUT_TOKENS:
        sanitized = sanitized.replace(token, '')
    return sanitized

def _stringify_content_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_response_content_parts(content, images: list[str]) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
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
        elif part_type == 'function_call_output':
            value = _stringify_content_value(part.get('output'))
            if value:
                text_parts.append(value)
    return ' '.join(text_parts).strip()


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

        item_type = item.get('type')
        if item_type == 'message':
            role = _normalize_role(item.get('role'))
            messages.append({'role': role, 'content': _normalize_response_content_parts(item.get('content'), images)})
            continue
        if item_type in {'input_text', 'output_text', 'text'}:
            messages.append({'role': 'user', 'content': _stringify_content_value(item.get('text') or item.get('content'))})
            continue
        if item_type in {'input_image', 'image_url'}:
            image_url = _extract_image_url(item.get('image_url', item))
            if image_url:
                images.append(image_url)
            continue
        if item_type == 'function_call_output':
            output_value = _stringify_content_value(item.get('output'))
            messages.append({'role': 'tool', 'content': output_value})
            continue
        if item_type == 'function_call':
            call_name = item.get('name') or 'tool'
            call_arguments = _stringify_content_value(item.get('arguments'))
            messages.append({'role': 'assistant', 'content': f'Tool call {call_name}: {call_arguments}'.strip()})
            continue

        role = _normalize_role(item.get('role'))
        content = item.get('content')
        if content is None:
            messages.append({'role': role, 'content': ''})
            continue
        messages.append({'role': role, 'content': _normalize_response_content_parts(content, images)})

    return messages, images


def _response_function_call_item(call: dict) -> dict:
    function = call.get('function', {})
    return {
        'id': call.get('id') or f"fc_{uuid.uuid4().hex}",
        'type': 'function_call',
        'call_id': call.get('id') or f"call_{uuid.uuid4().hex}",
        'name': function.get('name'),
        'arguments': function.get('arguments', '{}'),
        'status': 'completed',
    }


def _build_response_output_items(text: str, tool_calls: list[dict], message_id: str) -> tuple[list[dict], str]:
    sanitized_text = _sanitize_output_text(text)
    items: list[dict] = []
    if sanitized_text.strip() or not tool_calls:
        items.append(_assistant_output_message(sanitized_text, message_id))
    items.extend(_response_function_call_item(call) for call in tool_calls)
    return items, sanitized_text


def _extract_response_include(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError('include must be a list of strings')


def _resolve_response_tools(server_module, processor, tools):
    if not tools:
        return None, None
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    if not hasattr(tokenizer, 'chat_template'):
        return None, None
    tool_parser_type = server_module._infer_tool_parser(tokenizer.chat_template)
    if tool_parser_type is None:
        return None, None
    tool_module = importlib.import_module(f'mlx_lm.tool_parsers.{tool_parser_type}')
    return tool_parser_type, tool_module


def _parse_response_tool_result(server_module, text: str, tool_module, tools):
    sanitized_text = _sanitize_output_text(text)
    if tool_module is None:
        return {'calls': [], 'remaining_text': sanitized_text}
    parsed = server_module.process_tool_calls(model_output=sanitized_text, tool_module=tool_module, tools=tools)
    parsed['remaining_text'] = _sanitize_output_text(parsed.get('remaining_text', ''))
    return parsed


def _build_response_payload(
    response_id: str,
    generated_at: int,
    instructions: str | None,
    max_output_tokens: int,
    model_name: str,
    output_items: list[dict],
    output_text: str,
    temperature: float,
    top_p: float,
    usage: dict,
    user: str | None,
    metadata: dict | None = None,
    input_items: Any = None,
) -> dict:
    return {
        'id': response_id,
        'object': 'response',
        'created_at': generated_at,
        'completed_at': int(time.time()),
        'status': 'completed',
        'error': None,
        'incomplete_details': None,
        'instructions': instructions,
        'input': input_items if input_items is not None else [],
        'max_output_tokens': max_output_tokens,
        'model': model_name,
        'output': output_items,
        'output_text': output_text,
        'temperature': temperature,
        'top_p': top_p,
        'truncation': 'disabled',
        'usage': usage,
        'user': user,
        'metadata': metadata or {},
    }


def _stream_event_payload(event_type: str, sequence_number: int, **payload) -> dict:
    return {'type': event_type, 'sequence_number': sequence_number, **payload}


def _response_in_progress_payload(
    response_id: str,
    generated_at: int,
    instructions: str | None,
    max_output_tokens: int,
    model_name: str,
    temperature: float,
    top_p: float,
    usage: dict,
    user: str | None,
    metadata: dict | None = None,
    input_items: Any = None,
) -> dict:
    return {
        'id': response_id,
        'object': 'response',
        'created_at': generated_at,
        'status': 'in_progress',
        'error': None,
        'incomplete_details': None,
        'instructions': instructions,
        'input': input_items if input_items is not None else [],
        'max_output_tokens': max_output_tokens,
        'model': model_name,
        'output': [],
        'output_text': '',
        'temperature': temperature,
        'top_p': top_p,
        'truncation': 'disabled',
        'usage': usage,
        'user': user,
        'metadata': metadata or {},
    }


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
    _load_response_store()
    if not getattr(app.state, 'qwen3_openai_error_handlers_installed', False):
        from fastapi.responses import JSONResponse
        from starlette.exceptions import HTTPException as StarletteHTTPException

        @app.exception_handler(StarletteHTTPException)
        async def openai_http_exception_handler(request, exc):
            if request.url.path.startswith('/v1/') or request.url.path.startswith('/responses'):
                detail = exc.detail
                message = detail.get('message') if isinstance(detail, dict) else str(detail)
                code = detail.get('code') if isinstance(detail, dict) else None
                param = detail.get('param') if isinstance(detail, dict) else None
                err_type = detail.get('type') if isinstance(detail, dict) else 'invalid_request_error'
                return JSONResponse(status_code=exc.status_code, content={'error': {'message': message, 'type': err_type, 'param': param, 'code': code}})
            raise exc

        @app.exception_handler(Exception)
        async def openai_exception_handler(request, exc):
            if request.url.path.startswith('/v1/') or request.url.path.startswith('/responses'):
                return JSONResponse(status_code=500, content={'error': {'message': str(exc), 'type': 'server_error', 'param': None, 'code': None}})
            raise exc

        app.state.qwen3_openai_error_handlers_installed = True

    app.router.routes = [
        route
        for route in app.router.routes
        if getattr(route, 'path', None) not in {
            '/responses',
            '/v1/responses',
            '/responses/{response_id}',
            '/v1/responses/{response_id}',
            '/chat/completions',
            '/v1/chat/completions',
        }
    ]

    original_chat_endpoint = getattr(server_module, 'chat_completions_endpoint', None)

    @app.post('/chat/completions', response_model=None)
    @app.post('/v1/chat/completions', response_model=None, include_in_schema=False)
    async def patched_chat_completions_endpoint(body: dict = Body(...)):
        if original_chat_endpoint is None:
            raise HTTPException(
                status_code=500,
                detail={
                    'message': 'chat completions endpoint is unavailable',
                    'type': 'server_error',
                    'param': None,
                    'code': None,
                },
            )
        try:
            request_model = server_module.ChatRequest.model_validate(body)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    'message': str(exc),
                    'type': 'invalid_request_error',
                    'param': 'body',
                    'code': None,
                },
            )

        extras = dict(getattr(request_model, '__pydantic_extra__', {}) or {})
        _apply_reasoning_aliases(body, extras)
        request_model.__pydantic_extra__ = extras
        return await original_chat_endpoint(request_model)

    @app.post('/responses')
    @app.post('/v1/responses', include_in_schema=False)
    async def patched_responses_endpoint(body: dict = Body(...)):
        model_name = body.get('model')
        if not model_name:
            raise HTTPException(status_code=400, detail={'message': 'model is required', 'type': 'invalid_request_error', 'param': 'model', 'code': None})

        previous_response_id = body.get('previous_response_id')
        store = body.get('store', True)
        instructions = body.get('instructions')
        stream = bool(body.get('stream', False))
        try:
            include = _extract_response_include(body.get('include'))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={'message': str(exc), 'type': 'invalid_request_error', 'param': 'include', 'code': None})
        metadata = body.get('metadata') if isinstance(body.get('metadata'), dict) else {}
        tools = body.get('tools') if isinstance(body.get('tools'), list) else None
        max_output_tokens = int(body.get('max_output_tokens', server_module.DEFAULT_MAX_TOKENS))
        temperature = float(body.get('temperature', server_module.DEFAULT_TEMPERATURE))
        top_p = float(body.get('top_p', server_module.DEFAULT_TOP_P))

        try:
            model, processor, config = server_module.get_cached_model(model_name)
            current_messages, images = _normalize_responses_input(body.get('input'))
        except KeyError:
            raise HTTPException(status_code=404, detail={'message': f'previous_response_id not found: {previous_response_id}', 'type': 'not_found_error', 'param': 'previous_response_id', 'code': None})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={'message': str(exc), 'type': 'invalid_request_error', 'param': 'input', 'code': None})

        if previous_response_id:
            try:
                prior_messages = _stored_messages_from_response(previous_response_id)
            except KeyError:
                raise HTTPException(status_code=404, detail={'message': f'previous_response_id not found: {previous_response_id}', 'type': 'not_found_error', 'param': 'previous_response_id', 'code': None})
        else:
            prior_messages = []

        chat_messages = prior_messages + current_messages
        if instructions:
            chat_messages = [{'role': 'system', 'content': instructions}] + chat_messages

        _, tool_module = _resolve_response_tools(server_module, processor, tools)

        template_kwargs = {
            key: value
            for key, value in body.items()
            if key in server_module.ALLOWED_TEMPLATE_KWARGS
        }
        _apply_reasoning_aliases(body, template_kwargs)
        formatted_prompt = server_module.apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            **template_kwargs,
        )

        kwargs = dict(template_kwargs)
        _apply_reasoning_aliases(body, kwargs)
        generated_at = int(time.time())
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if stream:
            async def stream_generator():
                full_text = ''
                usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                sequence_number = 0

                def emit(event_type: str, **payload) -> str:
                    nonlocal sequence_number
                    sequence_number += 1
                    return f"event: {event_type}\ndata: {json.dumps(_stream_event_payload(event_type, sequence_number, **payload), ensure_ascii=False)}\n\n"

                base_response = _response_in_progress_payload(
                    response_id=response_id,
                    generated_at=generated_at,
                    instructions=instructions,
                    max_output_tokens=max_output_tokens,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    usage=usage,
                    user=body.get('user'),
                    metadata=metadata,
                    input_items=body.get('input'),
                )
                yield emit('response.created', response=base_response)
                yield emit('response.in_progress', response=base_response)
                item = {'id': message_id, 'type': 'message', 'status': 'in_progress', 'role': 'assistant', 'content': []}
                yield emit('response.output_item.added', output_index=0, item=item)
                part = {'type': 'output_text', 'text': '', 'annotations': []}
                yield emit('response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=part)

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
                        chunk.text = _sanitize_output_text(chunk.text)
                        full_text += chunk.text
                        usage = {
                            'input_tokens': chunk.prompt_tokens,
                            'output_tokens': chunk.generation_tokens,
                            'total_tokens': chunk.total_tokens,
                        }
                        yield emit('response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=chunk.text)

                    parsed = _parse_response_tool_result(server_module, full_text, tool_module, tools)
                    output_items, output_text = _build_response_output_items(parsed['remaining_text'], parsed['calls'], message_id)
                    final_part = {'type': 'output_text', 'text': output_text, 'annotations': []}
                    completed = _build_response_payload(
                        response_id=response_id,
                        generated_at=generated_at,
                        instructions=instructions,
                        max_output_tokens=max_output_tokens,
                        model_name=model_name,
                        output_items=output_items,
                        output_text=output_text,
                        temperature=temperature,
                        top_p=top_p,
                        usage=usage,
                        user=body.get('user'),
                        metadata=metadata,
                        input_items=body.get('input'),
                    )
                    if include:
                        completed['include'] = include
                    if store:
                        _store_response(response_id, completed, chat_messages + [{'role': 'assistant', 'content': output_text}], generated_at)
                    yield emit('response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=output_text)
                    yield emit('response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_part)
                    for output_index, item in enumerate(output_items):
                        if item.get('type') == 'function_call':
                            yield emit('response.output_item.added', output_index=output_index, item=item)
                            yield emit(
                                'response.function_call_arguments.done',
                                item_id=item['id'],
                                output_index=output_index,
                                name=item.get('name'),
                                arguments=item.get('arguments', '{}'),
                            )
                        yield emit('response.output_item.done', output_index=output_index, item=item)
                    yield emit('response.completed', response=completed)
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=500, detail={'message': f'Generation failed: {exc}', 'type': 'server_error', 'param': None, 'code': None})
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
            parsed = _parse_response_tool_result(server_module, result.text, tool_module, tools)
            output_items, output_text = _build_response_output_items(parsed['remaining_text'], parsed['calls'], message_id)
            response = _build_response_payload(
                response_id=response_id,
                generated_at=generated_at,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                model_name=model_name,
                output_items=output_items,
                output_text=output_text,
                temperature=temperature,
                top_p=top_p,
                usage={
                    'input_tokens': result.prompt_tokens,
                    'output_tokens': result.generation_tokens,
                    'total_tokens': result.total_tokens,
                },
                user=body.get('user'),
                metadata=metadata,
                input_items=body.get('input'),
            )
            if include:
                response['include'] = include
            if store:
                _store_response(
                    response_id,
                    response,
                    chat_messages + [{'role': 'assistant', 'content': output_text}],
                    generated_at,
                )
            server_module.mx.clear_cache()
            server_module.gc.collect()
            return JSONResponse(response)
        except HTTPException:
            raise
        except Exception as exc:
            server_module.mx.clear_cache()
            server_module.gc.collect()
            raise HTTPException(status_code=500, detail={'message': f'Generation failed: {exc}', 'type': 'server_error', 'param': None, 'code': None})

    @app.get('/responses/{response_id}')
    @app.get('/v1/responses/{response_id}', include_in_schema=False)
    async def patched_get_response(response_id: str):
        stored = _RESPONSE_STORE.get(response_id)
        if not stored:
            raise HTTPException(status_code=404, detail={'message': f'response not found: {response_id}', 'type': 'not_found_error', 'param': 'response_id', 'code': None})
        return JSONResponse(stored['response'])

    @app.delete('/responses/{response_id}')
    @app.delete('/v1/responses/{response_id}', include_in_schema=False)
    async def patched_delete_response(response_id: str):
        if not _delete_response(response_id):
            raise HTTPException(status_code=404, detail={'message': f'response not found: {response_id}', 'type': 'not_found_error', 'param': 'response_id', 'code': None})
        return JSONResponse({'id': response_id, 'object': 'response', 'deleted': True})


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

    original_generate = mlx_vlm.server.generate
    original_stream_generate = mlx_vlm.server.stream_generate

    def patched_generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        if hasattr(result, 'text'):
            result.text = _sanitize_output_text(result.text)
        return result

    def patched_stream_generate(*args, **kwargs):
        for chunk in original_stream_generate(*args, **kwargs):
            if chunk is not None and hasattr(chunk, 'text'):
                chunk.text = _sanitize_output_text(chunk.text)
            yield chunk

    mlx_vlm.server.generate = patched_generate
    mlx_vlm.server.stream_generate = patched_stream_generate
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
