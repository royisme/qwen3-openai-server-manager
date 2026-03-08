# Qwen3 OpenAI Server Manager

A production-oriented local lifecycle manager for the OpenAI-compatible API server around `z-lab/Qwen3.5-4B-PARO`.

This project is designed for people who want a **clean local service experience** instead of manually juggling virtual environments, backend selection, process supervision, logs, and platform-specific serving details.

## Attribution

This repository is a **secondary development / engineering-completion wrapper** built on top of the original `paroquant` project by `z-lab`.

- Upstream project: <https://github.com/z-lab/paroquant>
- Upstream model: <https://huggingface.co/z-lab/Qwen3.5-4B-PARO>
- This repository is **not** the official upstream repository
- This repository focuses on local service lifecycle management, OpenAI-compatible serving ergonomics, macOS `launchd` integration, and MLX multimodal serving fixes for `Qwen3.5-4B-PARO`

In short: upstream provides the core inference stack, while this repository turns it into a more complete local service workflow.

## Highlights

- OpenAI-compatible local API for `z-lab/Qwen3.5-4B-PARO`
- Works well as a free local provider for agent runtimes and tools
- Apple Silicon friendly with automatic `mlx` routing
- Supports image understanding in the local serving path
- Includes process lifecycle management, health checks, logs, and config-driven startup
- Supports persistent macOS background service management via `launchd`
- Designed as a practical companion layer for privacy-aware and cost-aware AI workflows

## What This Project Adds

Compared with the raw upstream serving flow, this project adds:

- Background start / stop / restart management
- PID-aware single-instance process control
- Health checks and status inspection
- Persistent log files with easy tailing
- Default config loading from `config/config.json`
- `uv`-based dependency bootstrapping
- macOS `launchd` service installation and management
- MLX multimodal serving support for image requests on `Qwen3.5-4B-PARO`

## Verified Capabilities

The following capabilities have been verified locally on **Apple Silicon + MLX**:

- OpenAI-compatible text generation works
- OpenAI-style tool calling works
- OpenAI-compatible `responses` API works
- Image input via `image_url` works
- Local service management commands work
- `launchd` service mode works

Important nuance:

- The upstream `paroquant.cli.chat` path is still text-only
- The upstream default MLX serving path does not fully expose the image pipeline for this model
- This repository patches the serving wrapper to make the multimodal path usable in a local OpenAI-compatible server setup

## Backend Selection

The manager auto-selects the backend by platform:

- `macOS + Apple Silicon` -> `mlx`
- `Linux + NVIDIA` -> `vllm`

You can also override it manually:

```bash
uv run qwen3-server-manager start --backend mlx
uv run qwen3-server-manager start --backend vllm
```

## Commands

Available commands:

- `doctor`
- `start`
- `stop`
- `restart`
- `status`
- `health`
- `logs`
- `install-service`
- `uninstall-service`
- `start-service`
- `stop-service`
- `restart-service`
- `service-status`

## Prerequisite

Install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Official docs: <https://docs.astral.sh/uv/>

Then, from the project directory:

```bash
uv sync
```

After that, use the CLI via:

```bash
uv run qwen3-server-manager <command>
```

## Quick Start

### 1. Check environment

```bash
uv run qwen3-server-manager doctor
```

### 2. Start the server

```bash
uv run qwen3-server-manager start
```

By default, it reads `config/config.json`.

### 3. Check status

```bash
uv run qwen3-server-manager status
```

The status output includes lifecycle-oriented fields such as:

- `mode`: whether the server is running as a direct process or through `launchd`
- `lifecycle_state`: typically `starting`, `running`, or `stopped`
- `ready`: whether the health check is already passing

This is especially useful in service mode, where `launchd` may have already started the process while the model is still warming up.

### 4. Check logs

```bash
uv run qwen3-server-manager logs --tail 80
```

### 5. Stop the server

```bash
uv run qwen3-server-manager stop
```

## Default Configuration

Default config file:

```text
config/config.json
```

Current default example:

```json
{
  "model": "z-lab/Qwen3.5-4B-PARO",
  "host": "127.0.0.1",
  "port": 8288,
  "backend": "mlx",
  "extra_args": []
}
```

Once this file is configured, you can simply run:

```bash
uv run qwen3-server-manager start
```

CLI flags override config values.

## OpenAI-Compatible Responses API

This project also works with OpenAI-style `responses` requests, which is useful if you are migrating from Chat Completions to the newer Responses API shape.

The compatibility layer currently supports:

- plain string `input`
- message-list `input`
- `store: true`
- `previous_response_id` for follow-up turns
- local disk-backed response persistence for stored responses
- `GET /v1/responses/{response_id}` for retrieving stored responses
- `DELETE /v1/responses/{response_id}` for deleting stored responses
- OpenAI-style `error` objects for `/v1/*` failures

Example:

```bash
curl -X POST "http://127.0.0.1:8288/v1/responses" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "z-lab/Qwen3.5-4B-PARO",
    "input": "Say hello in one short sentence.",
    "max_output_tokens": 32,
    "temperature": 0
  }'
```

## OpenAI-Compatible Image Request Example

After the server starts, you can call `/v1/chat/completions` directly:

```bash
curl -X POST "http://127.0.0.1:8288/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "z-lab/Qwen3.5-4B-PARO",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image in one sentence."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 80,
    "temperature": 0
  }'
```

For local images, converting them to a `data:image/...;base64,...` URL is recommended.

## macOS Service Mode

If you want the server to run like a real background service on macOS, use `launchd`:

```bash
uv run qwen3-server-manager install-service
uv run qwen3-server-manager restart-service
uv run qwen3-server-manager service-status
```

Default service label:

```text
local.qwen3-server-manager
```

The generated plist is installed to:

```text
~/Library/LaunchAgents/local.qwen3-server-manager.plist
```

## Runtime Directory

Runtime files are stored under:

```text
.runtime/qwen3-server-manager/
```

Typical contents:

- `state.json`
- `server.log`
- `venv/`
- `launchd/`

## Why This Exists

The upstream project already provides the core serving capability, but many local users still need a more complete operational layer:

- repeatable startup
- safe restart behavior
- process supervision
- predictable config loading
- logs and health inspection
- beginner-friendly installation flow
- platform-aware backend selection

This repository exists to fill that engineering gap.

