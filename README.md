# Qwen3 OpenAI Server Manager

一个给 `paroquant.cli.serve` 增加本地生命周期管理的轻量包装器。

核心目标是：**用户只需要装 `uv`，剩下的环境准备、依赖安装、启动和管理都自动完成。**

## 已确认的上游能力

`z-lab/Qwen3.5-4B-PARO` 的官方推荐启动方式是：

```bash
python -m paroquant.cli.serve --model z-lab/Qwen3.5-4B-PARO --port 8000
```

也就是说，上游已经提供 OpenAI-Compatible API Server；但它默认是前台进程，没有现成的：

- 后台启动
- 停止 / 重启
- 状态查询
- 日志落盘 / tail
- 健康检查
- 自动安装依赖

本项目只补这一层“管理壳”，不改 ParoQuant 推理逻辑。

## Attribution

This repository is a secondary development / production-oriented wrapper built on top of the original `paroquant` project by `z-lab`:

- Upstream project: <https://github.com/z-lab/paroquant>
- Upstream paper/project: ParoQuant
- This repository focuses on local service lifecycle management, OpenAI-compatible serving ergonomics, macOS `launchd` integration, and MLX multimodal serving fixes for `z-lab/Qwen3.5-4B-PARO`.

It is an independent follow-up implementation for engineering completeness and local usability, rather than the official upstream repository.

## 自动后端选择

管理器会按平台自动选择推理后端：

- Apple Silicon（`macOS + arm64`）→ `mlx`
- NVIDIA Linux（检测到 `nvidia-smi`）→ `vllm`

也支持手动覆盖：

```bash
uv run qwen3-server-manager start --backend mlx ...
uv run qwen3-server-manager start --backend vllm ...
```

## 功能

- `doctor`: 检查 `uv`、平台、推荐后端、运行时环境状态
- `start`: 自动建环境、安装依赖并后台启动 OpenAI-Compatible API Server
- `stop`: 优雅停止，必要时强制终止
- `restart`: 重启服务
- `status`: 查看进程、端口、健康状态
- `health`: 主动检查 `http://host:port/health` 与 `http://host:port/v1/models`
- `logs`: 查看落盘日志，支持 tail
- `install-service` / `restart-service`: 用 `launchd` 作为真正的 macOS 后台服务

## 已确认的多模态能力

当前在 **Apple Silicon + MLX** 路线下，`z-lab/Qwen3.5-4B-PARO` 已完成本地实测：

- 文本 OpenAI-Compatible API 可用
- OpenAI 风格 `tools` / tool calling 可用
- `image_url` 多模态请求可用

需要注意：

- 上游原始 `paroquant.cli.chat` 目前仍是文本交互，不支持直接传图
- 上游默认 `paroquant.cli.serve` 的 MLX 文本路径没有把这条视觉链路完整打通
- 本项目在 `qwen3_server_manager/paro_serve.py` 中补齐了 Qwen3.5-4B-PARO 在 MLX 下的视觉服务包装

## 唯一前置依赖

先安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或参考官方文档：<https://docs.astral.sh/uv/>

推荐在项目目录先执行一次：

```bash
uv sync
```

这样后续统一使用：

```bash
uv run qwen3-server-manager <command>
```

之后本项目不要求用户手动：

- 创建虚拟环境
- 激活虚拟环境
- `pip install paroquant[...]`

## 快速开始

### 1. 检查环境

```bash
uv run qwen3-server-manager doctor
```

### 2. 一键启动

```bash
uv run qwen3-server-manager start \
  --model z-lab/Qwen3.5-4B-PARO \
  --port 8288
```

启动时会自动：

- 检查 `uv`
- 创建运行时虚拟环境
- 根据平台选择 `paroquant[mlx]` 或 `paroquant[vllm]`
- 安装缺失依赖
- 启动 OpenAI-Compatible API Server

### 3. 查看状态

```bash
uv run qwen3-server-manager status
```

### 4. 查看日志

```bash
uv run qwen3-server-manager logs --tail 80
```

### 5. 停止服务

```bash
uv run qwen3-server-manager stop
```

## 配置文件

默认配置文件路径为 `config/config.json`：

```json
{
  "model": "z-lab/Qwen3.5-4B-PARO",
  "host": "127.0.0.1",
  "port": 8288,
  "backend": "mlx",
  "extra_args": []
}
```

使用方式：

```bash
uv run qwen3-server-manager start
```

命令行参数优先级高于默认配置文件。

也就是说，你把 `config/config.json` 改好之后，直接：

```bash
uv run qwen3-server-manager start
```

就会默认按该配置在后台启动。

## 图片请求示例

服务启动后，可以直接走 OpenAI-Compatible 的 `/v1/chat/completions`。

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

如果你本地图片很多，更推荐把文件转成 `data:image/...;base64,...` 再发给接口。

## macOS 服务化

如果你希望它作为真正的后台服务长期运行，推荐使用 `launchd`：

```bash
uv run qwen3-server-manager install-service
uv run qwen3-server-manager service-status
uv run qwen3-server-manager stop-service
uv run qwen3-server-manager start-service
uv run qwen3-server-manager restart-service
uv run qwen3-server-manager uninstall-service
```

默认服务标签是 `local.qwen3-server-manager`，默认读取 `config/config.json`。
`install-service` 会生成并安装 `~/Library/LaunchAgents/local.qwen3-server-manager.plist`。

如果你希望它长期常驻在本机并随登录启动，推荐这一套：

```bash
uv run qwen3-server-manager install-service
uv run qwen3-server-manager restart-service
uv run qwen3-server-manager service-status
```

## 运行目录

默认运行时目录为：

```text
.runtime/qwen3-server-manager/
```

其中包含：

- `state.json`: 当前实例状态
- `server.log`: 服务标准输出/错误日志
- `venv/`: 由 `uv` 自动创建和维护的运行时环境

## Agent 定位

`paroquant.cli.agent` 已在当前环境完成真实验证，能够通过 MCP 成功调用：

- `time`
- `filesystem`
- `fetch`

但从架构定位上，它更适合作为：

- 官方示例 agent
- 本地 tool-calling 冒烟测试入口
- 调试模型工具调用能力的参考实现

而不是必须绑定的主运行时。

如果你已经有自己的 agent runtime（例如自定义 orchestration、workflow engine、memory runtime 或其他 agent 框架），更推荐直接对接本管理器启动的 OpenAI-Compatible API Server，而不是依赖 `paroquant.cli.agent` 本身。

## 说明

- 默认不依赖项目已有 `.venv`，而是由管理器通过 `uv` 自己维护运行时环境
- 如果依赖已存在，不会重复安装
- 若服务已存在且 PID 仍存活，重复 `start` 会直接拒绝，避免多开
- `paroquant.cli.serve` 的后端选择由其自身与环境决定，管理器只负责准备正确依赖
