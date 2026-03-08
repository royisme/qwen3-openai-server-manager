from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_server_manager.manager import DEFAULT_CONFIG_PATH, DEFAULT_SERVICE_LABEL, ServerConfig, ServerManager


def add_shared_start_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument('--model')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    parser.add_argument('--backend', choices=['auto', 'vllm', 'mlx'], default='auto')
    parser.add_argument('--python-executable')
    parser.add_argument('--module')
    parser.add_argument('--extra-arg', action='append', dest='extra_args')
    parser.add_argument('--wait-seconds', type=float, default=3.0)
    parser.add_argument('--skip-install', action='store_true')


def add_service_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--label', default=DEFAULT_SERVICE_LABEL)
    parser.add_argument('--launch-agents-dir')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Manage local ParoQuant OpenAI-compatible API server')
    parser.add_argument('--runtime-dir', default='/.invalid')
    parser.set_defaults(runtime_dir=None)
    subparsers = parser.add_subparsers(dest='command', required=True)

    start_parser = subparsers.add_parser('start')
    add_shared_start_options(start_parser)

    stop_parser = subparsers.add_parser('stop')
    stop_parser.add_argument('--timeout', type=float, default=10.0)

    restart_parser = subparsers.add_parser('restart')
    add_shared_start_options(restart_parser)
    restart_parser.add_argument('--timeout', type=float, default=10.0)

    subparsers.add_parser('status')

    health_parser = subparsers.add_parser('health')
    health_parser.add_argument('--host')
    health_parser.add_argument('--port', type=int)

    logs_parser = subparsers.add_parser('logs')
    logs_parser.add_argument('--tail', type=int, default=80)

    doctor_parser = subparsers.add_parser('doctor')
    doctor_parser.add_argument('--backend', choices=['auto', 'vllm', 'mlx'], default='auto')

    install_service_parser = subparsers.add_parser('install-service')
    add_shared_start_options(install_service_parser)
    add_service_options(install_service_parser)
    install_service_parser.add_argument('--keep-alive', action=argparse.BooleanOptionalAction, default=True)
    install_service_parser.add_argument('--run-at-load', action=argparse.BooleanOptionalAction, default=True)

    uninstall_service_parser = subparsers.add_parser('uninstall-service')
    add_service_options(uninstall_service_parser)

    start_service_parser = subparsers.add_parser('start-service')
    add_service_options(start_service_parser)

    stop_service_parser = subparsers.add_parser('stop-service')
    add_service_options(stop_service_parser)

    restart_service_parser = subparsers.add_parser('restart-service')
    add_service_options(restart_service_parser)

    service_status_parser = subparsers.add_parser('service-status')
    add_service_options(service_status_parser)

    return parser


def print_json(data: object) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_config(args: argparse.Namespace) -> ServerConfig:
    cli_values = {
        'model': getattr(args, 'model', None),
        'host': getattr(args, 'host', None),
        'port': getattr(args, 'port', None),
        'backend': getattr(args, 'backend', None),
        'python_executable': getattr(args, 'python_executable', None),
        'module': getattr(args, 'module', None),
        'extra_args': getattr(args, 'extra_args', None),
    }
    return ServerConfig.from_sources(getattr(args, 'config', None), cli_values)


def ensure_config_or_model(parser: argparse.ArgumentParser, args: argparse.Namespace, verb: str) -> None:
    if not getattr(args, 'model', None) and not Path(args.config).exists():
        parser.error(f'{verb} requires --model or an existing config file at {args.config}')


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manager = ServerManager(None if args.runtime_dir == '/.invalid' else args.runtime_dir)

    if args.command == 'start':
        ensure_config_or_model(parser, args, 'start')
        state = manager.start(build_config(args), wait_seconds=args.wait_seconds, install=not args.skip_install)
        print_json({'ok': True, 'message': 'server started', 'state': state.__dict__})
        return

    if args.command == 'stop':
        stopped = manager.stop(timeout=args.timeout)
        print_json({'ok': stopped, 'message': 'server stopped' if stopped else 'server not running'})
        return

    if args.command == 'restart':
        ensure_config_or_model(parser, args, 'restart')
        manager.stop(timeout=args.timeout)
        state = manager.start(build_config(args), wait_seconds=args.wait_seconds, install=not args.skip_install)
        print_json({'ok': True, 'message': 'server restarted', 'state': state.__dict__})
        return

    if args.command == 'status':
        print_json(manager.status())
        return

    if args.command == 'health':
        status = manager.status()
        fallback_config = ServerConfig.from_sources(str(DEFAULT_CONFIG_PATH), {}) if DEFAULT_CONFIG_PATH.exists() else None
        host = args.host or status.get('config', {}).get('host') or getattr(fallback_config, 'host', '127.0.0.1')
        port = args.port or status.get('config', {}).get('port') or getattr(fallback_config, 'port', 8000)
        print_json(manager.health(host, int(port)))
        return

    if args.command == 'logs':
        print(manager.read_logs(tail=args.tail))
        return

    if args.command == 'doctor':
        print_json(manager.doctor(args.backend))
        return

    if args.command == 'install-service':
        ensure_config_or_model(parser, args, 'install-service')
        print_json(
            manager.install_service(
                build_config(args),
                label=args.label,
                launch_agents_dir=args.launch_agents_dir,
                keep_alive=args.keep_alive,
                run_at_load=args.run_at_load,
            )
        )
        return

    if args.command == 'uninstall-service':
        print_json(manager.uninstall_service(label=args.label, launch_agents_dir=args.launch_agents_dir))
        return

    if args.command == 'start-service':
        print_json(manager.start_service(label=args.label, launch_agents_dir=args.launch_agents_dir))
        return

    if args.command == 'stop-service':
        print_json(manager.stop_service(label=args.label))
        return

    if args.command == 'restart-service':
        print_json(manager.restart_service(label=args.label, launch_agents_dir=args.launch_agents_dir))
        return

    if args.command == 'service-status':
        print_json(manager.service_status(label=args.label))
        return


if __name__ == '__main__':
    main()
