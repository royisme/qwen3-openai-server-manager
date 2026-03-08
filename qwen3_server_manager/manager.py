from __future__ import annotations

import json
import os
import platform
import plistlib
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_RUNTIME_DIR = Path('.runtime/qwen3-server-manager')
DEFAULT_CONFIG_PATH = Path('config/config.json')
DEFAULT_SERVICE_LABEL = 'local.qwen3-server-manager'


@dataclass
class ServerConfig:
    model: str
    host: str = '127.0.0.1'
    port: int = 8000
    backend: str = 'auto'
    module: str = 'qwen3_server_manager.paro_serve'
    extra_args: list[str] | None = None
    python_executable: str | None = None

    @classmethod
    def from_sources(cls, config_path: str | None, cli_values: dict[str, Any]) -> 'ServerConfig':
        data: dict[str, Any] = {}
        resolved_config = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if resolved_config.exists():
            data = json.loads(resolved_config.read_text(encoding='utf-8'))
        merged = {**data, **{key: value for key, value in cli_values.items() if value is not None}}
        merged.setdefault('extra_args', data.get('extra_args', []))
        return cls(**merged)


@dataclass
class ServerState:
    pid: int
    started_at: float
    command: list[str]
    config: dict[str, Any]
    log_path: str


class ServerManager:
    def __init__(self, runtime_dir: str | Path | None = None):
        self.runtime_dir = Path(runtime_dir or DEFAULT_RUNTIME_DIR)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime_dir / 'state.json'
        self.log_path = self.runtime_dir / 'server.log'
        self.venv_dir = self.runtime_dir / 'venv'
        self.launchd_dir = self.runtime_dir / 'launchd'
        self.launchd_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> ServerState | None:
        if not self.state_path.exists():
            return None
        return ServerState(**json.loads(self.state_path.read_text(encoding='utf-8')))

    def save_state(self, state: ServerState) -> None:
        self.state_path.write_text(json.dumps(asdict(state), indent=2), encoding='utf-8')

    def clear_state(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()

    def is_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def platform_info(self) -> dict[str, str]:
        return {'system': platform.system(), 'machine': platform.machine().lower()}

    def find_uv(self) -> str | None:
        return shutil.which('uv')

    def venv_python(self) -> str:
        if os.name == 'nt':
            return str(self.venv_dir / 'Scripts' / 'python.exe')
        return str(self.venv_dir / 'bin' / 'python')

    def resolve_backend(self, preferred: str = 'auto') -> str:
        if preferred != 'auto':
            return preferred
        info = self.platform_info()
        if info['system'] == 'Darwin' and info['machine'] in {'arm64', 'aarch64'}:
            return 'mlx'
        if info['system'] == 'Linux' and shutil.which('nvidia-smi'):
            return 'vllm'
        raise RuntimeError('Cannot infer backend automatically. Use --backend mlx on Apple Silicon or --backend vllm on NVIDIA Linux.')

    def package_spec(self, backend: str) -> str:
        if backend == 'mlx':
            return 'paroquant[mlx]'
        if backend == 'vllm':
            return 'paroquant[vllm]'
        raise RuntimeError(f'Unsupported backend: {backend}')

    def run_command(self, command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, check=check, text=True, capture_output=True)

    def ensure_uv(self) -> str:
        uv = self.find_uv()
        if not uv:
            raise RuntimeError('uv is required but was not found in PATH. Please install uv first: https://docs.astral.sh/uv/')
        return uv

    def ensure_venv(self) -> str:
        uv = self.ensure_uv()
        python_executable = self.venv_python()
        if Path(python_executable).exists():
            return python_executable
        self.run_command([uv, 'venv', str(self.venv_dir)])
        return python_executable

    def python_can_import(self, python_executable: str, module: str) -> bool:
        process = subprocess.run(
            [python_executable, '-c', f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module}') else 1)"],
            text=True,
            capture_output=True,
        )
        return process.returncode == 0

    def ensure_dependencies(self, backend: str, python_executable: str | None = None) -> str:
        uv = self.ensure_uv()
        target_python = python_executable or self.ensure_venv()
        required_modules = ['paroquant']
        if backend == 'mlx':
            required_modules.extend(['mlx', 'mlx_lm'])
        elif backend == 'vllm':
            required_modules.append('vllm')
        if all(self.python_can_import(target_python, module) for module in required_modules):
            return target_python
        self.run_command([uv, 'pip', 'install', '--python', target_python, self.package_spec(backend)])
        return target_python

    def doctor(self, preferred_backend: str = 'auto') -> dict[str, Any]:
        info = self.platform_info()
        uv = self.find_uv()
        result: dict[str, Any] = {
            'uv': {'found': bool(uv), 'path': uv},
            'platform': info,
            'runtime_dir': str(self.runtime_dir),
            'managed_python': self.venv_python(),
            'default_config_path': str(DEFAULT_CONFIG_PATH),
        }
        try:
            backend = self.resolve_backend(preferred_backend)
            result['backend'] = {'requested': preferred_backend, 'resolved': backend, 'package': self.package_spec(backend)}
        except Exception as exc:  # noqa: BLE001
            result['backend'] = {'requested': preferred_backend, 'error': str(exc)}
            return result
        managed_python = self.venv_python()
        result['managed_env_exists'] = Path(managed_python).exists()
        if Path(managed_python).exists():
            result['managed_imports'] = {
                'paroquant': self.python_can_import(managed_python, 'paroquant'),
                backend: self.python_can_import(managed_python, backend if backend == 'vllm' else 'mlx_lm'),
            }
        return result

    def build_command(self, config: ServerConfig, python_executable: str) -> list[str]:
        command = [
            python_executable,
            '-m',
            config.module,
            '--model',
            config.model,
            '--host',
            config.host,
            '--port',
            str(config.port),
        ]
        if config.extra_args:
            command.extend(config.extra_args)
        return command

    def start(self, config: ServerConfig, wait_seconds: float = 3.0, install: bool = True) -> ServerState:
        state = self.load_state()
        if state and self.is_running(state.pid):
            raise RuntimeError(f'Server already running with pid={state.pid}')
        if state and not self.is_running(state.pid):
            self.clear_state()

        resolved_backend = self.resolve_backend(config.backend)
        if config.python_executable:
            python_executable = config.python_executable
        elif install:
            python_executable = self.ensure_dependencies(resolved_backend)
        else:
            python_executable = self.ensure_venv()
        command = self.build_command(config, python_executable)

        env = os.environ.copy()
        repo_root = str(Path.cwd())
        existing_pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = repo_root if not existing_pythonpath else repo_root + os.pathsep + existing_pythonpath

        with open(self.log_path, 'ab') as log_handle:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )

        state = ServerState(
            pid=process.pid,
            started_at=time.time(),
            command=command,
            config={**asdict(config), 'backend': resolved_backend, 'python_executable': python_executable},
            log_path=str(self.log_path),
        )
        self.save_state(state)
        time.sleep(wait_seconds)
        if not self.is_running(process.pid):
            self.clear_state()
            raise RuntimeError(f'Server exited immediately. Check logs at {self.log_path}')
        return state

    def stop(self, timeout: float = 10.0) -> bool:
        state = self.load_state()
        if not state:
            return False
        if not self.is_running(state.pid):
            self.clear_state()
            return False

        def _send(sig: int) -> None:
            try:
                os.killpg(state.pid, sig)
                return
            except (PermissionError, ProcessLookupError):
                pass
            try:
                os.kill(state.pid, sig)
            except ProcessLookupError:
                return

        _send(signal.SIGTERM)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_running(state.pid):
                self.clear_state()
                return True
            time.sleep(0.2)
        _send(signal.SIGKILL)
        self.clear_state()
        return True

    def status(self) -> dict[str, Any]:
        state = self.load_state()
        if not state:
            return {'running': False, 'reason': 'no state file'}
        running = self.is_running(state.pid)
        health = self.health(state.config['host'], int(state.config['port'])) if running else None
        return {
            'running': running,
            'pid': state.pid,
            'started_at': state.started_at,
            'command': state.command,
            'log_path': state.log_path,
            'config': state.config,
            'health': health,
        }

    def health(self, host: str, port: int, timeout: float = 2.0) -> dict[str, Any]:
        base = f'http://{host}:{port}'
        last_error = 'unknown'
        for path in ('/health', '/v1/models'):
            url = base + path
            try:
                with urlopen(url, timeout=timeout) as response:
                    body = response.read(512).decode('utf-8', errors='replace')
                    return {'ok': True, 'url': url, 'status': response.status, 'body_preview': body[:200]}
            except URLError as exc:
                last_error = str(exc)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        return {'ok': False, 'error': last_error, 'checked': [base + '/health', base + '/v1/models']}

    def read_logs(self, tail: int = 80) -> str:
        if not self.log_path.exists():
            return ''
        lines = self.log_path.read_text(encoding='utf-8', errors='replace').splitlines()
        return '\n'.join(lines if tail <= 0 else lines[-tail:])

    def launch_agents_dir(self, custom_dir: str | None = None) -> Path:
        return Path(custom_dir) if custom_dir else Path.home() / 'Library' / 'LaunchAgents'

    def service_plist_path(self, label: str = DEFAULT_SERVICE_LABEL, launch_agents_dir: str | None = None) -> Path:
        return self.launch_agents_dir(launch_agents_dir) / f'{label}.plist'

    def service_target(self, label: str = DEFAULT_SERVICE_LABEL) -> str:
        return f'gui/{os.getuid()}/{label}'

    def service_logs(self, label: str = DEFAULT_SERVICE_LABEL) -> tuple[Path, Path]:
        safe = label.replace('/', '-').replace(':', '-')
        out_log = self.launchd_dir / f'{safe}.out.log'
        err_log = self.launchd_dir / f'{safe}.err.log'
        out_log.parent.mkdir(parents=True, exist_ok=True)
        return out_log, err_log

    def build_service_plist(
        self,
        config: ServerConfig,
        label: str = DEFAULT_SERVICE_LABEL,
        keep_alive: bool = True,
        run_at_load: bool = True,
    ) -> dict[str, Any]:
        resolved_backend = self.resolve_backend(config.backend)
        python_executable = config.python_executable or self.ensure_dependencies(resolved_backend)
        program_arguments = self.build_command(config, python_executable)
        out_log, err_log = self.service_logs(label)
        return {
            'Label': label,
            'ProgramArguments': program_arguments,
            'WorkingDirectory': str(Path.cwd()),
            'RunAtLoad': run_at_load,
            'KeepAlive': keep_alive,
            'StandardOutPath': str(out_log),
            'StandardErrorPath': str(err_log),
            'EnvironmentVariables': {
                'PATH': os.environ.get('PATH', ''),
                'HOME': str(Path.home()),
                'PYTHONPATH': str(Path.cwd()),
            },
        }

    def install_service(
        self,
        config: ServerConfig,
        label: str = DEFAULT_SERVICE_LABEL,
        launch_agents_dir: str | None = None,
        keep_alive: bool = True,
        run_at_load: bool = True,
    ) -> dict[str, Any]:
        if platform.system() != 'Darwin':
            raise RuntimeError('launchd service management is only supported on macOS')
        plist_path = self.service_plist_path(label, launch_agents_dir)
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist = self.build_service_plist(config, label=label, keep_alive=keep_alive, run_at_load=run_at_load)
        plist_path.write_bytes(plistlib.dumps(plist))
        domain = f'gui/{os.getuid()}'
        self.run_command(['launchctl', 'bootout', self.service_target(label)], check=False)
        self.run_command(['launchctl', 'bootstrap', domain, str(plist_path)])
        if run_at_load:
            self.run_command(['launchctl', 'kickstart', '-k', self.service_target(label)], check=False)
        return {'ok': True, 'label': label, 'plist_path': str(plist_path), 'target': self.service_target(label)}

    def uninstall_service(self, label: str = DEFAULT_SERVICE_LABEL, launch_agents_dir: str | None = None) -> dict[str, Any]:
        if platform.system() != 'Darwin':
            raise RuntimeError('launchd service management is only supported on macOS')
        plist_path = self.service_plist_path(label, launch_agents_dir)
        self.run_command(['launchctl', 'bootout', self.service_target(label)], check=False)
        if plist_path.exists():
            plist_path.unlink()
        return {'ok': True, 'label': label, 'plist_path': str(plist_path)}

    def start_service(self, label: str = DEFAULT_SERVICE_LABEL, launch_agents_dir: str | None = None) -> dict[str, Any]:
        if platform.system() != 'Darwin':
            raise RuntimeError('launchd service management is only supported on macOS')
        plist_path = self.service_plist_path(label, launch_agents_dir)
        if plist_path.exists():
            self.run_command(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(plist_path)], check=False)
        self.run_command(['launchctl', 'kickstart', '-k', self.service_target(label)])
        return {'ok': True, 'label': label, 'target': self.service_target(label)}

    def stop_service(self, label: str = DEFAULT_SERVICE_LABEL) -> dict[str, Any]:
        if platform.system() != 'Darwin':
            raise RuntimeError('launchd service management is only supported on macOS')
        self.run_command(['launchctl', 'bootout', self.service_target(label)], check=False)
        return {'ok': True, 'label': label, 'target': self.service_target(label)}

    def restart_service(self, label: str = DEFAULT_SERVICE_LABEL, launch_agents_dir: str | None = None) -> dict[str, Any]:
        self.stop_service(label)
        return self.start_service(label, launch_agents_dir)

    def service_status(self, label: str = DEFAULT_SERVICE_LABEL) -> dict[str, Any]:
        if platform.system() != 'Darwin':
            raise RuntimeError('launchd service management is only supported on macOS')
        result = self.run_command(['launchctl', 'print', self.service_target(label)], check=False)
        return {
            'ok': result.returncode == 0,
            'label': label,
            'target': self.service_target(label),
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
        }
