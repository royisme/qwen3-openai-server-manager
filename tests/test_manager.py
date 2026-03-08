from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen3_server_manager.manager import DEFAULT_SERVICE_LABEL, ServerConfig, ServerManager, ServerState


class ManagerTests(unittest.TestCase):
    def test_build_command_includes_required_arguments(self) -> None:
        manager = ServerManager(tempfile.mkdtemp())
        config = ServerConfig(
            model='z-lab/Qwen3.5-4B-PARO',
            host='0.0.0.0',
            port=9000,
            backend='vllm',
            extra_args=['--max-model-len', '8192'],
        )
        command = manager.build_command(config, 'python3')
        self.assertEqual(
            command,
            [
                'python3',
                '-m',
                'qwen3_server_manager.paro_serve',
                '--model',
                'z-lab/Qwen3.5-4B-PARO',
                '--host',
                '0.0.0.0',
                '--port',
                '9000',
                '--max-model-len',
                '8192',
            ],
        )

    def test_config_merges_json_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.json'
            config_path.write_text(json.dumps({'model': 'from-file', 'port': 8001}), encoding='utf-8')
            config = ServerConfig.from_sources(str(config_path), {'model': 'from-cli', 'port': None})
            self.assertEqual(config.model, 'from-cli')
            self.assertEqual(config.port, 8001)

    def test_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ServerManager(tmpdir)
            state = ServerState(
                pid=123,
                started_at=1.0,
                command=['python3', '-m', 'qwen3_server_manager.paro_serve'],
                config={'model': 'demo'},
                log_path='/tmp/server.log',
            )
            manager.save_state(state)
            self.assertEqual(manager.load_state(), state)

    def test_resolve_backend_prefers_apple_mlx(self) -> None:
        manager = ServerManager(tempfile.mkdtemp())
        original = manager.platform_info
        manager.platform_info = lambda: {'system': 'Darwin', 'machine': 'arm64'}
        try:
            self.assertEqual(manager.resolve_backend('auto'), 'mlx')
        finally:
            manager.platform_info = original

    def test_resolve_backend_respects_explicit_choice(self) -> None:
        manager = ServerManager(tempfile.mkdtemp())
        self.assertEqual(manager.resolve_backend('vllm'), 'vllm')

    def test_start_status_stop_with_fake_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ServerManager(tmpdir)
            config = ServerConfig(
                model='fake-model',
                host='127.0.0.1',
                port=18080,
                python_executable='python3',
                module='tests.fake_server',
            )
            state = manager.start(config, wait_seconds=0.5, install=False)
            try:
                self.assertTrue(manager.is_running(state.pid))
                status = manager.status()
                self.assertTrue(status['running'])
                self.assertTrue(status['health']['ok'])
            finally:
                self.assertTrue(manager.stop(timeout=2.0))
                self.assertFalse(manager.load_state())

    def test_build_service_plist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ServerManager(tmpdir)
            config = ServerConfig(model='demo', port=8288, backend='mlx', python_executable='python3')
            original = manager.resolve_backend
            original_ensure = manager.ensure_dependencies
            manager.resolve_backend = lambda preferred='auto': 'mlx'
            manager.ensure_dependencies = lambda backend, python_executable=None: 'python3'
            try:
                plist = manager.build_service_plist(config, label=DEFAULT_SERVICE_LABEL)
            finally:
                manager.resolve_backend = original
                manager.ensure_dependencies = original_ensure
            self.assertEqual(plist['Label'], DEFAULT_SERVICE_LABEL)
            self.assertIn('--port', plist['ProgramArguments'])
            self.assertIn('python3', plist['ProgramArguments'])
            self.assertTrue(plist['RunAtLoad'])
            self.assertTrue(plist['KeepAlive'])

    def test_install_service_writes_plist_and_bootstraps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ServerManager(tmpdir)
            config = ServerConfig(model='demo', backend='mlx', python_executable='python3')
            calls = []
            original_run = manager.run_command
            original_build = manager.build_service_plist
            manager.run_command = lambda cmd, check=True: calls.append(cmd) or type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
            manager.build_service_plist = lambda *args, **kwargs: {'Label': DEFAULT_SERVICE_LABEL, 'ProgramArguments': ['python3'], 'RunAtLoad': True, 'KeepAlive': True, 'StandardOutPath': '/tmp/out', 'StandardErrorPath': '/tmp/err'}
            try:
                result = manager.install_service(config, launch_agents_dir=tmpdir)
            finally:
                manager.run_command = original_run
                manager.build_service_plist = original_build
            self.assertTrue(Path(result['plist_path']).exists())
            self.assertEqual(calls[1][0:2], ['launchctl', 'bootstrap'])
