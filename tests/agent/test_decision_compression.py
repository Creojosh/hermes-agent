"""Real HTTP contracts for selective compression and profile isolation."""
import copy
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor, _PRUNED_TOOL_PLACEHOLDER
from agent.conversation_compression import _run_summary_dispatch
from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


@contextmanager
def server(behavior='ok'):
    calls = []
    entered, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # Deadline/cancellation deliberately disconnects the client.

        def do_GET(self):
            calls.append((self.path, self.headers.get('Authorization'), None))
            if self.path == '/v1/models':
                self.reply(200, {'data': [{'id': 'loaded-model', 'status': {
                    'value': 'unloaded' if behavior == 'unloaded' else 'loaded'}}]})
                return
            self.reply(200, {'build_info': 'test-llama', 'model_path': 'fixture.gguf'})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, self.headers.get('Authorization'), body))
            if self.headers.get('Authorization') == 'Bearer B' or behavior == 'unsupported':
                self.reply(400, {'error': 'decisions are disabled: start the server with --decision-seqs N'})
                return
            if behavior == 'slow':
                entered.set()
                release.wait(10)
            candidates = json.loads(body['contexts'][0])['candidates']
            fields = {name: {'value': False, 'probability': 0.99} for name in body['schema']}
            for slot, candidate in candidates.items():
                if candidate['tool'] == 'keep':
                    fields[f'retain_{slot}']['value'] = True
                if candidate['tool'] == 'uncertain':
                    fields[f'retain_{slot}']['probability'] = 0.89
            result = {'fields': fields, 'decision': {name: field['value'] for name, field in fields.items()}}
            if behavior == 'malformed' and len([c for c in calls if c[2]]) > 1:
                del fields['retain_0']
            self.reply(200, {'object': 'decision', 'results': [result]})

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}/v1', calls, entered, release
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(10)


def transcript():
    rows = [{'role': 'system', 'content': 'Stable system prompt'},
            {'role': 'user', 'content': 'Investigate the build and keep the final diagnosis.'}]
    for i in range(23):
        tool = {1: 'keep', 2: 'uncertain', 3: 'skill_view'}.get(i, 'terminal')
        rows.extend([
            {'role': 'assistant', 'tool_calls': [{'id': 'reused', 'type': 'function',
                'function': {'name': tool, 'arguments': '{}'}}], 'content': ''},
            {'role': 'tool', 'tool_call_id': 'reused',
             'content': f'output {i}: ' + ('useful instructions ' * 30 if tool != 'terminal' else 'old log line ' * 1600)},
        ])
    rows.extend([{'role': 'assistant', 'content': 'Ready to check the fix.'},
                 {'role': 'user', 'content': 'Verify that the build now succeeds.'}])
    return rows


def agent_for(url, key='A'):
    comp = ContextCompressor(model='loaded-model', config_context_length=65536, quiet_mode=True,
                             protect_first_n=1, protect_last_n=3)
    comp.tail_token_budget = 500
    return SimpleNamespace(context_compressor=comp, provider='custom', model='loaded-model',
                           base_url=url, api_key=key, client=None, session_id='fixture')


def dispatch(agent, rows, *, extra=None, cancel=None, fence=None):
    fallbacks = []

    def fallback(messages, **kwargs):
        fallbacks.append(copy.deepcopy(messages))
        return messages

    result = _run_summary_dispatch(agent, rows, fallback, extra or {}, commit_fence=fence,
                                   attempt_generation=None, hard_cancel_event=cancel)
    return result, fallbacks


@pytest.mark.parametrize('behavior', ['ok', 'malformed', 'unsupported', 'unloaded', 'slow', 'cancel', 'fence', 'insufficient', 'manual'])
def test_selection_is_atomic_and_preserves_conversation(tmp_path, behavior):
    rows = transcript()
    original = copy.deepcopy(rows)
    with server('slow' if behavior in ('cancel', 'fence') else behavior) as (url, calls, entered, release):
        agent = agent_for(url)
        cancel = threading.Event()
        fence = None
        if behavior == 'insufficient':
            agent.context_compressor.threshold_tokens = 100
        if behavior in ('cancel', 'fence'):
            if behavior == 'fence':
                from agent.conversation_compression import CompressionCommitFence
                fence = CompressionCommitFence()
            def stop():
                if entered.wait(10):
                    if fence is not None:
                        fence.cancel_before_commit()
                    else:
                        cancel.set()
            stopper = threading.Thread(target=stop, daemon=True)
            stopper.start()
            with pytest.raises(AuxiliaryExplicitCancellation):
                dispatch(agent, rows, cancel=cancel, fence=fence)
            release.set()
            stopper.join(10)
            assert rows == original
            assert agent.context_compressor.compression_count == 0
            return
        result, fallbacks = dispatch(agent, rows, cancel=cancel, fence=fence,
                                     extra={'focus_topic': 'build'} if behavior == 'manual' else None)
        assert rows == original
        if behavior != 'ok':
            assert fallbacks == [original]
            assert result == original
            assert agent.context_compressor.compression_count == 0
            if behavior in ('unloaded', 'insufficient', 'manual'):
                assert not any(body for _, _, body in calls)
            return
        assert not fallbacks
        assert agent.context_compressor.compression_count == 1
        assert len(result) == len(rows)
        changes = [i for i in range(len(rows)) if result[i] != rows[i]]
        assert changes
        for i in changes:
            assert rows[i]['role'] == 'tool'
            assert result[i] == {**rows[i], 'content': _PRUNED_TOOL_PLACEHOLDER}
        for i, row in enumerate(rows):
            if row.get('role') != 'tool' or i in (5, 7, 9) or i >= len(rows) - 4:
                assert result[i] == row
        posts = [c[2] for c in calls if c[2]]
        assert len(posts) >= 2
        assert all(p['model'] == agent.model and p['mode'] == 'tree' and p['cache_prompt'] for p in posts)
        assert all(p['schema'] == posts[0]['schema'] and p['instructions'] == posts[0]['instructions'] for p in posts)


def test_profile_scope_and_capability_cache_use_real_config_and_http(tmp_path, monkeypatch):
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    homes = {name: tmp_path / name for name in ('A', 'B', 'off')}
    for name, home in homes.items():
        home.mkdir()
        (home / 'config.yaml').write_text('compression:\n  decision:\n    mode: ' + ('off' if name == 'off' else 'auto') + '\n')
    with server() as (url, calls, _, __):
        agents = {name: agent_for(url, name) for name in homes}
        set_multiplex_active(True)
        try:
            for name in ('A', 'B', 'A', 'B', 'off'):
                home_token = set_hermes_home_override(homes[name])
                secret_token = set_secret_scope({})
                try:
                    _, fallbacks = dispatch(agents[name], transcript())
                    assert bool(fallbacks) == (name != 'A')
                finally:
                    reset_secret_scope(secret_token)
                    reset_hermes_home_override(home_token)
        finally:
            set_multiplex_active(False)
        auths = [auth for _, auth, body in calls if body]
        assert auths.count('Bearer B') == 1
        assert 'Bearer off' not in auths
        assert auths[0] == auths[-1] == 'Bearer A'
        assert len([c for c in calls if c[0] == '/props' and c[1] == 'Bearer A']) == 1
