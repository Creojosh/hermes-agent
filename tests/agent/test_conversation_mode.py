"""Real transport and durable mode transitions through the actual agent turn loop."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from agent.conversation_mode import mode_state, request_mode
from hermes_state import SessionDB
from run_agent import AIAgent
from types import SimpleNamespace


@contextmanager
def endpoint(unsupported=False, entered=None, release=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, {'data': [{'id': 'loaded-model'}]} if self.path.endswith('models')
                       else {'build_info': 'llama-test'})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, self.headers.get('Authorization'), body))
            if self.path.endswith('/decision'):
                if entered is not None:
                    entered.set()
                    release.wait(8)
                if unsupported:
                    self.reply(404, {})
                    return
                value = 'inspect' in json.loads(body['contexts'][0])['request']
                self.reply(200, {'object': 'decision', 'results': [{
                    'fields': {'needs_agent': {'value': value, 'probability': .99}},
                    'decision': {'needs_agent': value}}]})
                return
            if body.get('stream'):
                delta = {'role': 'assistant', 'content': 'A short answer.'}
                fabricated = body['messages'][-1]['content'] == 'pretend'
                if fabricated:
                    delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'fabricated',
                        'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}]}
                chunk = {'id': 'answer', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': 'loaded-model', 'choices': [{'index': 0,
                         'finish_reason': 'tool_calls' if fabricated else 'stop', 'delta': delta}]}
                payload = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.reply(200, {'id': 'answer', 'object': 'chat.completion', 'created': 1,
                'model': 'loaded-model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'A short answer.'}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 4, 'total_tokens': 104}})

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


@pytest.mark.parametrize('unsupported', [False, True])
def test_chat_promotion_resume_and_explicit_boundary(tmp_path, unsupported):
    with endpoint(unsupported) as (url, calls), SessionDB(tmp_path / 'sessions.db') as db:
        def make():
            return AIAgent(base_url=url, api_key='fixture', provider='custom', model='loaded-model',
                platform='cli', session_id='same-session', session_db=db, enabled_toolsets=['terminal'],
                quiet_mode=True, skip_context_files=True, skip_memory=True, skip_background_review=True,
                max_iterations=2)

        agent = make()
        result = agent.run_conversation('Explain a rainbow.')
        assert result['final_response'] == 'A short answer.'
        assert mode_state(agent)['active'] == ('agent' if unsupported else 'chat')
        chats = [b for p, _, b in calls if p.endswith('/chat/completions')]
        assert bool(chats[-1].get('tools')) == unsupported
        prefix = chats[-1]['messages'][0]['content']
        result = agent.run_conversation('Explain it simply.', conversation_history=result['messages'])
        chats = [b for p, _, b in calls if p.endswith('/chat/completions')]
        assert chats[-1]['messages'][0]['content'] == prefix
        result = agent.run_conversation('inspect the project', conversation_history=result['messages'])
        assert mode_state(agent)['active'] == 'agent'
        decisions = sum(p.endswith('/decision') for p, _, _ in calls)
        agent = make()
        result = agent.run_conversation('Hello again.', conversation_history=result['messages'])
        assert mode_state(agent)['active'] == 'agent'
        assert sum(p.endswith('/decision') for p, _, _ in calls) == decisions
        old_prompt = agent._cached_system_prompt
        request_mode(agent, 'chat')
        assert agent._cached_system_prompt == old_prompt
        result = agent.run_conversation('Translate hello.', conversation_history=result['messages'])
        assert mode_state(agent)['active'] == 'chat'
        assert agent.session_id == 'same-session'
        assert db.get_session_model_config_value('same-session', 'conversation_mode')['active'] == 'chat'
        chats = [b for p, _, b in calls if p.endswith('/chat/completions')]
        assert not chats[-1].get('tools')
        assert all(key == 'Bearer fixture' for _, key, _ in calls)
        users = [m['content'] for m in result['messages'] if m['role'] == 'user']
        assert users == ['Explain a rainbow.', 'Explain it simply.', 'inspect the project',
                         'Hello again.', 'Translate hello.']
        result = agent.run_conversation('pretend', conversation_history=result['messages'])
        assert 'cannot execute tools' in result['final_response']
        assert result['messages'][-1]['role'] == 'assistant'
        assert not any(m.get('tool_calls') or m['role'] == 'tool' for m in result['messages'])


def test_profile_defaults_and_cancelled_decision_never_commit(tmp_path, monkeypatch):
    from agent.conversation_mode import prepare_mode_turn, persist_mode_boundary
    from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    homes = {name: tmp_path / name for name in ('A', 'B')}
    for name, home in homes.items():
        home.mkdir()
        (home / 'config.yaml').write_text('agent:\n  conversation_mode: ' + ('auto' if name == 'A' else 'agent') + '\n')
    with endpoint() as (url, calls):
        set_multiplex_active(True)
        try:
            for index, name in enumerate(('A', 'B', 'A')):
                token = set_hermes_home_override(homes[name])
                secret = set_secret_scope({})
                try:
                    with SessionDB(homes[name] / 'sessions.db') as db:
                        db.create_session(str(index), 'cli')
                        agent = SimpleNamespace(provider='custom', platform='cli', base_url=url,
                            model='loaded-model', api_key=name, client=None, _session_db=db,
                            session_id=str(index), tools=[], valid_tool_names=set(),
                            _cached_system_prompt='original', _build_system_prompt=lambda _: 'chat prompt')
                        prepare_mode_turn(agent, 'Hello', [])
                        persist_mode_boundary(agent)
                        assert mode_state(agent)['active'] == ('chat' if name == 'A' else 'agent')
                finally:
                    reset_secret_scope(secret)
                    reset_hermes_home_override(token)
        finally:
            set_multiplex_active(False)
        assert [key for path, key, _ in calls if path.endswith('/decision')] == ['Bearer A', 'Bearer A']
    entered, release = threading.Event(), threading.Event()
    with endpoint(entered=entered, release=release) as (url, _):
        agent = SimpleNamespace(provider='custom', platform='cli', base_url=url, model='loaded-model',
            api_key='test', client=None, tools=[{'function': {'name': 'terminal'}}],
            valid_tool_names={'terminal'}, _cached_system_prompt='original', _interrupt_requested=False)
        def cancel():
            if entered.wait(5):
                agent._interrupt_requested = True
                release.set()
        worker = threading.Thread(target=cancel, daemon=True)
        worker.start()
        prepare_mode_turn(agent, 'Hello', [])
        worker.join(5)
        assert agent._cached_system_prompt == 'original'
        assert agent.valid_tool_names == {'terminal'}
        assert not hasattr(agent, '_conversation_mode_pending_commit')
