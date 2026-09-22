"""Synthetic live receipt: chat, first/repeated decisions, RSS, and model residency.

Run with python -m evals.compaction.scripts.llama_decision_probe. This never loads,
unloads or restarts a model. First-use is only cold if the server cache was cold.
"""
import argparse
import json
import os
from pathlib import Path
import time

import httpx
import psutil

from agent.decision_compression import _INSTRUCTIONS, _SCHEMA, _parse_response


def probe(base_url, model, api_key='', server_pid=None):
    root = base_url.rstrip('/').removesuffix('/v1')
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    process = psutil.Process(server_pid) if server_pid else None

    def rss():
        return process.memory_info().rss if process else None

    with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
        def models():
            response = client.get(root + '/v1/models')
            response.raise_for_status()
            return {row['id']: row.get('status', {}).get('value', 'loaded') for row in response.json()['data']}

        before = models()
        if before.get(model) != 'loaded':
            raise ValueError('Select a model already loaded on this server')

        def chat():
            start = time.monotonic()
            response = client.post(root + '/v1/chat/completions', json={
                'model': model, 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
                'max_tokens': 1, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
            })
            response.raise_for_status()
            return {'seconds': time.monotonic() - start, 'model': response.json().get('model')}

        receipt = {'model': model, 'rss_before': rss(), 'chat_before': chat(), 'decisions': []}
        context = json.dumps({'active_task': 'The build succeeded. Report the success.',
                              'history': [{'role': 'assistant', 'text': 'All tests passed.'}],
                              'candidates': {'0': {'tool': 'terminal', 'arguments': 'build',
                                                   'output': 'Obsolete progress log: compiling...'}}})
        for phase in ('first', 'repeat'):
            start = time.monotonic()
            response = client.post(root + '/v1/decision', json={
                'model': model, 'schema': _SCHEMA, 'instructions': _INSTRUCTIONS,
                'contexts': [context], 'mode': 'tree', 'cache_prompt': True,
            })
            response.raise_for_status()
            data = response.json()
            _parse_response(data)
            receipt['decisions'].append({'phase': phase, 'seconds': time.monotonic() - start,
                                         'model': data.get('model'), 'usage': data.get('usage'),
                                         'timings': data.get('timings'), 'rss': rss()})
        receipt.update(chat_after=chat(), rss_after=rss(), residency_unchanged=before == models())
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--key-env', default='LLAMA_API_KEY')
    parser.add_argument('--server-pid', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = probe(args.base_url, args.model, os.environ.get(args.key_env, ''), args.server_pid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
