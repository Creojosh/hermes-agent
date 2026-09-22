"""Finite decisions on the session's existing llama.cpp route.

No generation probes: the first useful decision confirms support. The bounded
cache contains capability answers only, never conversation text or credentials.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from hermes_constants import hermes_home_key


class DecisionUnavailable(Exception):
    """The caller must use ordinary compression, without committing a partial batch."""


@dataclass(frozen=True)
class DecisionRoute:
    base_url: str
    model: str
    api_key: str
    provider: str
    generation: int = 0

    @property
    def root(self) -> str:
        parts = urlsplit(self.base_url.rstrip('/'))
        if parts.scheme not in ('http', 'https') or not parts.netloc or parts.query or parts.fragment:
            raise DecisionUnavailable('invalid_route')
        path = parts.path.removesuffix('/v1')
        return urlunsplit((parts.scheme, parts.netloc, path, '', ''))

    @property
    def cache_key(self) -> tuple:
        return (hermes_home_key(), self.root, self.model, self.provider, self.generation,
                hashlib.sha256(self.api_key.encode()).digest())


_CAPABILITIES: OrderedDict[tuple, tuple[float, bool]] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _cached(key: tuple) -> bool | None:
    with _CACHE_LOCK:
        entry = _CAPABILITIES.get(key)
        if entry is not None and entry[0] > time.monotonic():
            _CAPABILITIES.move_to_end(key)
            return entry[1]
        _CAPABILITIES.pop(key, None)
    return None


def _remember(key: tuple, available: bool) -> None:
    with _CACHE_LOCK:
        _CAPABILITIES[key] = (time.monotonic() + 300, available)
        _CAPABILITIES.move_to_end(key)
        while len(_CAPABILITIES) > 128:
            _CAPABILITIES.popitem(last=False)


class DecisionTransport:
    def __init__(self, route: DecisionRoute, client: httpx.AsyncClient):
        self.route, self.client = route, client
        self.key = route.cache_key

    async def check_route(self) -> None:
        known = _cached(self.key)
        if known is False:
            raise DecisionUnavailable('unsupported_cached')
        if known is not True and self.route.provider not in {'llamacpp', 'llama.cpp', 'llama-cpp'}:
            # Generic custom providers must fingerprint as llama.cpp first.
            response = await self.client.get(self.route.root + '/props')
            if response.status_code in (404, 405):
                _remember(self.key, False)
                raise DecisionUnavailable('not_llamacpp')
            response.raise_for_status()
            props = response.json()
            if not isinstance(props, dict) or not props.get('build_info'):
                _remember(self.key, False)
                raise DecisionUnavailable('not_llamacpp')
        # Router autoload must not turn a background compression into a model load.
        # Single-model llama-server omits status; its only advertised model is resident.
        response = await self.client.get(self.route.root + '/v1/models')
        response.raise_for_status()
        data = response.json()
        models = data.get('data') if isinstance(data, dict) else None
        if not isinstance(models, list):
            raise DecisionUnavailable('invalid_models')
        for model in models:
            if not isinstance(model, dict):
                continue
            aliases = model.get('aliases') or []
            if self.route.model != model.get('id') and self.route.model not in aliases:
                continue
            status = model.get('status', {})
            if not isinstance(status, dict) or status.get('value', 'loaded') != 'loaded':
                raise DecisionUnavailable('model_not_loaded')
            return
        raise DecisionUnavailable('model_not_loaded')

    async def decide(self, schema: dict, instructions: str, context: str) -> dict:
        response = await self.client.post(self.route.root + '/v1/decision', json={
            'model': self.route.model, 'schema': schema, 'instructions': instructions,
            'contexts': [context], 'mode': 'tree', 'cache_prompt': True,
        })
        if response.status_code in (404, 405, 501) or (
            response.status_code == 400 and 'decisions are disabled' in response.text.lower()
        ):
            _remember(self.key, False)
            raise DecisionUnavailable('unsupported')
        response.raise_for_status()
        return response.json()

    def confirm(self) -> None:
        _remember(self.key, True)


async def run_bounded(work, *, deadline: float, cancelled) -> object:
    """Cancel socket I/O as well as the waiter when the compression owner leaves."""
    task = asyncio.create_task(work)
    try:
        while True:
            if cancelled():
                from agent.auxiliary_client import AuxiliaryExplicitCancellation
                raise AuxiliaryExplicitCancellation()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DecisionUnavailable('timeout')
            done, _ = await asyncio.wait({task}, timeout=min(0.05, remaining))
            if done:
                return task.result()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
