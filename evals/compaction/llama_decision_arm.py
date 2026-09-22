"""Evaluate the production selector with a matched ordinary-compression control."""
import copy
import time
from types import SimpleNamespace

from agent.context_compressor import ContextCompressor
from agent.decision_compression import try_decision_compression
from agent.model_metadata import estimate_messages_tokens_rough


def compress(spec, messages):
    route = spec['route']
    compressor = ContextCompressor(
        model=route['model'], provider='llamacpp', base_url=route['base_url'], api_key=route['api_key'],
        config_context_length=route['context_length'], quiet_mode=True,
    )
    agent = SimpleNamespace(context_compressor=compressor, provider='llamacpp', model=route['model'],
                            base_url=route['base_url'], api_key=route['api_key'], client=None)
    before = estimate_messages_tokens_rough(messages)
    kwargs = {'current_tokens': before}
    started = time.monotonic()
    candidate = None
    if spec['engine'] == 'llama_decision':
        candidate = try_decision_compression(agent, copy.deepcopy(messages), kwargs)
    selected = candidate is not None
    selection_seconds = time.monotonic() - started
    from evals.compaction.runner import _AuxMeter
    with _AuxMeter() as meter:
        if candidate is None:
            candidate = compressor.compress(copy.deepcopy(messages), current_tokens=before, force=True)
    return candidate, compressor, {
        **meter.summary(), 'decision_selected': selected, 'decision_seconds': selection_seconds,
        'decision_saved_tokens': before - estimate_messages_tokens_rough(candidate),
    }
