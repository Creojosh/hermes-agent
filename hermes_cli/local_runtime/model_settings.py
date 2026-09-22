"""Validated, machine-scoped per-model llama.cpp preset overrides."""

from __future__ import annotations

import math
import re


# INI keys, not shell fragments: identity, paths and router lifecycle stay Hermes-owned.
FIELDS = {
    "ctx-size": ("context", "integer", 512, None),
    "batch-size": ("context", "integer", 1, None),
    "ubatch-size": ("context", "integer", 1, None),
    "threads": ("context", "integer", 1, None),
    "threads-batch": ("context", "integer", 1, None),
    "predict": ("context", "integer", -1, None),
    "keep": ("context", "integer", -1, None),
    "reasoning": ("context", "choice", None, ["on", "off", "auto"]),
    "reasoning-budget": ("context", "integer", -1, None),
    "n-gpu-layers": ("gpu", "integer", -1, None),
    "split-mode": ("gpu", "choice", None, ["none", "layer", "row", "tensor"]),
    "tensor-split": ("gpu", "text", None, None),
    "main-gpu": ("gpu", "integer", 0, None),
    "load-mode": ("gpu", "choice", None, ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]),
    "n-cpu-moe": ("gpu", "integer", 0, None),
    "flash-attn": ("cache", "choice", None, ["on", "off", "auto"]),
    "cache-type-k": ("cache", "choice", None, ["f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"]),
    "cache-type-v": ("cache", "choice", None, ["f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"]),
    "cache-ram": ("cache", "integer", -1, None),
    "cache-reuse": ("cache", "integer", 0, None),
    "temp": ("sampling", "number", 0, None),
    "top-k": ("sampling", "integer", 0, None),
    "top-p": ("sampling", "probability", 0, None),
    "min-p": ("sampling", "probability", 0, None),
    "typical": ("sampling", "probability", 0, None),
    "xtc-probability": ("sampling", "probability", 0, None),
    "xtc-threshold": ("sampling", "probability", 0, None),
    "dynatemp-range": ("sampling", "number", 0, None),
    "dynatemp-exp": ("sampling", "number", 0, None),
    "mirostat": ("sampling", "choice", None, ["0", "1", "2"]),
    "mirostat-lr": ("sampling", "number", 0, None),
    "mirostat-ent": ("sampling", "number", 0, None),
    "seed": ("sampling", "integer", -1, None),
    "repeat-penalty": ("penalties", "number", 0, None),
    "repeat-last-n": ("penalties", "integer", -1, None),
    "presence-penalty": ("penalties", "number", None, None),
    "frequency-penalty": ("penalties", "number", None, None),
    "dry-multiplier": ("penalties", "number", 0, None),
    "dry-base": ("penalties", "number", 1, None),
    "dry-allowed-length": ("penalties", "integer", 0, None),
    "dry-penalty-last-n": ("penalties", "integer", -1, None),
    "rope-scaling": ("rope", "choice", None, ["none", "linear", "yarn"]),
    "rope-scale": ("rope", "positive", 0, None),
    "rope-freq-base": ("rope", "number", 0, None),
    "rope-freq-scale": ("rope", "number", 0, None),
    "yarn-orig-ctx": ("rope", "integer", 0, None),
    "yarn-ext-factor": ("rope", "number", -1, None),
    "yarn-attn-factor": ("rope", "number", -1, None),
    "yarn-beta-slow": ("rope", "number", -1, None),
    "yarn-beta-fast": ("rope", "number", -1, None),
    "spec-type": ("speculative", "choice", None, [
        "none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark",
        "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
    ]),
    "spec-draft-n-max": ("speculative", "integer", 0, None),
    "spec-draft-n-min": ("speculative", "integer", 0, None),
    "spec-draft-p-min": ("speculative", "probability", 0, None),
    "image-min-tokens": ("vision", "integer", -1, None),
    "image-max-tokens": ("vision", "integer", -1, None),
}

_ALIASES = {
    "-c": "ctx-size", "-b": "batch-size", "-ub": "ubatch-size",
    "-t": "threads", "-tb": "threads-batch", "-ngl": "n-gpu-layers",
    "--gpu-layers": "n-gpu-layers", "-sm": "split-mode", "-ts": "tensor-split",
    "-mg": "main-gpu", "-fa": "flash-attn", "-ctk": "cache-type-k",
    "-ctv": "cache-type-v", "-s": "seed",
    "-n": "predict", "--n-predict": "predict", "--temperature": "temp",
    "--typical-p": "typical", "-lm": "load-mode", "-ncmoe": "n-cpu-moe",
    "-cram": "cache-ram", "-rea": "reasoning", "--draft-p-min": "spec-draft-p-min",
    **{f"--{key}": key for key in FIELDS},
}


def split_model_args(args: list[str]) -> tuple[dict[str, str], list[str]]:
    """Move model defaults into presets: router CLI flags otherwise win over every preset."""
    values, remaining = {}, []
    index = 0
    while index < len(args):
        flag, separator, inline = args[index].partition("=")
        key = _ALIASES.get(flag)
        if key is not None and (separator or index + 1 < len(args)):
            values[key] = inline if separator else args[index + 1]
            index += 1 if separator else 2
        else:
            remaining.append(args[index])
            index += 1
    return values, remaining


def global_model_settings() -> dict[str, str]:
    from hermes_cli.config import load_config_readonly

    return split_model_args((load_config_readonly().get("local_runtime") or {}).get("extra_args") or [])[0]


def validate_settings(values: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, raw in values.items():
        if key not in FIELDS:
            raise ValueError(f"Unsupported model option: {key}")
        value = str(raw).strip()
        if not value:
            continue
        _, kind, minimum, choices = FIELDS[key]
        if any(c in value for c in "\r\n\x00"):
            raise ValueError(f"Invalid value for {key}")
        if choices is not None and value not in choices:
            raise ValueError(f"{key}: choose {', '.join(choices)}")
        if kind in {"integer", "number", "probability", "positive"}:
            number = float(value)
            if (not math.isfinite(number) or (minimum is not None and number < minimum)
                    or (kind == "integer" and not re.fullmatch(r"-?\d+", value))
                    or (kind == "positive" and number <= 0)
                    or (kind == "probability" and number > 1)):
                raise ValueError(f"Invalid value for {key}: {value}")
        if key == "tensor-split":
            if not re.fullmatch(r"\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)*", value):
                raise ValueError("tensor-split: use comma-separated weights, e.g. 2,1")
            if not any(float(part) > 0 for part in value.split(",")):
                raise ValueError("tensor-split: at least one weight must be positive")
        result[key] = value
    if "batch-size" in result and "ubatch-size" in result:
        if int(result["ubatch-size"]) > int(result["batch-size"]):
            raise ValueError("ubatch-size must not exceed batch-size")
    for lower, upper in (("spec-draft-n-min", "spec-draft-n-max"), ("image-min-tokens", "image-max-tokens")):
        if lower in result and upper in result and 0 <= int(result[upper]) < int(result[lower]):
            raise ValueError(f"{lower} must not exceed {upper}")
    return result


def get_model_settings(model_id: str) -> dict[str, str]:
    from hermes_cli.local_runtime.bootstrap import _machine_runtime_section

    return dict((_machine_runtime_section().get("model_settings") or {}).get(model_id) or {})


def save_model_settings(model_id: str, values: dict[str, str]) -> None:
    from hermes_cli import config as config_mod
    from hermes_constants import get_default_hermes_root, set_hermes_home_override, reset_hermes_home_override

    values = validate_settings(values)
    token = set_hermes_home_override(get_default_hermes_root())
    try:
        config = config_mod.load_config()
        models = config.setdefault("local_runtime", {}).setdefault("model_settings", {})
        if values:
            models[model_id] = values
        else:
            models.pop(model_id, None)
        config_mod.save_config(config)
    finally:
        reset_hermes_home_override(token)
