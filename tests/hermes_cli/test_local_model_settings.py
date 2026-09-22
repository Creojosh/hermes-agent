"""Per-model overrides persist and reach the real preset writer without leaking to peers."""
import struct
from pathlib import Path

import pytest

from hermes_cli.local_runtime.model_settings import get_model_settings, save_model_settings, validate_settings


def test_settings_roundtrip_to_presets_and_reset(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import presets
    from hermes_cli.local_runtime.estimator import HardwareBudget
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli import config as config_mod
    from hermes_cli.local_runtime.model_settings import split_model_args

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    config = config_mod.load_config()
    config["local_runtime"]["extra_args"] = [
        "--split-mode", "layer", "--tensor-split=1,1", "--typical-p", "0.9", "-n", "4096", "--metrics",
    ]
    config_mod.save_config(config)
    assert split_model_args(config["local_runtime"]["extra_args"])[1] == ["--metrics"]
    library = tmp_path / "models"
    library.mkdir()
    for name in ("Alpha", "Beta"):
        (library / f"{name}.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
    overrides = {
        "split-mode": "tensor", "tensor-split": "2,1", "ctx-size": "8192", "cache-type-k": "q8_0",
        "load-mode": "mmap", "n-cpu-moe": "4", "predict": "2048", "reasoning-budget": "512",
        "typical": "0.95", "presence-penalty": "-0.5", "frequency-penalty": "0.2",
        "dry-multiplier": "0.8", "mirostat": "2", "rope-scaling": "yarn", "rope-scale": "2",
        "spec-type": "ngram-simple", "spec-draft-n-min": "0", "spec-draft-n-max": "3",
        "image-min-tokens": "-1", "image-max-tokens": "1024",
    }
    save_model_settings("Alpha", overrides)
    for home in (tmp_path / ".hermes", tmp_path / "B", tmp_path / ".hermes"):
        token = set_hermes_home_override(home)
        try:
            assert get_model_settings("Alpha") == overrides
            ini = tmp_path / "presets.ini"
            presets.generate_presets(library, HardwareBudget(12 << 30, 12 << 30, 16 << 30), ini)
            result = presets.read_preset_decisions(ini)
            assert all(result["Alpha"].keys[key] == value for key, value in overrides.items())
            assert result["Beta"].keys.get("split-mode") == ("layer" if home.name == ".hermes" else None)
            assert result["Beta"].keys.get("tensor-split") != "2,1"
            assert result["Beta"].keys.get("predict") == ("4096" if home.name == ".hermes" else None)
            assert result["Beta"].keys.get("typical") == ("0.9" if home.name == ".hermes" else None)
            assert "frequency-penalty" not in result["Beta"].keys
        finally:
            reset_hermes_home_override(token)
    save_model_settings("Alpha", {})
    assert get_model_settings("Alpha") == {}
    assert presets.preset_for_model(library / "Alpha.gguf", HardwareBudget(12 << 30, 12 << 30, 16 << 30), set()).keys["tensor-split"] == "1,1"


@pytest.mark.parametrize("values", [
    {"tensor-split": "2,1\n[other]"}, {"tensor-split": "0,0"},
    {"ctx-size": "nan"}, {"ctx-size": "1"}, {"port": "8080"},
    {"split-mode": "invalid"}, {"top-p": "1.1"},
    {"presence-penalty": "nan"}, {"frequency-penalty": "inf"},
    {"xtc-probability": "1.1"}, {"mirostat": "3"}, {"rope-scale": "0"},
    {"dry-base": "0.5"}, {"reasoning-budget": "-2"}, {"load-mode": "invalid"},
    {"spec-draft-n-min": "5", "spec-draft-n-max": "2"},
    {"image-min-tokens": "2048", "image-max-tokens": "1024"},
])
def test_invalid_model_options_are_rejected(values):
    with pytest.raises(ValueError):
        validate_settings(values)
