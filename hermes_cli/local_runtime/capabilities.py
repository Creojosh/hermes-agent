"""Capability answers for models served by the managed runtime.

Capability lookups (vision, and whatever comes next) consult cloud-shaped catalogs that have never
heard of a local GGUF, so a vision-capable local model reads as text-only and images detour to an
auxiliary cloud model — the wrong behavior twice over for a local-first user (broken feature, and a
screenshot silently leaving the machine).
"""

from __future__ import annotations

from contextlib import suppress
import logging

logger = logging.getLogger(__name__)

from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES as _LLAMACPP_ALIASES

# Image formats the managed server's decoder actually handles. llama.cpp decodes with stb_image:
# PNG/JPEG/GIF/BMP yes, WebP NO — and a WebP part fails SILENTLY (no HTTP error, no log line; the
# model just never sees an image and confabulates). Anything outside this set must be transcoded
# before the request. Measured live: the same red square answered 'Red' as PNG, 'Unseen' as WebP.
ACCEPTED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg"})


def is_managed_provider(provider: str, base_url: str = "") -> bool:
    """True when this provider/base_url pair points at the managed server. ``custom`` only counts
    when the base_url IS the managed endpoint — never claim someone else's custom server."""
    p = (provider or "").strip().lower()
    if p in _LLAMACPP_ALIASES:
        return True
    if p != "custom" or not base_url:
        return False
    with suppress(Exception):
        from hermes_cli.local_runtime.growth import is_managed_endpoint

        return is_managed_endpoint(base_url)
    return False


def _props_modalities(model_id: str) -> "bool | None":
    """Ask the running server whether this loaded child sees images. None when the server is down,
    the model isn't loaded, or the build doesn't report modalities."""
    with suppress(Exception):
        from hermes_cli.local_runtime.endpoint import managed_get_json, managed_root

        ep = managed_root()
        if ep is None:
            return None
        modalities = managed_get_json(*ep, f"/props?model={model_id}", timeout_s=3).get("modalities")
        if isinstance(modalities, dict) and "vision" in modalities:
            return bool(modalities["vision"])
    return None


def managed_model_supports_vision(model_id: str) -> "bool | None":
    """Ground-truth vision capability for a staged model, or None when the model isn't ours /
    nothing is known (caller keeps falling through)."""
    if not model_id:
        return None

    with suppress(Exception):
        from hermes_cli.local_runtime.bootstrap import (
            model_vision_enabled, staged_model_path, vision_projector_for)

        # Only answer for models actually staged with us.
        model_path = staged_model_path(model_id)
        if model_path is None:
            return None
        if not model_vision_enabled(model_id):
            return False
        live = _props_modalities(model_id)
        if live is not None:
            return live
        # Staged but not loaded (or an older server build): answer from the actual paired projector,
        # including user-selected nested libraries the curated catalog has never heard of.
        return vision_projector_for(model_path) is not None
    return None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402
import urllib.request  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
