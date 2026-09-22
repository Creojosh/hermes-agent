---
sidebar_position: 4
title: Local Models
description: Run models entirely on your own machine — no account, no API key, nothing leaves your computer.
---

# Local Models

Hermes can run open models entirely on your own machine. It downloads and
manages the inference engine (llama.cpp), picks the right build of each
model for your hardware, and handles memory so you never configure context
sizes or GPU layers unless you want to override them. You pick a model; Hermes supplies defaults.

Nothing leaves your computer: no account, no API key, and no network access
after a model is downloaded.

## Getting started

1. Open **Settings → Providers → Local Models** (or choose **Run models
   locally** during onboarding).
2. Click **Install runtime**. Hermes downloads the official llama.cpp
   build for your hardware (a few hundred MB), verifies it, and keeps it
   updated.
3. Optionally choose a **Model storage folder**. Hermes immediately detects
   complete `.gguf` models anywhere below that folder (including
   publisher/repository subfolders), and future downloads use that folder.
4. Pick a model from the catalog and click **Download**.
5. Click **Use**. New chats now run on the local model.

That's the whole flow. The server starts and stops with Hermes, restarts
survive app restarts, and switching back to a cloud provider is one click
in the model picker.

## Parameters per model

Use **Model parameters** beside a downloaded model to adjust GPU distribution,
context and batch sizes, CPU threads, KV cache, flash attention, and sampling.
Additional groups expose repetition/presence/frequency and DRY penalties, Mirostat
and dynamic temperature, RoPE/YaRN context scaling, speculative decoding, and image
token limits. Model loading mode, CPU MoE offloading, output token limits, and
reasoning budget are also available. Search matches parameter names and group labels.
Speculative decoding needs a compatible model and, for external draft modes, an
already configured draft model. Vision limits only apply to vision models. Sampling
and reasoning settings are server defaults; explicit values in a request take precedence.
Blank fields inherit existing defaults; **Reset overrides**, then **Save parameters**,
restores them. Parameters are stored per model in the machine's default configuration
(`local_runtime.model_settings`) and shared by all profiles using that library.

Explicit model values take precedence over global model defaults and generated presets.
For example, one model can use `split-mode: tensor` and `tensor-split: 2,1` while
another uses `split-mode: layer`. These values are written into llama.cpp's per-model
presets; global model flags are moved into those presets so the router cannot overwrite
individual choices. A custom context size stays fixed instead of growing automatically.

Saving restarts the local server when it is running, unloading all resident models;
they reload on demand. **Global options · advanced** holds server-wide controls and
lets you remove unrecognized arguments explicitly. **Raw llama-server --help** remains
available for reference.

## How Hermes picks what to download

Every model in the catalog is priced against **your machine** before you
download anything. Each row shows:

- **Memory fit** — green (*Fits your GPU*: runs entirely in GPU memory),
  amber (*Uses system RAM*: works, but slower), or red (*Too big for this
  machine*).
- **Context** — the window the model starts with and the maximum it can
  grow to.
- The download size of the build selected for your hardware.

Models ship in several quality grades (quantizations). Hermes picks the
highest-quality build that runs fully on your GPU; machines with less
memory get a more compact build of the same model with the same
guarantees. Below 4-bit the quality loss is too severe, so Hermes never
offers builds smaller than that — a machine that can't run the 4-bit
build spilled to system RAM simply can't run that model.

Models that don't fit stay visible with the reason, so you always know
what a hardware upgrade would unlock.

## How memory management works

Local models live or die by memory placement, so Hermes manages it
end-to-end by default; explicit per-model parameters override these defaults:

- **Models start at a context window that fully fits your GPU** and grow
  toward their native maximum as your conversation needs more room. You
  may see "Context window grown" in the status feed during long sessions
  — that's the window expanding, not an error.
- **Every recommended model gets at least a 64K context window.** When a
  model is larger than your GPU's memory, Hermes deliberately places the
  overflow in system RAM in the order that hurts least (expert weights
  first, never the attention cache), trading some speed to protect the
  context guarantee.
- **Memory fit includes the launch configuration**, not just the model file:
  context state, runtime buffers, the vision projector, and MTP buffers all
  count. For multi-token prediction (MTP), Hermes uses smaller batches when
  larger batches would spill at the same context window. MTP stays enabled.
  The same calculation runs when a grown window is restored after restart.
- **Conversation compression follows a growth check.** If a larger window
  cannot fit, generation is too slow, or the native maximum is reached,
  Hermes compresses instead of claiming a window the server did not receive.
- Idle models are unloaded after 15 minutes to free GPU memory; they
  reload automatically on the next message.

## The status bar

Right-click the status bar and enable **System resources** to see live GPU
utilization, GPU memory, and RAM while local models run. The context meter
always reflects the window the model is actually running with.

## Finding more models

The catalog is a curated starting point, not a boundary. The **Find more
models** section on the same page searches all of Hugging Face:

- Results show download counts and a per-file fit check sized to your
  machine, so you know before downloading whether a build runs fully on
  your GPU.
- Anything you download behaves exactly like a catalog model — Hermes
  reads the model file itself to pick its context window and memory
  placement. The only difference: community models don't carry our
  "validated" testing badge.
- Already have a `.gguf` file on disk? **Add model file** links it into
  your library without copying it (the original stays where it is), and
  it's usable immediately.
- Vision projectors such as `mmproj-F16.gguf` are paired with the model in
  the same folder. They never appear as standalone models; use the model's
  **Vision** checkbox to decide whether Hermes loads the projector.

## Using your own llama-server

If a llama-server is already running on your machine, Hermes detects it
and uses it instead of starting its own. Point a custom endpoint at any
OpenAI-compatible server for full manual control — the managed runtime is
a default, not a requirement. You can enter the server root (for example
`http://127.0.0.1:8080`) or the full `/v1` URL: the endpoint test tries
both and saves the variant that actually served `/models`, so chat
requests go to the same prefix the model list came from. For manual setups (Ollama, MLX, custom
builds, headless CLI machines), see
[Run Hermes Locally with Ollama](../guides/local-ollama-setup.md) and
[Run Local LLMs on Mac](../guides/local-llm-on-mac.md).

## Configuration

The managed runtime is controlled by the `local_runtime` section of
`config.yaml`. The desktop UI writes these values for you; they're
documented for CLI and headless use. Because the model library is a machine
asset, `models_path` is read from the default profile's `config.yaml`:

```yaml
local_runtime:
  enabled: false     # true = start the managed server with Hermes.
                     # The desktop "Use" button sets this automatically.
  backend: auto      # auto | cuda | metal | vulkan | hip | cpu
  tag: b10362        # pinned llama.cpp release; Hermes updates it with
                     # each release after re-validation
  models_path: ""    # empty = <default Hermes home>/models; a custom path
                     # is shared by every profile on this machine
  vision_disabled_models: []  # managed by the per-model Vision checkboxes
  detect_ports: [8081]  # extra ports to probe for a llama-server you run
                        # yourself (the default probe is :8080 only)
```

Running `llama-server` yourself on a fixed port works with the same
`model.provider: llamacpp` selection — either list the port in
`local_runtime.detect_ports`, or define the endpoint explicitly under
`providers:` (an explicit entry wins over server detection):

```yaml
providers:
  llamacpp:
    base_url: http://127.0.0.1:8081/v1
    model: my-model
```

The `/model` → **Local** picker row and `provider: llamacpp` resolve to that
server; with no server reachable the error names the local runtime ("the local
model server isn't running") instead of an unknown-provider or missing-API-key
message.

By default, models and runtime builds live under the Hermes home directory
(`models/` and `runtimes/llamacpp/`). The desktop's **Model storage folder**
control can place the shared model library elsewhere; switching profiles does
not switch or duplicate that library. With a custom library selected, the pane
shows only detected models instead of the curated recommendations. Selecting a
local model as your main model uses the standard `model.provider: llamacpp` +
`model.default` settings — the same shape as every other provider.

## Selective context compression with a compatible fork

Hermes can use the loaded model's `POST /v1/decision` endpoint to clear obsolete
tool outputs during automatic compression. When the remaining context meets the
compression target, useful content stays verbatim and no summary is generated.
Otherwise Hermes runs its normal compression on the original history.

This requires a build supporting the
[parallel-decision API](https://github.com/thecodacus/llama.cpp/tree/parallel-decision/tools/parallel-decision).
It does not install TypeSafe Jev or replace the official runtime download.
For a supplied fork binary, use the existing settings:

```yaml
local_runtime:
  runtime_path: /path/to/directory/containing/llama-server
  extra_args: ["--decision-seqs", "12"]
compression:
  decision:
    mode: auto  # auto | off
```

Preserve any other `extra_args` you already use. Start an external server with the
same decision flag and configure its endpoint normally. The fork requires at least
three decision sequences; additional sequences consume model-dependent memory,
especially with sliding-window or recurrent attention. Twelve is an example,
not a universal optimum. Runtime flag changes take effect on the next server start.

Hermes uses the session's endpoint, model and credentials. It sends batches only
at the compression boundary, with a five-second total budget. Unsupported servers,
invalid responses, insufficient savings and timeouts retain normal compression.
An output is cleared only on a discard decision with probability at least 0.90;
this score is not a calibrated guarantee. Recent turns, instructional skills and
tool-call structure are protected. Focused/manual compression and custom context
engines retain their existing behavior. No chat `/decision` command is added.

Set `compression.decision.mode: off` to disable selection. Capability answers are
cached for five minutes per profile, endpoint, model and session client identity;
reconnecting or switching routes rechecks capability. No prompt contents or
credentials are written to decision diagnostics.

## Lightweight Chat and Agent modes

New interactive llama.cpp conversations default to `agent.conversation_mode: auto`.
While in Auto/Chat, a bounded `/v1/decision` request classifies each incoming user
request on the same loaded model. Self-contained questions use a short Chat prompt,
without tool schemas, execution instructions or the skill catalog. Identity,
user preferences, history, streaming, storage and compression remain available.
Chat cannot execute tools, including tool calls fabricated by the model.

Requests needing tools promote the session to Agent between turns. Agent keeps its
existing tool loop and Tool Search; no classifier runs between tool calls. Promotion
is sticky across restarts. Existing conversations and canonical Bot Chat sessions
retain Agent. Unsupported decisions, uncertainty or the two-second routing deadline
fall back to Agent. Other providers retain their existing behavior.

In Desktop, the composer exposes **Auto**, **Chat**, and **Agent** for supported
sessions. An explicit selection takes effect on the next request. `Chat` forces
text-only conversation; `Agent` bypasses routing. The profile configuration controls
the default for new sessions:

```yaml
agent:
  conversation_mode: auto  # auto | chat | agent
```

Decision, Chat and Agent have separate stable prompt prefixes. A promotion or
explicit mode change is a deliberate cache boundary; it preserves session identity
and history without duplicating messages. These logical prefixes do not reserve
three KV caches: actual cache residency depends on llama.cpp slots and memory.
Chat reduces input tokens, but routing adds latency and the first Agent request
must process its new prefix. Measure end-to-end latency with cold and warm caches
before assuming a speed or memory benefit.

## Requirements and limits

- **Windows and Linux:** NVIDIA GPU (CUDA) or CPU. **macOS:** Apple
  Silicon (Metal). Vulkan builds serve AMD GPUs.
- A GPU with 8 GB+ of memory runs the small catalog models comfortably;
  16 GB+ runs the 27–35B models at high quality.
- Model downloads are byte-size checked against the catalog during the
  transfer; an incomplete download is deleted and reported, never
  half-used. (Only the runtime engine zips are SHA-256 verified.)
- Deleting a model removes every file it staged, including vision
  adapters and speculative-decoding companions.
