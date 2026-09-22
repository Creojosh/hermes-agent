# llama.cpp selective compression: initial validation

Synthetic live requests on 2026-09-22. Server build `b11052-14d04e755`,
loaded model `Qwen3.8-Flash-Next-UD-IQ3_XXS`, 12 decision sequences.
No model was loaded, unloaded, or restarted by the probes.

| Route | First decision | Repeated decision | Cached prefix tokens | One-token chat before / after |
| --- | ---: | ---: | ---: | ---: |
| Direct child | 7.969 s | 2.641 s | 0 / 574 | 1.062 / 0.219 s |
| Router, same child | 2.796 s | 2.641 s | 574 / 574 | 0.219 / 0.234 s |

Both routes returned the requested model for chat and decisions; residency was
unchanged. The direct request used a cold decision prefix; the router reused it.
This is not an independent cold-router comparison. The synthetic decision had
16 boolean fields, 645 prompt tokens and 102 scored rows.

Child process RSS went from 52,976,173,056 to 53,573,566,464 bytes during the direct
probe, then to 53,687,554,048 during the router probe. These are process snapshots,
not isolated allocations attributable solely to decisions. A later GPU snapshot
reported 13,381 MiB on the RTX 5080 and 190 MiB on the RTX 3080; there is no paired
GPU baseline and therefore no GPU-memory delta claim.

The actual production selector was also called through the router with a small
synthetic conversation. It returned fallback after 5.047 s, preserving the original
history. Thus the five-second deadline is useful here but prevents a claim that
selective compression is faster or successful for this model/workload.

Validation: 10 HTTP/profile/selection cases plus 191 existing compression,
timeout and commit-fence cases passed via `scripts/run_tests.sh`. The repository
venv referenced a missing Python, so tests used an isolated Python 3.12 environment.

The eval runner now provides `llama_decision` and `llama_summary` arms for matched
recall/token/time comparisons. Full transcript recall evaluation and independent
cold-router/GPU memory measurements remain unperformed. No performance or recall
improvement is claimed by this receipt.

Chat routing follow-up: 161 targeted Python tests (including real streaming HTTP,
durable promotion/resume, A→B→A profile defaults, cancellation, refused fabricated
tool calls and RPC mode changes) and 17 Desktop UI tests passed; Desktop TypeScript
checking passed. A live router recheck found no loaded models, so no model was
loaded for benchmarking. The two-second Chat router's real-model latency and
end-to-end Chat/Agent savings remain unmeasured.
