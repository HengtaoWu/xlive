# AsyncVoiceBench

AsyncVoiceBench evaluates voice agents that keep conversing while background tasks run: starting work, changing requirements, checking progress, cancelling, retrying, and reporting results. It provides recorded user speech, a deterministic task sandbox, a real-time duplex runner, and scoring. Systems supply their own speech and agent stack.

## Dataset

100 English episodes across 29 templates contain 456 utterances (26 min 14 s, mono 16 kHz PCM16). They require 144 task starts, 38 modifications, 10 cancellations, and 7 retries.

- `data/episodes/<episode_id>.json`: dialogue scripts, sandbox materials, goals, and rubrics.
- `data/audio/<episode_id>/<turn_id>.wav`: user recordings.

Only public context, tool schemas, and the domain catalog reach the agent. Reference transcripts, goals, hidden inventory, and rubrics stay with the evaluator. Relative dates use the episode's session date and timezone.

## Setup

Run from `benchmark/` with Python 3.10+:

```bash
cd benchmark
python -m pip install -r requirements-audio.txt
python -m async_voice_bench.cli validate
python -m async_voice_bench.index_audio
```

## Connect an agent

Implement an async factory `module:create_agent(config, runtime)` returning the [DuplexAgent interface](async_voice_bench/audio_agent.py):

| Method | Purpose |
| --- | --- |
| `await start(context)` | Initialize a fresh session with context, tools, catalog, and audio formats |
| `await send_audio(pcm)` | Receive continuous 20 ms mono 16 kHz PCM16 frames, including silence |
| `events()` | Yield audio, stream endings/cancellations, tool calls, and errors |
| `await tool_result(call_id, result)` | Receive tool acknowledgements or validation errors |
| `await backend_event(event)` | Receive asynchronous results, failures, or clarification requests |
| `await playback(event)` | Receive playback and interruption feedback |
| `await aclose()` | Release resources |

Input, output, tool replies, and backend events run concurrently; inference must not block the event loop. The agent handles endpointing and explicitly cancels playback when needed. Use unique utterance `stream_id` and tool `call_id` values; do not silently retry mutating calls. Errors default to infrastructure failures; use `failure_kind="model"` only for model failures.

`config` comes from adapter JSON. `runtime` supplies `case_dir` and a trusted local `sandbox_url`; these are integration metadata, not model input. Keep API keys in environment variables because configs are saved in the run manifest.

Alternatively, use `async_voice_bench.websocket_agent:create_agent` with `{"url":"ws://127.0.0.1:9000/agent"}`. See the [WebSocket adapter](async_voice_bench/websocket_agent.py) for message formats. The server must return `{"type":"ready"}` within 60 s of `start`, before other output; the episode clock starts after initialization.

## Run and score

Provide importable `my_agent:create_agent` and async `my_asr:transcribe(wav_path, config) -> str` implementations, plus their JSON configs:

```bash
bash run_audio.sh \
  --agent my_agent:create_agent --agent-config agent.json \
  --transcriber my_asr:transcribe --transcriber-config asr.json \
  --system-name my-agent --out artifacts/my-agent \
  --programmatic-report artifacts/my-agent.summary.json

# Full evaluation, including LLM judgments:
export JUDGE_API_KEY="your-openai-api-key"
bash evaluate.sh \
  --runs artifacts/my-agent --out artifacts/my-agent-evaluation
```

Replace the key above in your terminal. `OPENAI_API_KEY` also works; `JUDGE_API_KEY` takes precedence. Set `JUDGE_BASE_URL` for an alternative endpoint (default: `https://api.openai.com/v1`) or `ASYNCBENCH_PYTHON` to select an interpreter.

Useful options:

- `--episode en_t01_tokyo_first_trip`: run one diagnostic episode.
- `--start-index 3`: begin at the third episode in dataset order.
- `--continue-on-error`: continue after infrastructure failures, stopping after 3 consecutive failures by default (`--max-consecutive-errors`).

Existing attempts are skipped on resume. Use a new output directory when settings change, and keep the programmatic report outside the run directory. To recompute it without calling the judge:

```bash
python -m async_voice_bench.score_programmatic \
  --runs artifacts/my-agent --out artifacts/my-agent.summary.json
```

Episodes have a 420 s default budget. Within that budget, final observation requires 90 s after the last user turn, 30 s after the latest task event, and 12 s without playback.

## Task sandbox

All systems use `submit_task`, `amend_task`, `get_task`, `cancel_task`, and `list_tasks`. Submission returns a task ID immediately; results arrive as backend events. Amendments merge changes into existing arguments and create a new version. Execution is deterministic, using finite session materials; `research_sources` searches frozen records, without live web access. Empty search results mean no inventory match. Missing tool materials return `tool_unavailable` without creating a task.

For an external harness:

```bash
python -m async_voice_bench.realtime serve \
  --episode en_t01_tokyo_first_trip --port 8765
```

The [HTTP interface](async_voice_bench/realtime.py) exposes `/context`, `/tool`, and `/events`. `/reset`, `/turn_start`, `/turn_end`, and `/dump` are harness-only and must remain inaccessible to the model. Custom harnesses must preserve trigger rules, playback handling, observation budgets, and timing.

## Metrics

| Metric | Definition |
| --- | --- |
| ToolSel ↑ | Effect-level task-operation F1, without per-operation deadlines |
| ArgAcc ↑ | Operation and final-state argument accuracy, including unchanged constraints |
| Pass@1 ↑ | Exact required operations, arguments, and final task states in one attempt |
| RespQual ↑ | Correctness and helpfulness, mapped from 1–5 to 0–1 |
| Natural ↑ | Per-turn interaction and applicable delivery quality, mapped to 0–1 |
| FU-Ans ↑ | Fraction of eligible checkpoint turns receiving a substantive response |
| FU-Lat ↓ | Substantive response latency p50/p90, with answered/eligible counts |

The first three are programmatic. The others use heard transcripts, backend evidence, measured timing, and the judge `gpt-5.2-2025-12-11` (`reasoning_effort=medium`, `max_completion_tokens=16384`). Judge outputs and prompts are cached; judge errors block scoring.

Pass@1 uses `tool_selection_exact_and_args_exact_v1`; dialogue quality does not veto it. `InteractionSuccess` separately checks complete execution, grounded responses, required delivery, and result faithfulness ≥4/5. Deadlines, talk-over, dispatch silence, and rejected calls are diagnostics.

## Outputs and reporting

Runs save `manifest.json`, per-episode JSON, and `cases/<episode_id>/` containing input/output WAVs, heard segments and transcripts, adapter events, and sandbox evidence. Evaluations add `summary.md`, `summary.json`, per-episode judgments, and `evaluation_status.json`.

Report the evaluated scope, model/decoding settings, endpointing or client-side VAD, implementation revisions, and any benchmark-based tuning. Preserve failed attempts; do not select the best retry. Missing runs and infrastructure failures block a complete comparison; model failures remain in the denominator.

Scores do not measure prosody, timbre, or acoustic naturalness. FU-Lat uses utterance-segment starts rather than word alignment. Fixed user scripts cannot answer arbitrary clarifications; preserved audio supports optional listening analysis.
