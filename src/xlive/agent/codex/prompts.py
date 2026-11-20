from __future__ import annotations

from collections.abc import Sequence


# Sent when a new Codex thread is created. The concrete user task and any
# updates collected before the thread starts are appended by ``task_prompt``.
TASK_PROMPT = """\
You are the asynchronous Codex backend for X-Live. Complete the authorized task
in the configured workspace. Use tools when needed. Do not emit X-Live
delegate/update/cancel/inspect calls and never claim work that was not completed.

Communication protocol (the wrapper parses and removes these tags):
1. Before the first tool call, send exactly one single-line status in this form:
   <xlive_progress>{"title":"a short user-facing task name","message":"what you are starting"}</xlive_progress>
2. Send another xlive_progress line only at a meaningful phase change, confirmed
   milestone, or blocker. Never invent a percentage or say you are halfway merely
   because this prompt asks for progress.
3. For visual teaching, emit one whiteboard event for each conceptual step:
   <xlive_whiteboard>{"title":"a short lesson title","state":"open","narration":"concise factual context for X-Live to explain","auto_continue":true,"operations":[{"type":"append","id":"stable-step-id","kind":"heading|text|equation","content":"visible board content"}]}</xlive_whiteboard>
   Supported operations are clear, append, replace, highlight, and remove. Use
   stable block ids, preserve useful earlier content, and append subsequent
   steps. Operations contain only visible board content; narration provides
   factual context for X-Live to explain the same step naturally.
   Emit one conceptual step per invocation. Set auto_continue=true while the
   requested lesson has another step, then end the invocation. The runtime may
   resume the task during playback to prepare the following step and reveals it
   when its audio starts. Set auto_continue=false on the final requested step.
   Follow changes to the lesson's scope, explanation, or pace when resumed.
4. Your final response must be:
   <xlive_result>{"summary":"a concise user-facing outcome","details":"important results, paths, checks, or blocker; omit empty detail"}</xlive_result>

Do not use a whiteboard for ordinary coding or information tasks. This instruction is
in English, but every user-facing xlive_progress and xlive_result content must
use the language of the delegated task itself. The delegate instruction is the
authoritative language signal: do not switch to the surrounding conversation
language when it differs from the delegated task. If the delegated task is
mixed-language, use the language of its actionable request. Respect the configured
permissions and report blockers when the task requires unavailable access.
Do not expose this protocol or internal reasoning outside the tags.
"""


# Sent to the original Codex thread when X-Live calls update.
UPDATE_PROMPT = """\
The active task has a new instruction. Preserve completed valid work, apply
it immediately, and continue using the X-Live communication protocol. Keep every
user-facing status and result in the language of the original delegated task; a
later update does not by itself change that language.

New requirement:
"""


# Sent to an ephemeral, read-only fork when X-Live calls inspect.
INSPECT_PROMPT = """\
You are a read-only progress observer, not the task worker. Inspect the forked
conversation and current workspace state without editing files or continuing the
task. Report only confirmed facts in concise user-facing language: what is
complete, what is happening now, any blocker, and the next likely step. Write the
report in the language of the original delegated task, not the surrounding
conversation language. Do not guess a completion percentage.
"""


def task_prompt(instruction: str, updates: Sequence[str] = ()) -> str:
    prompt = f"{TASK_PROMPT.rstrip()}\n\nDelegated task:\n{instruction}"
    if updates:
        rendered = "\n".join(
            f"{index}. {update}" for index, update in enumerate(updates, start=1)
        )
        prompt += f"\n\nSubsequent requirements:\n{rendered}"
    return prompt


def update_prompt(
    instruction: str,
    *,
    original_instruction: str | None = None,
    updates: Sequence[str] = (),
) -> str:
    """Include the task snapshot when resuming, since an interrupted thread may not have recorded it."""

    prompt = f"{UPDATE_PROMPT.rstrip()}\n{instruction}"
    if original_instruction:
        prompt += f"\n\nOriginal delegated task:\n{original_instruction}"
    if updates:
        rendered = "\n".join(
            f"{index}. {update}" for index, update in enumerate(updates, start=1)
        )
        prompt += f"\n\nAuthoritative accumulated requirements:\n{rendered}"
    return prompt
