# Capture & Task State Machine (Keyboard + Manual Control)

## Concepts
- **Session**: one recording run under `captures/{date}/sess_YYYYMMDD_HHMMSS`. Each new capture starts a fresh session.
- **Task**: selected template (script or VLM-generated); state machine tracks steps.
- **Modes**: Script (predefined task) or VLM (generated task). Mode only affects task selection, not capture controls.
- **Triggers**: keyboard or UI buttons; both map to the same actions.

## States
- `Idle` (no task selected)
- `Ready` (task selected, not recording)
- `SubtaskReady` (task selected, recording started for session; current subtask not yet started)
- `Recording` (current subtask running, frames stored)
- `Paused` (recording paused)
- `Completed` (all subtasks/steps finished)
- `Aborted` (task interrupted/stopped before completion)

Task sub-state: `TaskStateMachine` tracks `currentSubtaskIndex`, `currentStepIndex`, `steps[done/attempts]`.

## Transitions & Actions
- **Select Task**: `Idle -> Ready` (task loaded/reset).
- **Start Recording** (button or key):
  - Requires `Ready`.
  - Create new session dir `captures/{date}/sess_...`.
  - Copy task to `task_used.json`; write `meta.json`; open `events.jsonl`; set `SubtaskReady`.
  - Task state: `Ready -> SubtaskReady`, current subtask = first, current step = first.
  - Log event/annotation; play audio “start”.
- **Begin Subtask / Advance Step** (button or key, default Space):
  - If `SubtaskReady`: begin current subtask, state -> `Recording`, announce step1 of current subtask.
  - If `Recording`: mark current step done/attempts++; if more steps in subtask -> move to next; if subtask done and more subtasks -> go to next subtask and enter `SubtaskReady`; if all done -> `Completed`.
  - Log event/annotation; play audio “next_step” or “complete” (or subtask start).
- **Retry** (key/button): log event/annotation; play “retry” (no state change).
- **Skip**: log; play “skip” (no state change).
- **Error**: log; play “error” (no state change).
- **Pause**: `Recording -> Paused`; log event; resume returns to `Recording`.
- **Abort**: from `Recording`/`Ready` -> `Aborted`; log; play “abort”.
- **Stop Recording**:
  - From `Recording`/`Paused`: stop capture; if task `Completed` -> final “stop_completed”; else “stop_running”.
  - Close streams; log event/annotation; end session.
  - Task state remains `Completed` or `Aborted` (if stopped early).
- **VLM Generate** (VLM mode): capture latest frame -> call VLM -> validate -> load as current task (source=vml), set `Ready`; play success/fail.

## Files per Session
- `task_used.json`: selected task (script or generated).
- `meta.json`: session/task info, start time, template path/version.
- `events.jsonl`: raw events (start/stop/step/retry/skip/error/abort/pause/resume/VLM generate).
- `annotations.jsonl`: references `task_used.json` + runtime info (timestamp, trigger, state, step overrides).
- `frames/`: color/depth per camera; `timestamps.csv` per camera.

## Keyboard Mappings (configurable in config.json)
- `start_stop` → default `Enter` (start from Ready -> SubtaskReady; stop from Recording/Paused)
- `advance` → default `Space` (from SubtaskReady: begin subtask & announce step1; from Recording: step advance)
- `retry` → `R` (optional)
- `skip` → `K` (optional)
- `error` → `E` (optional)
- `abort` → `A`

## Manual vs Keyboard
- UI buttons and keyboard shortcuts call the same handlers; behavior is identical.
- Switching tasks/modes is manual; all run/step/stop controls can be keyboard-driven.

## Audio Prompts (Piper)
- On start, step advance/complete, retry/skip/error/abort, stop (completed/interrupt), VLM success/fail.
- Text templates with placeholders `{task_id}`, `{step_id}`, `{step_desc}`.
