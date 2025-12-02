# Task State Machine Transitions

States:
- `Idle`: no task selected
- `Ready`: task selected, not started
- `Running`: task in progress
- `Completed`: all steps finished
- `Aborted`: task aborted

Events and transitions:
- Select task: `Idle -> Ready` (TaskMachine::setTask)
- Start task/recording: `Ready -> Running` (MainWindow::onStartRecording calls `start`)
- Advance step (Space / Step button): in `Running`
  - Mark current step done, attempts++
  - If not last step: `currentStepIndex++` (stay `Running`)
  - If last step: `state -> Completed`
- Retry (R / Retry button): log annotation/event, state unchanged
- Skip (K / Skip button): log annotation/event, state unchanged
- Error (E / Error button): log annotation/event, state unchanged
- Abort (A / Abort button): `Running|Ready -> Aborted`
- Stop recording: if `Running` then `Abort` (current code), logs stop; `state` set accordingly

Logged artifacts:
- `events.jsonl`: raw key/button events
- `annotations.jsonl`: state, current step, step overrides, trigger

Triggers to state/annotation:
- `Start` (button): logs start_recording, sets task selection, start state machine
- `Stop` (button): logs stop_recording, aborts if still running
- `Space`/`Step` button: advance; `annotations` with step progress
- `E`/`Error` button: log error annotation
- `K`/`Skip` button: log skip annotation
- `R`/`Retry` button: log retry annotation
- `A`/`Abort` button: abort state
