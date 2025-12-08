#include "TaskStateMachine.h"

void TaskStateMachine::setTask(const TaskTemplate &task)
{
    _task = task;
    _stepStatus.clear();
    _sessionActive = false;
    if (!_task.task.subtasks.empty())
    {
        // Flatten first subtask steps for status tracking; subtask index managed separately
        for (const auto &step : _task.task.subtasks.front().steps)
        {
            StepStatus s;
            s.id = step.id;
            _stepStatus.push_back(s);
        }
        _currentSubtaskIndex = 0;
    }
    else
    {
        for (const auto &step : _task.task.steps)
        {
            StepStatus s;
            s.id = step.id;
            _stepStatus.push_back(s);
        }
        _currentSubtaskIndex = -1;
    }
    _currentStepIndex = _stepStatus.empty() ? -1 : 0;
    _state = _stepStatus.empty() ? State::Idle : State::Ready;
}

void TaskStateMachine::reset()
{
    _stepStatus.clear();
    _currentStepIndex = -1;
    _currentSubtaskIndex = -1;
    _state = State::Idle;
    _sessionActive = false;
}

TaskStateMachine::Transition TaskStateMachine::beginSession()
{
    Transition t;
    if (_state == State::Ready)
        _sessionActive = true;
    t.state = _state;
    if (auto cid = currentStepId(); cid.has_value())
    {
        t.current = StepRef{currentSubtaskId().value_or(""), *cid, {}};
    }
    return t;
}

TaskStateMachine::Transition TaskStateMachine::beginSubtask()
{
    Transition t;
    if (_state == State::SubtaskReady)
    {
        _state = State::Running;
        t.subtaskStarted = true;
    }
    t.state = _state;
    if (auto cid = currentStepId(); cid.has_value())
    {
        t.current = StepRef{currentSubtaskId().value_or(""), *cid, {}};
    }
    return t;
}

TaskStateMachine::Transition TaskStateMachine::advance()
{
    Transition t;
    if (!_sessionActive || _stepStatus.empty())
    {
        t.state = _state;
        return t;
    }

    if (_state == State::Ready)
    {
        _state = State::SubtaskReady;
        t.state = _state;
        if (auto cid = currentStepId(); cid.has_value())
            t.current = StepRef{currentSubtaskId().value_or(""), *cid, {}};
        return t;
    }

    if (_state == State::SubtaskReady)
    {
        _state = State::Running;
        t.state = _state;
        if (auto cid = currentStepId(); cid.has_value())
            t.current = StepRef{currentSubtaskId().value_or(""), *cid, {}};
        t.subtaskStarted = true;
        return t;
    }

    if (_currentStepIndex >= 0 && _currentStepIndex < static_cast<int>(_stepStatus.size()))
    {
        auto &step = _stepStatus[static_cast<size_t>(_currentStepIndex)];
        step.done = true;
        step.attempts += 1;
        if (_currentStepIndex + 1 < static_cast<int>(_stepStatus.size()))
        {
            _currentStepIndex += 1;
        }
        else
        {
            t.subtaskCompleted = true;
            if (_currentSubtaskIndex >= 0 && !_task.task.subtasks.empty())
            {
                const int total = static_cast<int>(_task.task.subtasks.size());
                const bool lastSubtask = (_currentSubtaskIndex + 1 >= total);
                _currentSubtaskIndex = (total == 0) ? -1 : (lastSubtask ? 0 : _currentSubtaskIndex + 1);
                _stepStatus.clear();
                if (_currentSubtaskIndex >= 0 && _currentSubtaskIndex < total)
                {
                    for (const auto &s : _task.task.subtasks[static_cast<size_t>(_currentSubtaskIndex)].steps)
                    {
                        StepStatus st;
                        st.id = s.id;
                        _stepStatus.push_back(st);
                    }
                }
                if (lastSubtask)
                    t.taskCompleted = true;
                _currentStepIndex = _stepStatus.empty() ? -1 : 0;
                _state = State::Ready;
            }
            else
            {
                // Single-task (no subtasks) loops steps
                t.taskCompleted = true;
                _currentStepIndex = _stepStatus.empty() ? -1 : 0;
                _state = State::Ready;
            }
        }
    }
    t.state = _state;
    if (auto cid = currentStepId(); cid.has_value())
        t.current = StepRef{currentSubtaskId().value_or(""), *cid, {}};
    return t;
}

TaskStateMachine::Transition TaskStateMachine::abort()
{
    Transition t;
    if (_state == State::Running || _state == State::Ready || _state == State::SubtaskReady)
        _state = State::Aborted;
    _sessionActive = false;
    t.state = _state;
    return t;
}

TaskStateMachine::Transition TaskStateMachine::stop()
{
    Transition t;
    if (_state == State::Completed || _state == State::Aborted)
    {
        _state = State::Ready;
        _currentStepIndex = _stepStatus.empty() ? -1 : 0;
    }
    _sessionActive = false;
    t.state = _state;
    return t;
}

std::optional<std::string> TaskStateMachine::currentStepId() const
{
    if (_currentStepIndex >= 0 && _currentStepIndex < static_cast<int>(_stepStatus.size()))
        return _stepStatus[static_cast<size_t>(_currentStepIndex)].id;
    return std::nullopt;
}

std::optional<std::string> TaskStateMachine::currentSubtaskId() const
{
    if (_currentSubtaskIndex >= 0 && _currentSubtaskIndex < static_cast<int>(_task.task.subtasks.size()))
        return _task.task.subtasks[static_cast<size_t>(_currentSubtaskIndex)].id;
    return std::nullopt;
}

TaskStateMachine::Snapshot TaskStateMachine::snapshot() const
{
    Snapshot s;
    s.state = _state;
    s.currentSubtaskIndex = _currentSubtaskIndex;
    s.currentStepIndex = _currentStepIndex;
    if (auto id = currentStepId(); id.has_value())
        s.currentStepId = *id;
    if (_currentSubtaskIndex >= 0 && _currentSubtaskIndex < static_cast<int>(_task.task.subtasks.size()))
        s.currentSubtaskId = _task.task.subtasks[static_cast<size_t>(_currentSubtaskIndex)].id;
    s.steps = _stepStatus;
    s.subtaskSteps = _stepStatus;
    return s;
}
