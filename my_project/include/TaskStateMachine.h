#pragma once

#include <optional>
#include <string>
#include <vector>

#include "TaskLoader.h"

class TaskStateMachine
{
public:
    enum class State
    {
        Idle,
        Ready,
        SubtaskReady,
        Running,
        Completed,
        Aborted
    };

    struct StepRef
    {
        std::string subtaskId;
        std::string stepId;
        std::string stepDescription;
    };

    struct Transition
    {
        State state{State::Idle};
        std::optional<StepRef> current;
        bool taskCompleted{false};
        bool subtaskCompleted{false};
        bool subtaskStarted{false};
    };

    struct StepStatus
    {
        std::string id;
        bool done{false};
        int attempts{0};
    };

    struct Snapshot
    {
        State state{State::Idle};
        int currentSubtaskIndex{-1};
        int currentStepIndex{-1};
        std::string currentStepId;
        std::string currentSubtaskId;
        std::vector<StepStatus> steps;
        std::vector<StepStatus> subtaskSteps; // flattened current subtask steps
    };

    TaskStateMachine() = default;

    void setTask(const TaskTemplate &task);
    void reset();
    // Ready -> SubtaskReady
    Transition beginSession();
    // SubtaskReady -> Running
    Transition beginSubtask();
    // Handle start-step request based on current state
    Transition startStep();
    // Finish current step -> next step/subtask/completion
    Transition finishStep();
    Transition abort();
    Transition stop(); // Completed/Aborted -> Ready

    State state() const { return _state; }
    std::optional<std::string> currentStepId() const;
    std::optional<std::string> currentSubtaskId() const;
    Snapshot snapshot() const;

private:
    TaskTemplate _task;
    State _state{State::Idle};
    int _currentSubtaskIndex{-1};
    int _currentStepIndex{-1};
    std::vector<StepStatus> _stepStatus;
    bool _sessionActive{false};
};
