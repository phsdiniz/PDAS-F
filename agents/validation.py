"""
agents/validation.py — ValidationAgent

A single logical agent with two invocation contexts, matching the
architecture description ("the Validation Agent verifies both the
user-provided responses and the outputs of the Task Agents"):

  - call_validation_agent:        validates the user's last message.
  - call_validation_agent_output: validates a Task Agent's output against
    the plan's expected output, and — on the plan's final stage — also
    compiles the final filled form (subsuming the old Output Generator
    Agent's role).

Both share the same agent identity ("validation_agent") for model
resolution and logging purposes; they are kept as two separate prompts
because the input shapes and goals differ enough (a chat message vs. a
structured task output plus end-of-plan form compilation) that merging
them into a single prompt would make the instructions harder to follow
and more error-prone, mirroring how the Planning Agent already uses
multiple specialized prompts under one identity.
"""

from __future__ import annotations

from agents.base import AgentResult, run_agent
from config import RunConfig

AGENT_NAME = "validation_agent"

# Fake response used when fake_validation=True (avoids unnecessary API calls)
_FAKE_VALIDATED = AgentResult(
    content       = '{"status": "VALIDATED"}',
    input_tokens  = 0,
    output_tokens = 0,
    tiktoken_input=0,
    tiktoken_output=0,
)


def call_validation_agent(
    context: list[dict[str, str]],
    run_cfg: RunConfig,
) -> AgentResult:
    """
    Args:
        context: Message history. The second-to-last is the context, the last is the message to validate.
        run_cfg: Run configuration. If run_cfg.fake_validation=True, returns VALIDATED without an API call.
    """
    if run_cfg.fake_validation:
        return _FAKE_VALIDATED

    if len(context) < 2:
        # Not enough history to validate — accept by default
        return _FAKE_VALIDATED

    history_text = f"{context[-2]['who']}: {context[-2]['message']}"
    message_text = f"{context[-1]['who']}: {context[-1]['message']}"

    return run_agent(
        prompt_file  = "ValidationAgent.txt",
        replacements = {
            "[Inserir 1]": history_text,
            "[Inserir 2]": message_text,
        },
        run_cfg      = run_cfg,
        agent_name   = AGENT_NAME,
    )


def call_validation_agent_output(
    task_action: str,
    task_input: str,
    expected_output: str,
    generated_output: str,
    current_task_number: int,
    total_tasks: int,
    run_cfg: RunConfig,
    original_form: dict | None = None,
    form_answers: dict | None = None,
) -> AgentResult:
    """
    Second invocation of the Validation Agent per turn: instead of
    validating the user's message, it checks whether the output generated
    by a Task Agent satisfies the expected output described in the plan's
    stage. A NOT_VALIDATED here acts as the replanning trigger (see
    agents.PDAS.call_planning_agent_revise and
    graphs.shared_nodes.node_validate_output / node_replan_stage).

    Only minimal positional context is sent (current task number, total
    number of tasks, and the current stage's action/input/output) rather
    than the full plan, to keep input tokens low — the full plan is only
    needed by the replanning call, not by validation.

    On the plan's final stage (current_task_number == total_tasks), if the
    output is VALIDATED, the agent additionally compiles the final filled
    form from original_form + form_answers and returns it under
    "final_output" — this is when "it's time to generate the final
    output", subsuming the old Output Generator Agent's role. original_form
    and form_answers are only actually injected into the prompt in that
    case, to avoid spending tokens on them at every intermediate stage.

    Args:
        task_action:          "action" field of the plan stage that was executed.
        task_input:           "input" field of the plan stage (description).
        expected_output:      "output" field of the stage (description of the expected output).
        generated_output:     Output actually produced by the SingleTaskAgent.
        current_task_number:  1-based index of the current stage.
        total_tasks:          Total number of stages in the plan.
        run_cfg:               Run configuration. If run_cfg.fake_validation=True,
                                returns VALIDATED without an API call (same flag used
                                for validating the user's message).
        original_form:         Original form JSON — only used when this is the final stage.
        form_answers:          Answers collected so far — only used when this is the final stage.
    """
    if run_cfg.fake_validation:
        return _FAKE_VALIDATED

    is_last_stage = current_task_number == total_tasks
    form_placeholder    = original_form if (is_last_stage and original_form) else "(não aplicável — não é a última etapa)"
    answers_placeholder = form_answers  if (is_last_stage and form_answers is not None) else "(não aplicável — não é a última etapa)"

    return run_agent(
        prompt_file  = "ValidationAgentOutput.txt",
        replacements = {
            "[Inserir 1]": task_action,
            "[Inserir 2]": task_input,
            "[Inserir 3]": expected_output,
            "[Inserir 4]": generated_output,
            "[Inserir tarefa atual]": str(current_task_number),
            "[Inserir total tarefas]": str(total_tasks),
            "[Inserir formulário]": form_placeholder,
            "[Inserir respostas]": answers_placeholder,
        },
        run_cfg      = run_cfg,
        agent_name   = AGENT_NAME,
    )
