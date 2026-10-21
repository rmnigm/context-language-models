import pytest

from clm_harness.utils import tokens as tk
from clm_harness.utils.budget import BudgetController

THINK = "let me think about the batch " * 50
TASK = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
OLD_TURN = {"role": "assistant", "content": "", "reasoning_content": THINK}
NEW_TURN = {"role": "assistant", "content": "", "reasoning_content": THINK,
            "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}


def reasoning_kept(budget: BudgetController, messages: list[dict]) -> list[bool]:
    return [bool(m.get("reasoning_content")) for m in budget.prompt_view(messages)]


@pytest.mark.parametrize(("mode", "messages", "kept"), [
    ("all", [*TASK, OLD_TURN, {"role": "user", "content": "next op"}], [False, False, True, False]),
    ("since_last_user", [*TASK, OLD_TURN, {"role": "user", "content": "next op"}],
     [False, False, False, False]),
    # Turns after the newest query keep their reasoning, tool results do not end the run of turns.
    ("since_last_user", [*TASK, OLD_TURN, {"role": "user", "content": "op"}, NEW_TURN,
                         {"role": "tool", "content": "out"}, NEW_TURN],
     [False, False, False, False, True, False, True]),
    # Qwen3's template does not count a user turn that wraps a tool response as a query.
    ("since_last_user", [*TASK, NEW_TURN,
                         {"role": "user", "content": "<tool_response>x</tool_response>"}],
     [False, False, True, False]),
    # The template tests for both tags anywhere in the turn, not for a wrapping pair.
    ("since_last_user", [*TASK, NEW_TURN,
                         {"role": "user", "content": "see <tool_response>x</tool_response> above"}],
     [False, False, True, False]),
    # With no query at all the template keeps no past reasoning.
    ("since_last_user", [{"role": "system", "content": "sys"}, NEW_TURN, NEW_TURN],
     [False, False, False]),
])
def test_prompt_view(mode, messages, kept):
    assert reasoning_kept(BudgetController(8192, reasoning_history=mode), messages) == kept


def test_prompt_view_leaves_messages_intact():
    messages = [*TASK, OLD_TURN, {"role": "user", "content": "op"}]
    BudgetController(8192, reasoning_history="since_last_user").prompt_view(messages)
    assert messages[2]["reasoning_content"] == THINK


def test_calibration_and_count_use_the_prompt_view():
    messages = [*TASK, OLD_TURN, {"role": "user", "content": "op"}]
    budget = BudgetController(8192, reasoning_history="since_last_user")
    served = tk.count_tokens([*TASK, {"role": "user", "content": "op"}])[0]
    budget.calibrate({"usage": {"prompt_tokens": served}}, messages)
    assert budget.tok_ratio == 1.0
    assert budget.count(messages) == served


def test_unknown_mode_refused():
    with pytest.raises(ValueError, match="reasoning_history"):
        BudgetController(8192, reasoning_history="none")
