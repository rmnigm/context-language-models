import pytest

from clm_harness.clm_agent.harness import _is_context_overflow


@pytest.mark.parametrize("text", [
    # llama.cpp llama-server
    'Error code: 400 - {"error": {"code": 400, "message": "request (17012 tokens) exceeds the '
    'available context size (16384 tokens), try increasing it", "type": "exceed_context_size_error"}}',
    "input (17012 tokens) is larger than the max context size (16384 tokens). skipping",
    "Context size has been exceeded.",
    # mistral.rs
    "Prompt sequence length is greater than 16384, perhaps consider using `truncate_sequence`?",
    # existing OpenAI/vLLM phrasing
    "This model's maximum context length is 32768 tokens",
])
def test_overflow_detected(text):
    assert _is_context_overflow(RuntimeError(text))


@pytest.mark.parametrize("text", ["Connection refused", "502 Bad Gateway"])
def test_other_errors_not_overflow(text):
    assert not _is_context_overflow(RuntimeError(text))
