"""Checks before a ContextBench trial starts: the task's budget and the o200k counter.

The task is read from its ``task.toml`` (written by ``contextbench package``): the agent's
environment is built from ``<task_dir>/environment``, so the file sits next to it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from clm_harness.utils import tokens as tk

_TASK_NAME = re.compile(
    r"^(?:kv-store|log-triage|needle-retention|sudoku-sketchpad)-(.+)-seed\d+$")


def task_dir(environment: Any) -> Path:
    return Path(environment.environment_dir).parent


def task_metadata(environment: Any) -> dict[str, Any]:
    """``[metadata]`` of the trial's ``task.toml``."""
    with (task_dir(environment) / "task.toml").open("rb") as fh:
        return tomllib.load(fh).get("metadata", {})


def level_of(environment: Any) -> str | None:
    """The level (``R8000``, ``K2-C192``, ``t150``) of a ``<family>-<level>-seed<N>`` task dir."""
    m = _TASK_NAME.match(task_dir(environment).name)
    return m.group(1) if m else None


def check_budget(metadata: dict[str, Any], budget: int, reserve: int) -> None:
    """Refuse an agent budget/reserve other than the task's: the wall and grading assume it."""
    want = (metadata.get("contextbench_budget_tokens"),
            metadata.get("contextbench_reserve_tokens"))
    if want != (budget, reserve):
        raise ValueError(
            f"agent context_budget_tokens/reserve {budget}/{reserve} != the task's "
            f"contextbench_budget_tokens/reserve {want[0]}/{want[1]} (task.toml); pass "
            f"--agent-kwarg context_budget_tokens={want[0]} "
            f"--agent-kwarg context_budget_reserve_tokens={want[1]}")


def check_tokenizer() -> None:
    """Refuse to run without the o200k encoder: the harness would count chars/4 instead, and
    the budget wall and every readout would be off (set ``TIKTOKEN_CACHE_DIR`` offline)."""
    if tk.get_encoder() is None:
        raise RuntimeError("tiktoken o200k_base is unavailable (offline without "
                           "TIKTOKEN_CACHE_DIR?); refusing to count tokens as chars/4")
