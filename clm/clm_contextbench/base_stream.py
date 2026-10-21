"""BaseStream: the paper's Base arm on ContextBench, as a no-edit variant of ``ClmStream``.

Same loop, budget wall, marker contract, snapshots and outputs, with no context mirror, no
edit path and the paper's base skill; README.md has the arm-by-arm comparison.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from clm_harness.context_env.env import ContextEnv
from clm_harness.context_env.types import StepResult

from .clm_stream import ClmStream, _default_skills_dir, family_of, place_skill
from .preflight import level_of

__all__ = ["BASE_SKILL_SHA16", "BaseStream", "paper_base_skill"]

#: sha256[:16] of ContextBench's ``clm_exp/skills/base/<file>``, the paper's base skills.
BASE_SKILL_SHA16: dict[str, str] = {
    "kv_store.md": "a0b9d4b54b496426",
    "log_triage.md": "5c68cf8a937003b4",
    "needle_retention.md": "f630bb9c68e77145",
    "kv_store.short.md": "58ba7d944434646f",
    "log_triage.short.md": "52bead31380b399b",
    "sudoku_sketchpad.short.md": "418fc80559b8f2c1",
}

#: The base skill each level of the paper's figure ran with (clm_exp/skills/README.md):
#: the detailed ``<family>.md`` on the hardest rungs, the short ``<family>.short.md`` on
#: the others, and no skill for needle_retention below K2-C192.
_PAPER_BASE_SKILL: dict[str, dict[str, str]] = {
    "kv_store": {**dict.fromkeys(("R130", "R260", "R520", "R1000", "R2000", "R4000"), "short"),
                 "R8000": "detailed", "R16000": "detailed"},
    "log_triage": {**dict.fromkeys(("L160", "L320", "L640", "L1600", "L3200"), "short"),
                   "L6400": "detailed", "L12800": "detailed"},
    "needle_retention": {**dict.fromkeys(("K4-C4", "K4-C6", "K4-C8", "K4-C24", "K4-C96",
                                          "K8-C48", "K8-C60", "K8-C62"), "none"),
                         "K2-C192": "detailed", "K2-C224": "detailed"},
    "sudoku_sketchpad": dict.fromkeys(("t4", "t8", "t15", "t30", "t60", "t100", "t150"), "short"),
}
_PAPER_BUDGET = 32768
_SKILL_SUFFIX = {"detailed": ".md", "short": ".short.md"}
_MIRROR_SECTION = "Your conversation is mirrored to"


def paper_base_skill(family: str, level: str | None, budget: int) -> str:
    """``detailed``, ``short`` or ``none``: the base skill of this level in the paper's figure."""
    choice = _PAPER_BASE_SKILL.get(family, {}).get(level or "")
    if budget != _PAPER_BUDGET or choice is None:
        raise ValueError(
            f"{family} level {level} at budget {budget} is not a level of the paper's figure; "
            "pass skill=short|none, or skill=detailed at budget 32768")
    return choice


class _NoMirrorEnv(ContextEnv):
    """``ContextEnv`` without the mirror: run the command, cap the output, add the readout."""

    async def setup_dirs(self, environment: Any) -> None:
        if self.persistent_bash:
            p = self.paths
            await environment.exec(
                command=f"mkdir -p {p.state_dir} && pwd > {p.state_dir}/cwd && "
                        f"export -p > {p.state_dir}/env", timeout_sec=10)

    async def step(self, command: str, messages: list[dict[str, Any]], *, environment: Any,
                   pending: dict[str, Any] | None = None) -> StepResult:
        t0 = time.monotonic()
        result = await self.execute(command, environment)
        exec_time = time.monotonic() - t0
        stdout_block = self.append_submit_warning(command, self.format_tool_result(result))
        readout = ""
        if self.context_budget_tokens:
            shown = messages + ([pending] if pending else [])
            stdout_block = self.budget.cap_newest_output(shown, stdout_block)
            tokens = self.budget.count(shown + [{"role": "tool", "content": stdout_block}])
            limit = self.budget.strict_target or self.context_budget_tokens
            over = " — OVER" if tokens > (self.budget.strict_target or 0) else ""
            readout = f"\n[context: ~{tokens}/{limit} tokens{over}]"
        return StepResult(result=result, ctx_changed=False, stdout_block=stdout_block,
                          readout=readout, notes="", exec_time=exec_time, touched_ctx=False)


class BaseStream(ClmStream):
    """``ClmStream`` with no context mirror and no edit path, and the paper's base skill.

    ``skill``: ``paper`` (default: the variant the paper ran at this level; only for the
    paper's levels at its 32768 budget), ``detailed`` (only at 32768), ``short`` or ``none``.
    """

    def __init__(self, *args: Any, skill: str | None = "paper", skills_dir: str | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, skill="base", skills_dir=skills_dir, **kwargs)
        self.base_skill = str(skill or "none").lower()
        if self.base_skill not in ("paper", "none", *_SKILL_SUFFIX):
            raise ValueError(f"skill must be paper|detailed|short|none, got {skill!r}")
        if self.nudge_ratios or self.persistent_nudge_ratio is not None \
                or self.max_num_retry_on_limit:
            raise ValueError("BaseStream has no context mirror: nudge_ratios, "
                             "persistent_nudge_ratio and max_num_retry_on_limit must be off")
        # Same state, mirror-free step: the loop and the readout keep using self._ctx.
        self._ctx.__class__ = _NoMirrorEnv
        self._cb_record["arm"] = "base"
        self._cb_level: str | None = None

    @staticmethod
    def name() -> str:
        return "base-contextbench"

    def _with_skill(self, instruction: str) -> str:
        family = family_of(instruction)
        if family is None:
            raise ValueError("cannot tell the ContextBench family from the instruction")
        variant = self.base_skill
        if variant == "paper":
            variant = paper_base_skill(family, self._cb_level, self.context_budget_tokens)
        if variant == "detailed" and self.context_budget_tokens != _PAPER_BUDGET:
            raise ValueError(
                "the detailed base skills state the paper's 32,768 budget and its batch sizes; "
                f"at budget {self.context_budget_tokens} pass skill=short|none")
        self._cb_record.update(level=self._cb_level, base_skill=variant)
        if variant == "none":
            return instruction
        name = f"{family}{_SKILL_SUFFIX[variant]}"
        path = Path(self.cb_skills_dir or _default_skills_dir()) / "base" / name
        text = path.read_text(encoding="utf-8")
        sha16 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if sha16 != BASE_SKILL_SHA16[name]:
            raise ValueError(f"{path} has sha16 {sha16}, expected {BASE_SKILL_SHA16[name]}")
        self._cb_record.update(skill_path=str(path), skill_sha16=sha16)
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        if first and first in instruction:
            return instruction  # the task was packaged with the skill already
        return place_skill(instruction, text, family)

    def _base_system(self, system: str) -> str:
        """The system prompt without the mirror-file instructions."""
        start = system.find(_MIRROR_SECTION)
        end = system.find(self.finish_policy.prompt_fragment())
        if start < 0 or end < start:
            raise ValueError("ClmAgent's system prompt changed: cannot cut the mirror section")
        return system[:start] + system[end:]

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        self._cb_level = level_of(environment)
        replay = self._resume.replay

        def _replay(messages: list[dict[str, Any]], *a: Any, **kw: Any) -> list[dict[str, Any]]:
            messages[0] = {**messages[0], "content": self._base_system(messages[0]["content"])}
            return replay(messages, *a, **kw)

        self._resume.replay = _replay
        try:
            await super().run(instruction, environment, context)
        finally:
            self._resume.replay = replay
