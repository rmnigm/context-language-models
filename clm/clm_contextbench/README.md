# clm_contextbench

CLM on [ContextBench](https://github.com/RulinShao/ContextBench) under Harbor. `ClmStream` runs
the unmodified `ClmAgent` from [`clm_harness`](../clm_harness/) on ContextBench tasks;
`BaseStream` runs the same loop with no context mechanism (the paper's Base arm).

ContextBench pushes each op as a user message into one continuous conversation. The agent
signals that it is ready for the next op with a standalone `READY_FOR_NEXT_OP`. It cannot fetch
an op itself. `ClmStream` adds ContextBench's `CBStreamMixin` to `ClmAgent` and hooks three points
of its loop instead of copying it:

| hook | what it does |
|---|---|
| `run()` | loads the stream (checked against `task.toml`, then removed from the container), inserts the method skill into the instruction, puts ContextBench's marker contract in the system prompt's finish paragraph, then calls `ClmAgent.run()` |
| `self._resume.replay` (wrapped for one run) | keeps a reference to the message list the loop owns and appends op 0 as a user message |
| `_is_successful_submit` | called by the loop right after each tool result; appends what the stream releases (the next op, an explanation, or the early-submit refusal) as user messages, and ends the loop once the stream has a terminal reason |

`_query_with_retry` is wrapped only to record a server-side context overflow, so the run's
`terminated_by` can say `budget_wall`.

## Install

This package depends on ContextBench; ContextBench does not depend on CLM. Use a source checkout
of ContextBench, because the method skills live in its `clm_exp/skills/`:

```bash
pip install -e .                                   # this repository
git clone https://github.com/RulinShao/ContextBench.git
pip install -e ContextBench[tokenizer]
```

## Run

```bash
# build tasks (once), from the ContextBench checkout
contextbench package --all --config clm_exp/configs/paper.yaml --out-dir cb_tasks/

# one trial; the task grades itself, so no verifier flag is needed
OPENAI_API_KEY=EMPTY harbor trials start -p cb_tasks/kv-store-R1000-seed42 -e docker \
    -a clm_contextbench:ClmStream -m openai/qwen36-27b \
    --agent-kwarg api_base=http://localhost:8000/v1
```

`-e singularity` works too. Serve the model as described in
[`clm_harness`](../clm_harness/README.md#serving). The server's context window must hold
`context_budget_tokens + max_tokens` (32768 + 8192).

## Settings of the paper's runs

`ClmStream` starts from `PAPER_KWARGS`, the settings of the paper's CLM runs on ContextBench.
Any `--agent-kwarg` overrides one of them. Where the value differs from the `ClmAgent` default,
the adapter passes it so the default cannot silently apply:

| kwarg | paper value | `ClmAgent` default | what it means here |
|---|---|---|---|
| `context_budget_tokens` | 32768 | required | ContextBench's budget; the limit is 32768 - 2048 |
| `max_steps` | 100000 | 64 | no step limit in practice; the stream ends the run |
| `lm_call_cap` | 0 (off) | 2 x `max_steps` + 24 | no LM-call cap |
| `max_tokens` | 8192 | 16384 | per-reply output cap |
| `observation_max_chars` | 50000 | 10000 | a whole op batch fits in one tool output |
| `nudge_ratios` | "" (off) | 0.25,0.5,0.75 | no compaction reminders |
| `persistent_nudge_ratio` | none (off) | adaptive | no urgent reminders near the limit |
| `finalize_nudge_turns` | 0 (off) | 3 | no final-turn notices |
| `max_num_retry_on_limit` | 0 | 50 | no rollback-retry: over the limit, the model gets one final turn and the run ends at the budget wall |
| `cost_metric` | usd | auto | no FLOPs accounting |

These are pinned at their `ClmAgent` defaults: `context_budget_reserve_tokens=2048`,
`temperature=0.7`, `top_p=0.95`, `allow_edit_growth=fit`, `ctx_hint=default`,
`finish_policy=submit`, `enable_thinking=true`, `send_chat_template_kwargs=true`. The system prompt
and the task template (`terminal_agent_tasks`) are `ClmAgent`'s own.

The paper's runs gave each trial 21600 s. That is the task's `[agent] timeout_sec`; pass
`--agent-timeout-sec 21600` to `contextbench package` to match it.

## Method skill

By default (`skill=clm`) the adapter reads `clm_exp/skills/clm/<family>.md` from the ContextBench
checkout that `contextbench` is imported from. This is the method text of the paper's CLM runs.
The text must match `PAPER_SKILL_SHA16`, or the run refuses to start:

| family | sha256[:16] |
|---|---|
| kv_store | 70695bdca1a7f39d |
| log_triage | 1a071e820bcc9a93 |
| needle_retention | e834d24bf004c7ae |
| sudoku_sketchpad | 378c4e5d928af06f |

The skill goes where the paper's task builder put it: before the closing "The run ends
automatically ..." line (kv_store, log_triage, needle_retention), or before
`## Turn protocol (IMPORTANT)` (sudoku_sketchpad). With it, the full kv_store, log_triage and
needle_retention instructions hash to the paper's (`beff2f70192abfa5`, `753307b2a4dcc9e3`,
`d02f1364c0c179bb`). ContextBench's sudoku scaffold has one sentence that the paper's CLM sudoku
tasks did not; see ContextBench's `docs/DEVIATIONS.md`. If the task was packaged with the skill
already (`contextbench package --method-file ...`), the skill is not inserted twice.

Other values: `skill=none` runs without a skill; `skill=<name>` reads
`<skills_dir>/<name>/<family>.md`; an absolute path names a skill directory directly.
`skills_dir` defaults to ContextBench's `clm_exp/skills`. Only `skill=clm` is sha-checked.

## Preflight

Both adapters refuse to start when:

- the agent's `context_budget_tokens` / `context_budget_reserve_tokens` differ from the task's
  `contextbench_budget_tokens` / `contextbench_reserve_tokens` (its `task.toml`, next to the
  environment directory Harbor builds the container from): the wall and the grading assume the
  task's budget;
- tiktoken's `o200k_base` is unavailable (offline without `TIKTOKEN_CACHE_DIR`): the harness would
  silently count chars/4, so every wall and readout would be off.

## Base arm (`BaseStream`)

`-a clm_contextbench:BaseStream` is the paper's Base arm as a variant of `ClmStream`: the same loop,
`PAPER_KWARGS`, marker contract, budget wall, snapshots and outputs, with the context mechanism
removed:

| | `ClmStream` | `BaseStream` |
|---|---|---|
| mirror `/tmp/.live_ctx/LIVE_CTX_MAIN.txt` | written before, read back after every command | none |
| edits | synced with `parse_back` (and flattening) | none; the context only grows, every turn costs a step |
| system prompt, "## Managing your context" | goal + budget + mirror-file instructions | goal + budget sentence only |
| readout after each command | `[context: ~N/M tokens]`, over the limit `— OVER; compact <file> now` | same line, `— OVER` only |
| method skill | `clm/<family>.md` | the paper's base skill, see below |
| nudges, rollback-retry | off in `PAPER_KWARGS` | off and refused if set (their text tells the model to compact the mirror) |

`skill` picks the base skill from ContextBench's `clm_exp/skills/base/`, inserted at the same
position as the CLM skill and sha-checked against `BASE_SKILL_SHA16`:

- `paper` (default): the variant the paper's figure ran at the task's level, read from the task
  directory name (`<family>-<level>-seed<N>`): the detailed `<family>.md` at kv R8000/R16000,
  log L6400/L12800 and needle K2-C192/K2-C224; the short `<family>.short.md` at the other kv and
  log rungs and every sudoku rung; no skill at the other needle rungs. Other levels or a budget
  other than 32768 are refused: pass a variant.
- `detailed`, `short`, `none`: that variant everywhere (sudoku has no detailed skill).

`clm_stream.json` records `arm`, `level` and `base_skill` besides the usual fields.

What this does not reproduce: the paper's base arm was a separate mini-swe-agent-style
append-only harness with its own prompts (see ContextBench's `docs/HARBOR_SMOKE.md`), not
`ClmAgent` with the mirror removed. `BaseStream` keeps `ClmAgent`'s system prompt (minus the
mirror section), task template, budget controller and readout, so the two arms differ only in
the mechanism and the skill. The skill texts state the paper's 32,768-token budget; at other
budgets they are run unchanged.

## Outputs

The trial's `agent/` directory holds `ClmAgent`'s usual outputs (`trajectory.json`,
`usage.json`, `timing.json`, `trajectory.ctx.json`) and:

- `context_snapshots/turn-NNNN.json`: `ClmAgent`'s per-call snapshots of the context sent to
  the model. ContextBench's verifier grades these.
- `interactive_stream.json`: ops delivered and total, `terminated_by`, refusals, and which
  command released each op.
- `clm_stream.json`: the skill (name, path, sha16), the sha16 of the full instruction, and the
  agent kwargs.
