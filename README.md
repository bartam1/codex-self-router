# codex-self-router

`codex-self-router` is a local WebSocket proxy for Codex. It lets a Codex agent request a model
and/or reasoning-effort change during an active turn under a configurable approval policy, and it
supports explicit per-instruction route directives without an extra approval round-trip.

The first release is intentionally a standalone proxy instead of a Codex fork:

```text
Codex TUI  <-- WebSocket -->  codex-self-router  <-- JSONL/stdin -->  codex app-server
```

It uses Codex app-server's experimental dynamic tools and `turn/settings/update` API. See the
[official Codex app-server documentation](https://learn.chatgpt.com/docs/app-server).

## Profiles and explicit directives

In a fresh thread, every new, unmarked instruction starts on Sol. Resumed threads keep their last
confirmed profile until explicitly changed or switched through the routing tool.
Put one of these markers at the beginning or as the
final standalone token of an instruction to choose its model and reasoning effort without an
approval dialog. The suffix means `1=low`, `2=medium`, and `3=xhigh`:

| Prefix | Profile | Model | Low | Medium | XHigh |
|---|---|---|---|---|---|
| `l` | Luna | `gpt-5.6-luna` | `l1` | `l2` | `l3` |
| `t` | Terra | `gpt-5.6-terra` | `t1` | `t2` | `t3` |
| `s` | Sol | `gpt-5.6-sol` | `s1` | `s2` | `s3` |
| `a` | Astra | `gpt-6-astra` | `a1` | `a2` | `a3` |

Examples:

```text
l1 collect the relevant files and summarize them
```

```text
a3 design the concurrency boundary and implement it
```

```text
create another directory t1
```

Markers in the middle of an instruction are ordinary text. The router removes an exact recognized
leading or trailing token before sending the instruction to the model. Legacy `#1`, `#2`, and `#3`
remain aliases for Luna/medium, Sol/medium, and Astra/xhigh respectively.

These markers bypass approval because they are explicit user authorization. Model switches
requested by the agent through `self_router.request_model_switch` follow the configured
`agent_switch_approval` policy. The agent can change only effort while keeping the model.
A marker also works when the instruction steers an already-running turn: the router applies
`turn/settings/update` before forwarding the cleaned steering message.

## Configuration

The default user configuration is `~/.config/codex-self-router/config.yaml`. It configures the
agent-switch approval policy, model IDs, default and allowed efforts, directive prefixes and
levels, legacy aliases, descriptions used by the routing policy, and the API-equivalent prices
captured in new reports. See
`config.example.yaml`. Use a different file with the global option
`--config /path/to/config.yaml`, before the subcommand. Create the defaults on another machine with:

```sh
codex-self-router init-config
```

Configuration is validated at startup. Duplicate prefixes, unsupported configured effort mappings,
missing profiles, malformed YAML, negative prices, and unknown defaults fail closed. The running
app-server catalog is also checked and warns if a configured model or effort is not advertised.

Choose how agent-requested switches are authorized:

```yaml
agent_switch_approval: never
```

- `always` prompts for every effective agent-requested change and is the default.
- `upgrades_only` automatically permits changes that do not increase any configured model price
  or reasoning-effort level, and prompts for upgrades.
- `never` automatically permits every valid model/effort change.

Override the configured value for one run with `--switch-approval`. This controls only the
self-router's model/effort prompt; normal Codex command permissions remain controlled by Codex:

```sh
codex-self-router run --switch-approval never -- \
  --sandbox danger-full-access --ask-for-approval on-request
```

The complete model-visible routing instruction is the `routing_policy_template: |` block in YAML,
not hidden application text. Edit it to tune routing behavior. The router expands these optional
placeholders when a thread is created or resumed:

- `{{approval_mode}}` — the active `agent_switch_approval` value.
- `{{profiles}}` — configured model IDs and profile descriptions.
- `{{directives}}` — the active explicit-route markers.

Unknown placeholders fail configuration validation. Older configuration files without a template
continue to use the documented built-in default. The default template distinguishes difficult
decisions from mechanical execution, routes routine commands and data collection toward Luna,
routes ordinary PR drafting toward Terra, retains Sol/Astra for substantive review claims, and
states explicitly that model-switch authorization never authorizes commands or external actions.

The namespace exposes a read-only `self_router.get_current_route` tool returning the exact active
profile, model ID, reasoning effort, and switch-approval policy without changing models or granting
permission for commands. Route state is returned as tool output rather than appended to user input,
so it does not alter or visibly duplicate the user's message.

## Requirements

- Python 3.11 or newer
- A Codex CLI/app-server build at version 0.154.0 or newer
- Access to the configured Luna, Terra, Sol, and Astra models
- `step_model_switching`, which the router enables on the app-server subprocess

The app-server APIs used here are experimental and can change between Codex releases. The router
refuses older binaries by default instead of failing halfway through a session.

## Install and run

With `uv`:

```sh
uv tool install .
codex-self-router doctor
codex-self-router run
```

Pass normal Codex options after `--`:

```sh
codex-self-router run -- --full-auto
```

If Codex is not on `PATH`, point to it explicitly:

```sh
codex-self-router --codex-bin /path/to/codex run
```

To run the proxy separately:

```sh
codex-self-router serve --listen 127.0.0.1:4501
codex --remote ws://127.0.0.1:4501
```

### Resume a router session

```sh
codex-self-router run -- resume THREAD_ID
```

Version 0.4 supports explicit resume of durable local threads created through the router. It
restores the last confirmed profile and reasoning effort, restores routing instructions, and
verifies that Codex saved the routing tool. Explicit overrides and route directives still work.
Use the same `--report-dir` as before for checkpoint and measurement continuity. Fixed-profile
control runs must resume with the same `--fixed-profile` setting.

New session reports link to earlier reports for that thread. Historical token usage is not billed
again; new response-level usage is read from Codex's local rollout because cold-resumed app-server
threads do not expose raw response events in Codex 0.154.0. Nested code-mode calls have no exposed
parent-response ID, so their decision-response association remains unknown rather than guessed.
An unfinished measured task carries its ID into the first resumed turn; aggregate reports count
these fragments once and do not treat offline time as measured task latency. Previously completed
tasks remain separate. Old reports retain their own pricing snapshots.

Pending approvals and tool calls are **not** replayed. Resume restores conversation and routing
state; it does not recover a running process or automatically continue unfinished work. Automatic
reconnect and `thread/fork` are not supported. Ordinary non-router threads cannot gain the routing
tool through resume. Older router reports provide best-effort profile restoration, but missing
historical measurements are not reconstructed. Two router connections using the same report
directory cannot own the same thread simultaneously.

Codex 0.154.0 restores a thread's original dynamic-tool schema and cannot replace it during resume.
Threads first created by router 0.3 can still resume and use model routing and all explicit 0.4
directives, but agent-requested effort-only routing requires a thread created by router 0.4.
Likewise, pre-0.5 threads do not gain the `get_current_route` tool after resume; start a new thread
when that on-demand tool is required.

## Reporting

The router records each raw upstream response under the model active for that sampling step and
writes a JSON report after every observed usage event or route decision. Show the latest report:

```sh
codex-self-router report
codex-self-router report --json
```

Reports contain:

- explicit directives and agent-requested switches;
- approved, denied, and failed outcomes;
- response-level token usage and model attribution;
- an API-equivalent routed cost estimate;
- comparisons with the same observed tokens priced as all-Sol and all-Astra;
- the exact per-million-token pricing snapshot used by that report.

The estimates are not ChatGPT subscription charges and are not true counterfactual runs. A
different model can take a different number of steps and tokens.

On macOS reports default to:

```text
~/Library/Application Support/codex-self-router/reports
```

Use the global `--report-dir` option to choose a different location.

### Measuring whether routing pays off

Version 0.2 writes schema-version-2 session reports and an append-only
`<session-id>.events.jsonl` journal. New measurements are collected automatically on restart;
older reports remain readable, but missing history is not reconstructed or invented.

The reports now retain:

- Router and Codex versions, profile/effort configuration, routing-policy hash, run label, and a
  pricing snapshot captured at session creation.
- Task and turn boundaries. Automatic continuation turns remain part of the original task;
  subsequent user turns start new tasks. Router interruptions and user interruptions are distinct.
- For agent-requested switches: source and destination model/effort, approval policy and
  authorization source, requested/applied times, approval wait, usage-event wait, apply time,
  mechanism, requesting response (when identifiable), and outcome, including
  declined/no-op/cancelled attempts.
- Observed response counts after each applied route, until the next route or end of the task.
  Closed phases show the difference from the agent's estimated step count; unfinished phases are
  marked open. The response requesting a switch belongs to the previous model.
- Input, cached input, cache writes, output, and reasoning tokens as reported by the server.
  Reasoning is a subset of output and is not billed twice. Missing/invalid usage or untracked
  models are explicitly counted as unpriced, never treated as known zero-cost responses.
- Task wall time from the observed request/start, time to the first completed response, blocking
  approval/question wait time (overlapping intervals counted once), tool durations and exit/status
  signals, upstream errors with announced retries, user steering events, and context compactions.

Exact inference duration and internal reasoning content are not exposed by this protocol.
Wall time excluding user waits still includes tools, network delays, and router work. Model names
are attributed from router state; the usage event does not independently attest which model ran.
Internal subagents without tracked routing state remain unpriced. Tool failures are signals for
review, not proof of task failure, and a completed turn is not automatically rated successful.

Inspect a session, its task IDs, and measured phases:

```sh
codex-self-router report
codex-self-router report --session SESSION_ID --json
codex-self-router report --all
codex-self-router report --all --label coding-benchmark --json
```

To compare routing against real fixed-model runs, use the same label and comparable tasks, initial
repository state, permissions, tools, and acceptance criteria:

```sh
codex-self-router run --label coding-benchmark
codex-self-router run --label coding-benchmark --fixed-profile sol
codex-self-router run --label coding-benchmark --fixed-profile astra
```

These options also work with `serve`. Put router options before `--` and Codex options after it.
Fixed-profile runs enforce the selected profile on proxy-managed turns, omit the router tool and
routing instructions, and reject conflicting model directives/settings. They do not replay tasks
automatically. Repeat representative tasks and vary run order to reduce cache/order bias.

After checking the result, record an explicit assessment using task IDs from `report`:

```sh
codex-self-router feedback --session SESSION_ID --task TASK_ID \
  --outcome success --tests passed
```

Outcomes are `success`, `rework`, or `failed`; tests are `passed`, `failed`, `not-run`, or `unknown`.
An optional `--note` records your assessment. Feedback is stored separately so a running router
cannot overwrite it. Unrated tasks remain unknown. Group reports show sample counts, cost coverage,
quality feedback, retries, short phases, estimate error, timing, and policy versions. JSON contains
the detailed metrics. Historical sessions retain their original prices when aggregated.

Same-token repricing remains a hypothetical comparison: another model could use different tokens,
cache hits, or steps. Fixed-model cohorts are observed comparisons, but different task mixes and
unrated results can still bias them. No automatic causal savings claim or confidence score is made.

The new event journal stores selected protocol metadata, not prompt text, command arguments/output,
or reasoning text. Existing switch reasons/next actions, upstream usage metadata, and explicit
feedback notes are still saved locally. A journal survives between snapshot saves; a hard crash
may leave an incomplete session, which is excluded from closed-task timing statistics.
Resume checkpoints in the JSON reports also retain developer instructions and collaboration-mode
settings so they can be restored. Treat the report directory as private session data.

## Routing behavior

- Luna is suggested for straightforward, low-risk, repetitive, mechanical, or read-only work,
  including short tasks that need a tool call followed by another model response.
- Terra is the balanced everyday production profile.
- Sol is the default for advanced coding, debugging, and ambiguous work.
- Astra is suggested for architecture, difficult ambiguity, sensitive decisions, or repeated
  Sol failures.
- Low effort is for well-bounded execution, medium for normal implementation and analysis, and
  xhigh for genuinely difficult or consequential reasoning. Prefer changing only effort when the
  active model is otherwise appropriate.
- The agent is expected to assess each substantial upcoming phase and proactively request the
  appropriate profile. Authorization follows `agent_switch_approval`; a possible approval prompt
  is not a reason for the agent to avoid requesting a switch.
- Difficult planning and architecture should trigger an Astra request even when the work is
  read-only. Complexity and risk take precedence over the generic "planning" label.
- At a new instruction or phase boundary, the agent should downgrade when the active profile is
  more capable than the upcoming work needs. Routing decisions count expected model inference
  steps: a single shell action commonly requires one response to invoke the tool and another to
  process its result, so it can still justify moving from Astra to Luna.
- Exact same-model/same-effort tool requests are no-ops and do not ask for approval.
- Only one agent-requested switch is processed at a time for a turn.
- Other tool calls from the same model response remain independent.
- The response that requested a switch is attributed to its original model before the settings
  update is applied; the tool result then releases the next inference step on the new model.
- Codex's TUI can keep displaying and resending the model/effort it selected before an internal
  router switch. The router tracks that client-side selection separately and ignores an unchanged
  stale echo, so a later user turn or interruption does not undo the routed model. A genuinely
  changed TUI selection is still accepted as an intentional client override. The TUI header may
  remain cosmetically stale; the router emits an immediate active-route notice during the turn,
  and `self_router.get_current_route` remains authoritative.
- Codex rejects some in-turn switches when the two models require different safety settings. For
  those combinations, including some GPT-5.6-to-Astra transitions in Codex 0.154.0, the router interrupts the
  current turn and immediately starts a model-switched continuation turn in the same thread. It
  passes the approved next action as standalone tool output, preserving conversation context while
  letting the destination model start with its required safety configuration. The continuation
  explicitly tells the destination model that this mechanical boundary is not a new request or a
  user cancellation, and that it should make reasonable assumptions rather than queue optional
  clarification questions merely because the turn changed. As a deterministic fallback, the
  router suppresses non-blocking asynchronous question items during that automatic continuation;
  ordinary blocking questions are still forwarded to the user.

Prompt caches are owned by the upstream service. The router keeps the stable instruction and tool
prefix unchanged, but a cache hit after returning to a model is never guaranteed.

## Development

```sh
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
```

The unit suite covers directive safety, profile injection, dynamic-tool merging, approval/no-op
state transitions, model×effort directives, YAML validation, catalog validation, usage attribution
ordering, cost calculations, and resume checkpoints, locking, tool persistence, and measurement
deduplication.

An optional live test creates its own read-only thread and exercises Luna/medium → Astra/xhigh →
app-server restart → resume Astra/xhigh → approved switch back to Luna/medium. It uses real
inference and approves only its own model-switch request:

```sh
ROUTER_LIVE_CODEX=/path/to/codex uv run --extra dev pytest tests/test_resume_live.py -s
```
