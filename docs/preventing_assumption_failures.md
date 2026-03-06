# Preventing Assumption-Driven Failures in Claude Code Experiments

## Incident: Full NBE Experiment Timeout Disaster (2026-03-05)

### What happened
User requested: "Run the full NBE experiment with all epochs (500). Save epochs per bucket per problem."

The executor created a script with a hardcoded **600-second (10 min) timeout per problem**. All 5 problems timed out. Zero usable data was produced. An entire day of compute was wasted.

### Why it happened
The executor made an assumption about runtime without:
1. Checking existing run data in the project (prior experiments clearly show runs taking 30+ minutes)
2. Asking the user how long to expect
3. Reasoning about the computation: 500 epochs x 33 NN buckets x (sampling + forward + backward pass) = clearly more than 10 minutes

The user's request said "full experiment" and "all the epochs." The word "full" should have been a signal that this was NOT a quick operation.

### What evidence existed in the codebase
- `num_epochs=500` is in every NBE config
- grid10x10 alone has 33 NN-eligible buckets (confirmed by quick-11)
- Prior experiment outputs in `nbe_eval_results/` show multi-phase evaluations taking significant time
- The hidden sizes script with `num_epochs=1` already took noticeable time

---

## Root Cause Analysis

The failure isn't a coding bug. It's a **decision-making failure**: the agent invented a constraint (600s timeout) that the user never requested, without checking whether it was reasonable, and without asking.

This maps to three systemic gaps in the current instruction files:

### Gap 1: No "uncertainty escalation" rule
The executor agent (`gsd-executor.md`) has deviation rules for code issues (bugs, missing functionality, blocking issues, architectural changes). But there is **no rule for operational uncertainty** — situations where the agent doesn't know a critical parameter (like expected runtime) and guesses instead of asking.

### Gap 2: No "check before you assume" mandate
Nothing in CLAUDE.md or the executor instructions says: "Before setting timeouts, resource limits, or other operational parameters, check existing data in the project or ask the user." The agent invents values from nothing.

### Gap 3: Notification instructions are incomplete
CLAUDE.md says to ping Discord when "waiting for user input" but doesn't explicitly say: "Ping when you're about to make a significant assumption that could waste resources if wrong."

---

## Recommended Changes

### 1. Add to CLAUDE.md: Experiment Execution Rules

Add a new section to the project CLAUDE.md:

```markdown
## Experiment Execution Rules

**NEVER invent timeouts, resource limits, or runtime assumptions.** If you don't know how long
an experiment will take:
1. Check existing run data in the project (notebooks/, nbe_eval_results/, lab_notebook.txt)
2. Estimate from the computation: num_epochs x num_buckets x per-epoch cost
3. If still uncertain, ASK the user via Discord ping before running
4. When in doubt, run WITHOUT a timeout rather than with an arbitrary one

**"Full" experiments mean full.** When the user says "run the full experiment" or "all epochs",
do not add timeouts, reduce epochs, or truncate in any way unless explicitly told to.

**Before running any experiment that trains neural networks:**
- Check the config's num_epochs value
- Count NN-eligible buckets (use get_large_message_buckets or prior run data)
- If total training iterations > 10,000, warn the user about expected runtime
- Run long experiments in background with no timeout; ping Discord when done
```

### 2. Add to CLAUDE.md: Assumption Escalation Protocol

```markdown
## Assumption Escalation

When you are about to hardcode a value that the user did not specify (timeout, batch size,
number of iterations, resource limits, etc.):

1. **Is this value observable?** Check existing data, configs, or prior runs.
2. **Is this value derivable?** Can you compute a reasonable value from known quantities?
3. **If neither:** STOP and ask the user. Ping Discord:
   `~/.claude/ai-ops/scripts/ping_nick.sh "Need input: [what you need to know and why]"`

**Never** silently invent operational parameters. A missing timeout is better than a wrong one.
The cost of asking is minutes. The cost of a wrong assumption is hours or days.
```

### 3. Add to gsd-executor.md: Deviation Rule 5

Add after Rule 4 in the `<deviation_rules>` section:

```markdown
**RULE 5: Escalate operational uncertainty**

**Trigger:** About to set a runtime parameter (timeout, resource limit, iteration count,
batch size) that the user didn't specify and you can't derive from existing project data.

**Examples:** Setting subprocess timeouts, choosing GPU memory limits, deciding how many
epochs to run, setting retry counts, choosing polling intervals for long-running processes.

**Action:** STOP → check existing run data in the project. If still uncertain → ask the user
(Discord ping or checkpoint). NEVER invent operational parameters from nothing.

**Rationale:** Wrong code is fixable in minutes. Wrong operational parameters waste hours
or days of compute and human time.
```

### 4. Add to gsd-planner.md: Runtime Awareness

Add to the planner's constraints or planning context:

```markdown
**Runtime awareness for experiment tasks:**
When planning tasks that run experiments or train models:
- Include estimated runtime in the task description
- If runtime is unknown, add a pre-task: "Estimate runtime from existing data or ask user"
- Never plan a task with a hardcoded timeout unless the user specified one
- For long-running tasks, plan for background execution with Discord notification on completion
```

### 5. Update global CLAUDE.md: Strengthen notification rules

Change the notification section from:
```
When you complete a significant task or need my input, ping Nick on Discord
```

To:
```
Ping Nick on Discord when:
- You complete a significant task
- You need input or have a question
- You're about to make an assumption that could waste resources if wrong
- An experiment will take longer than 5 minutes
- You encounter an error during a long-running process

When in doubt, ping. The cost of an unnecessary ping is zero.
The cost of a missed ping can be hours of wasted compute.
```

---

## Summary of Changes by File

| File | Change | Purpose |
|------|--------|---------|
| `CLAUDE.md` (project) | Add "Experiment Execution Rules" section | Prevent arbitrary timeouts on experiments |
| `CLAUDE.md` (project) | Add "Assumption Escalation Protocol" section | Force ask-before-assume behavior |
| `CLAUDE.md` (global) | Strengthen notification triggers | Lower the bar for pinging user |
| `gsd-executor.md` | Add Deviation Rule 5: operational uncertainty | Catch runtime parameter guessing |
| `gsd-planner.md` | Add runtime awareness for experiment tasks | Prevent bad plans from being created |

## Key Principle

**The cost hierarchy is: asking < wrong assumption < wasted compute day.**

An agent that asks too many questions is mildly annoying. An agent that silently invents parameters and wastes a day of GPU/CPU time is catastrophic. The instructions should encode this asymmetry explicitly.
