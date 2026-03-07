# Retrospective: Reasoning Errors in NBE Experiment Tasks

**Date:** 2026-03-07
**Scope:** Quick tasks 10-12 and follow-up fixes (hidden sizes script, full NBE experiment)
**Purpose:** Honest catalog of reasoning failures for ai-ops CEO review and protocol improvement

---

## Error 1: Overriding device to CPU when configs specify CUDA

**What happened:** The experiment script hardcoded `config['device'] = 'cpu'` and `device='cpu'` in the FastGM constructor, despite all 5 NBE configs specifying `device='cuda'` and 4 Titan RTX GPUs being available on the machine.

**Why it happened:** The earlier hidden-sizes script (quick-10) used CPU because it was a quick diagnostic (1 epoch). When the full experiment script was created, this pattern was cargo-culted forward without questioning whether CPU was appropriate for 500-epoch training.

**Root reasoning error:** Copying a parameter from a different context without asking "does this still apply?" The diagnostic script and the full experiment have fundamentally different compute requirements, but the agent treated them as the same.

**Principle violated:** Don't override user-specified or config-specified values unless you have a specific reason. The config IS the user's intent.

---

## Error 2: Inventing a 600-second timeout

**What happened:** The executor created a subprocess-per-problem architecture with a 600-second (10 min) timeout. All 5 problems timed out. Zero usable data was produced. A full day of compute was wasted.

**Why it happened:** The agent wanted to be "safe" by preventing runaway processes. But it chose an arbitrary number (600s) without:
- Checking `num_epochs=500` in the config
- Counting NN-eligible buckets (33 for grid10x10 alone)
- Looking at any prior run timing data
- Asking the user

**Root reasoning error:** Solving an imagined problem (runaway process) while creating a real one (premature termination). The user said "full experiment" and "all epochs" — adding a timeout directly contradicts the request.

**Principle violated:** When the user says "full," they mean full. Don't add constraints they didn't ask for. If you're worried about runaway processes, ask — don't silently truncate.

---

## Error 3: Not checking existing evidence before making assumptions

**What happened:** The project contains abundant data about experiment runtimes:
- `num_epochs=500` is in every config
- Quick-11 confirmed 33 NN-eligible buckets for grid10x10
- Prior experiment outputs exist in `nbe_eval_results/`
- The hidden-sizes script with `num_epochs=1` already took noticeable wall-clock time

None of this was consulted before setting the 600s timeout or the CPU device.

**Root reasoning error:** Acting on assumptions instead of observations. The agent had all the information needed to make a good decision but didn't look at it. This is the difference between "I don't know" (which is fine — ask) and "I didn't bother to check" (which is not).

**Principle violated:** Observation before assumption. Check existing data first, compute estimates second, ask the user third. Never guess.

---

## Error 4: Not asking the user when uncertain

**What happened:** CLAUDE.md explicitly says to ping Discord when uncertain or waiting for input. The agent was uncertain about runtime, device, and timeout values but never pinged.

**Root reasoning error:** Overconfidence. The agent assumed its guesses were good enough. It treated operational parameters (timeout, device) as low-stakes decisions when they're actually high-stakes — wrong values waste hours or days.

**Principle violated:** The cost hierarchy is: asking (minutes) < wrong assumption (hours/days). The instructions should encode this asymmetry, and the agent should internalize it.

---

## Error 5: Pre-elimination bucket width measurement (Quick-10)

**What happened:** The first hidden-sizes script checked bucket widths BEFORE variable elimination. At that point, buckets only contain their initial factors (max width 4 for grid10x10). During elimination, messages from earlier buckets arrive and grow later buckets to the induced width (21). The script reported 0 NN-eligible buckets — completely wrong.

**Why it happened:** The agent didn't understand the execution flow deeply enough. It knew `get_width()` returns message scope size, but didn't reason about WHEN in the algorithm's lifecycle the buckets have their final scope.

**Root reasoning error:** Surface-level understanding of the algorithm. The agent read the method signatures but didn't trace the data flow. Variable elimination is an iterative process — bucket state changes as you go.

**Principle violated:** Understand the algorithm before instrumenting it. "When does this data exist?" is a critical question for any measurement.

---

## Error 6: Python dict iteration bug

**What happened:** The worker script had `for bucket in fastgm.buckets` which iterates over dictionary keys (variables), not bucket objects. Should have been `for var, bucket in fastgm.buckets.items()`.

**Root reasoning error:** Basic Python mistake made under the assumption that buckets is a list. The agent didn't check the type of `fastgm.buckets` before iterating.

**Principle violated:** Verify data structure types before writing iteration code. A 2-second `type()` check would have caught this.

---

## Error 7: Leaving zombie processes

**What happened:** The first experiment launch (with 600s timeout) left worker subprocesses alive even after the parent timed out. A second launch was started without killing the first. Both ran simultaneously on CPU, wasting resources.

**Root reasoning error:** Not cleaning up before retrying. When relaunching an experiment, always check for and kill previous instances.

**Principle violated:** Clean slate before retry. Check `ps aux | grep` for related processes before launching new ones.

---

## General Principles Extracted

### 1. Config values are user intent — don't override without reason
If a config says `device='cuda'`, the user chose that. Overriding it to `cpu` requires an explicit reason and explicit user approval. Same for `num_epochs`, `ecl`, `iB`, etc.

### 2. "Full" means full — don't add invisible constraints
When the user says "run the full experiment," adding a timeout, reducing epochs, or truncating in any way directly contradicts the request. If you're worried about something, surface the concern — don't silently "fix" it.

### 3. Observation > Estimation > Asking > Guessing
Before setting any operational parameter:
1. **Observe:** Check existing data, configs, prior runs
2. **Estimate:** Compute from known quantities (epochs x buckets x per-epoch time)
3. **Ask:** Ping the user if still uncertain
4. **Never:** Guess and hope it's right

### 4. The cost of asking is always lower than the cost of a wrong assumption
A Discord ping takes 10 seconds. A wrong timeout wastes a day. A wrong device wastes two days. The math is overwhelmingly in favor of asking.

### 5. Understand the algorithm before measuring it
Surface-level API knowledge ("call get_width()") is not sufficient. You need to understand the algorithm's lifecycle: when data is populated, when it changes, when it's valid to read.

### 6. Check types and data structures before writing code
A `dict` is not a `list`. A bucket before elimination is not the same as a bucket during elimination. These are basic verification steps that prevent entire classes of bugs.

### 7. Clean up before retrying
Before launching a new experiment run, always:
- Check for existing processes (`ps aux | grep`)
- Kill stale workers
- Verify GPU memory is available (`nvidia-smi`)

---

## Recommendations for ai-ops Protocol Updates

1. **Pre-flight checklist for experiments:** Before any experiment launch, require: device check, runtime estimate, process cleanup, user confirmation for runs > 5 min.

2. **"Ask threshold" calibration:** Any operational parameter not specified by the user AND not derivable from existing data should trigger an ask. The threshold should be very low.

3. **Config fidelity rule:** Config values should be treated as user decisions. Overriding them requires explicit justification logged in the summary.

4. **Zombie process prevention:** Any workflow that launches subprocesses should include cleanup of previous instances as a first step.

5. **Algorithm literacy requirement:** Before instrumenting or modifying algorithm internals, the agent should demonstrate understanding of the execution lifecycle (not just API signatures).
