# Task: Log Per-Bucket Hidden Sizes to File

## Goal

Run grid10x10.f5.wrap with the NBE config and output each trained bucket's hidden sizes to a file.

## How Hidden Sizes Are Resolved

In `nce/inference/bucket.py` lines 208-219, when `hidden_sizes='nbe,1'`:
- `b = 1` (the multiplier after the comma)
- `message_size = bucket.get_message_size()` (product of variable domain sizes in the message scope)
- `h = b * ceil(log2(message_size))`
- Final hidden_sizes = `[h, h]` (two hidden layers of size h)

## Script to Write

Create `notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py`:

```python
import time
from nce.benchmark_problems import nbe_sanity_check
from nce.inference.graphical_model import FastGM
import math

model = nbe_sanity_check.problems[4]  # grid10x10.f5.wrap
config = dict(nbe_sanity_check.configs['nbe'][4])
config['num_epochs'] = 1  # just 1 epoch - we only care about the sizes
config['device'] = 'cpu'

fastgm = FastGM(model=model, nn_config=config, device='cpu')

# Collect info about NN-eligible buckets BEFORE elimination
# (elimination destroys buckets as they're processed)
iB = config['iB']
ecl = config['ecl']
hidden_sizes_cfg = config['hidden_sizes']  # 'nbe,1'

# Parse the b multiplier
b = int(hidden_sizes_cfg.split(',')[1]) if ',' in str(hidden_sizes_cfg) else 1

lines = []
lines.append(f"Model: {model.modelfile}")
lines.append(f"iB={iB}, ecl={ecl}, hidden_sizes_cfg='{hidden_sizes_cfg}', b={b}")
lines.append(f"")
lines.append(f"{'Bucket':<10} {'Width':<8} {'Msg Size':<12} {'h=b*ceil(log2(msg_size))':<28} {'hidden_sizes'}")
lines.append("-" * 80)

for var in fastgm.elim_order:
    bucket = fastgm.buckets[var]
    width = bucket.get_width()
    ec = bucket.get_ec()
    msg_size = bucket.get_message_size()

    if width <= iB and ec <= ecl:
        continue  # exact bucket, skip

    h = b * math.ceil(math.log2(msg_size)) if msg_size > 1 else b
    lines.append(f"{var.label:<10} {width:<8} {msg_size:<12} {h:<28} [{h}, {h}]")

output = "\n".join(lines)
print(output)

# Write to file
outpath = "notebooks/March-2026/claude_experiments/nbe_eval_results/grid10x10_hidden_sizes.txt"
with open(outpath, 'w') as f:
    f.write(output + "\n")
print(f"\nWritten to {outpath}")
```

## Run Command

```bash
/home/cohenn1/NCE/venv/bin/python notebooks/March-2026/claude_experiments/nbe_log_hidden_sizes.py
```

## Key Notes

- `get_message_size()` returns the product of domain sizes for variables in the message scope (for binary vars, this is `2^width`)
- The NN-eligible check is `width <= iB AND ec <= ecl` → if False, bucket uses NN
- `bucket.get_width()` = number of variables in message scope
- `bucket.get_ec()` = product of all variable domain sizes in the bucket's full scope (different from message size)
- Buckets are destroyed during elimination, so collect info BEFORE calling `get_log_partition_function()`
