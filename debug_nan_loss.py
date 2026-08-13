# %% [markdown]
# # Debug NaN Loss in or_chain Buckets
# Isolate one or_chain bucket, load the data, reconstruct the trainer,
# and call the loss function step-by-step to find what causes NaN.

# %%
import torch
import copy
import pickle

from nce.benchmark.training import _load_bucket_data, _find_problem, _reconstruct_bucket
from nce.benchmark_problems.small_problems import small_problems
from nce.config_schema import prepare_config
from nce.neural_networks.net import Net
from nce.neural_networks.train import Trainer
from nce.neural_networks.losses import unnormalized_kl

# %%
# Pick one of the NaN buckets
pt_path = "data/hard_buckets/or_chain_10_fg_uai__bucket_60.pt"
device = "cuda"

# Load bucket data
bucket_data = _load_bucket_data(pt_path, device)
metadata = bucket_data['metadata']
exact_fw = bucket_data['exact_fw']
exact_bw = bucket_data['exact_bw']

print(f"problem_key: {metadata['problem_key']}")
print(f"bucket_label: {metadata['bucket_label']}")
print(f"auto_ecl: {metadata['auto_ecl']}")

# %%
# Reconstruct config (same as benchmark)
nn_config = {
    'loss_fn': 'unnormalized_kl',
    'num_epochs': 10000,
    'batch_size': 512,
    'lr': 0.001,
    'hidden_sizes': [64, 64],
    'sampling_scheme': 'uniform',
    'num_samples': 10000,
    'device': device,
}

config = copy.deepcopy(nn_config)
problem_idx = _find_problem(metadata['problem_key'])
config['ecl'] = metadata['auto_ecl']
config['iB'] = small_problems.configs['default'][problem_idx].get('iB', 100)
config['error_tracking'] = False
config['sampling_scheme'] = 'all'
config['device'] = device
config = prepare_config(config, strict=False)

print(f"Config prepared. loss_fn={config['loss_fn']}, batch_size={config['batch_size']}")

# %%
# Reconstruct live bucket
fastgm, bucket = _reconstruct_bucket(problem_idx, metadata['bucket_label'], config, device)
message_size = bucket.get_message_size()
print(f"message_size: {message_size}")

# %%
# Create Net + Trainer (for setup / data loading only)
net = Net(bucket, hidden_sizes=config.get('hidden_sizes', []))
trainer = Trainer(net, bucket)

# %%
# Load all training data
all_data = trainer.dataloader.load_all()[0]
x_all = all_data['x']
y_all = all_data['y']
bw_all = all_data['bw']

print(f"x_all shape: {x_all.shape}")
print(f"y_all shape: {y_all.shape}")
print(f"bw_all: {'None' if bw_all is None else bw_all.shape}")
print(f"\ny_all stats: min={y_all.min():.4f}, max={y_all.max():.4f}, "
      f"mean={y_all.mean():.4f}, has_nan={torch.isnan(y_all).any()}")
if bw_all is not None:
    print(f"bw_all stats: min={bw_all.min():.4f}, max={bw_all.max():.4f}, "
          f"mean={bw_all.mean():.4f}, has_nan={torch.isnan(bw_all).any()}")

# %%
# Check normalizing constants from data_preprocessor
dp = trainer.data_preprocessor
print(f"normalizing_constant: {dp.normalizing_constant}")
print(f"bw_normalizing_constant: {dp.bw_normalizing_constant}")
print(f"global_max_targets: {dp.global_max_targets}")

# %%
# Forward pass: get NN prediction (before any training)
with torch.no_grad():
    outputs = net(x_all.to(device)).squeeze()

print(f"outputs shape: {outputs.shape}")
print(f"outputs stats: min={outputs.min():.4f}, max={outputs.max():.4f}, "
      f"mean={outputs.mean():.4f}, has_nan={torch.isnan(outputs).any()}")

# %%
# Pickle the prediction and target for standalone reproduction
pickle.dump({
    'outputs': outputs.cpu(),
    'targets': y_all.cpu(),
    'bw': bw_all.cpu() if bw_all is not None else None,
    'bw_normalizing_constant': dp.bw_normalizing_constant.cpu() if dp.bw_normalizing_constant is not None else None,
    'global_max_targets': dp.global_max_targets,
}, open('debug_nan_data.pkl', 'wb'))
print("Saved debug_nan_data.pkl")

# %%
# Now call the loss function directly (the same way the trainer does)
targets = y_all.to(device)
bw = bw_all.to(device) if bw_all is not None else None

# This is how the trainer wraps the loss:
loss = unnormalized_kl(
    outputs, targets, bw,
    bw_normalizing_constant=dp.bw_normalizing_constant,
    max_val=dp.global_max_targets,
)
print(f"\nLoss at epoch 0 (before training): {loss.item()}")

# %%
# Now train a few epochs and watch for NaN
print("\n--- Training a few epochs, watching for NaN ---")
optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])
batch_size = config['batch_size']

for epoch in range(1, 100):
    net.train()
    epoch_losses = []
    for i in range(0, len(x_all), batch_size):
        end_idx = min(i + batch_size, len(x_all))
        x_b = x_all[i:end_idx].to(device)
        y_b = y_all[i:end_idx].to(device)
        bw_b = bw_all[i:end_idx].to(device) if bw_all is not None else None

        optimizer.zero_grad()
        out = net(x_b).squeeze()

        loss = unnormalized_kl(
            out, y_b, bw_b,
            bw_normalizing_constant=dp.bw_normalizing_constant,
            max_val=dp.global_max_targets,
        )
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)

    if epoch <= 5 or epoch % 10 == 0:
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6e}")

    if any(torch.isnan(torch.tensor(l)) for l in epoch_losses):
        print(f"\n*** NaN detected at epoch {epoch}! ***")
        print(f"Per-batch losses: {epoch_losses}")

        # Inspect the state
        with torch.no_grad():
            full_out = net(x_all.to(device)).squeeze()
            print(f"\nOutputs after NaN epoch:")
            print(f"  min={full_out.min():.4f}, max={full_out.max():.4f}")
            print(f"  has_nan={torch.isnan(full_out).any()}")
            print(f"  has_inf={torch.isinf(full_out).any()}")

            # Check NN parameters
            for name, p in net.named_parameters():
                print(f"  param {name}: has_nan={torch.isnan(p).any()}, "
                      f"min={p.min():.4f}, max={p.max():.4f}")

        # Save state at NaN for further inspection
        pickle.dump({
            'outputs': full_out.cpu(),
            'targets': y_all.cpu(),
            'bw': bw_all.cpu() if bw_all is not None else None,
            'bw_normalizing_constant': dp.bw_normalizing_constant.cpu() if dp.bw_normalizing_constant is not None else None,
            'global_max_targets': dp.global_max_targets,
            'net_state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
            'epoch': epoch,
        }, open('debug_nan_at_failure.pkl', 'wb'))
        print("Saved debug_nan_at_failure.pkl")
        break

# %%
# Manual dissection of the loss function internals
# (Run this after NaN is detected, or use debug_nan_at_failure.pkl)
print("\n--- Manual loss dissection ---")
with torch.no_grad():
    out = net(x_all.to(device)).squeeze()
    targ = y_all.to(device)
    bw_hat = bw_all.to(device) if bw_all is not None else None

    # Step through unnormalized_kl internals
    if bw_hat is not None:
        bw_hat_detached = bw_hat.detach()
        if dp.bw_normalizing_constant is not None:
            bw_hat_detached = bw_hat_detached - dp.bw_normalizing_constant
        out_shifted = out + bw_hat_detached
        targ_shifted = targ + bw_hat_detached
    else:
        out_shifted = out
        targ_shifted = targ

    print(f"After bw shift:")
    print(f"  outputs: min={out_shifted.min():.4f}, max={out_shifted.max():.4f}, "
          f"nan={torch.isnan(out_shifted).any()}")
    print(f"  targets: min={targ_shifted.min():.4f}, max={targ_shifted.max():.4f}, "
          f"nan={torch.isnan(targ_shifted).any()}")

    # max_val computation
    if dp.global_max_targets is not None:
        max_val = max(dp.global_max_targets, out_shifted.max().item())
        max_val = torch.tensor(max_val, device=device)
    else:
        max_val = torch.max(torch.max(targ_shifted), torch.max(out_shifted))
    print(f"\nmax_val: {max_val.item():.4f}")

    log_p_tilde = targ_shifted - max_val
    log_q_tilde = out_shifted - max_val
    print(f"\nlog_p_tilde: min={log_p_tilde.min():.4f}, max={log_p_tilde.max():.4f}")
    print(f"log_q_tilde: min={log_q_tilde.min():.4f}, max={log_q_tilde.max():.4f}")

    p_tilde = torch.exp(log_p_tilde)
    q_tilde = torch.exp(log_q_tilde)
    print(f"\np_tilde: min={p_tilde.min():.6e}, max={p_tilde.max():.6e}, "
          f"nan={torch.isnan(p_tilde).any()}, inf={torch.isinf(p_tilde).any()}")
    print(f"q_tilde: min={q_tilde.min():.6e}, max={q_tilde.max():.6e}, "
          f"nan={torch.isnan(q_tilde).any()}, inf={torch.isinf(q_tilde).any()}")

    # The dangerous term: p_tilde * (log_p_tilde - log_q_tilde)
    log_ratio = log_p_tilde - log_q_tilde
    print(f"\nlog_ratio (log_p - log_q): min={log_ratio.min():.4f}, max={log_ratio.max():.4f}")

    term1 = p_tilde * log_ratio
    term2 = -p_tilde + q_tilde
    unsummed = term1 + term2
    print(f"\nterm1 (p * log(p/q)): min={term1.min():.6e}, max={term1.max():.6e}, "
          f"nan={torch.isnan(term1).any()}")
    print(f"term2 (-p + q): min={term2.min():.6e}, max={term2.max():.6e}")
    print(f"unsummed: nan={torch.isnan(unsummed).any()}, inf={torch.isinf(unsummed).any()}")

    result = torch.sum(unsummed, dim=0)
    print(f"\nFinal loss: {result.item()}")
