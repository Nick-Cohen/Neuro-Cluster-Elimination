#!/usr/bin/env python3
"""Test error tracking overhead using a precomputed hard bucket.

Compares training time with no error tracking vs tracking every epoch.
If error tracking adds <2x overhead, use Plan A (every epoch).
Otherwise use Plan B (scale by batch_size/message_size).
"""
import sys
import time
import os
sys.path.insert(0, '/home/cohenn1/NCE')

# Add GPU guard to auto-redirect to deepreasoning
from scripts.gpu_guard import ensure_gpu_server
ensure_gpu_server()

import torch
from nce.benchmark.training import _load_bucket_data, _reconstruct_bucket
from nce.neural_networks.net import Net
from nce.neural_networks.train import Trainer
from nce.inference.factor_nn import FactorNN
from nce.config_schema import prepare_config


def measure_training_time(pt_path, device, num_epochs, batch_size, track_every_epoch):
    """Train for num_epochs and measure wall time.
    
    Returns:
        Tuple of (elapsed_time_sec, message_size, batch_size)
    """
    # Load precomputed bucket
    bucket_data = _load_bucket_data(pt_path, device)
    metadata = bucket_data['metadata']
    exact_fw = bucket_data['exact_fw']
    exact_bw = bucket_data['exact_bw']
    
    problem_key = metadata['problem_key']
    bucket_label = metadata['bucket_label']
    
    # Find problem index
    from nce.benchmark_problems import small_problems
    problem_idx = None
    for idx, model in enumerate(small_problems.problems):
        if model.modelfile == problem_key:
            problem_idx = idx
            break
    
    if problem_idx is None:
        raise ValueError(f"Problem {problem_key} not found")
    
    # Use a full config from small_problems as the base
    from nce.benchmark_problems import small_problems
    base_config = small_problems.configs['default'][problem_idx].copy()
    
    # Override with test-specific values
    base_config['device'] = device
    base_config['ecl'] = metadata['auto_ecl']
    base_config['num_epochs'] = num_epochs
    base_config['batch_size'] = batch_size
    base_config['hidden_sizes'] = [8, 8]
    base_config['skip_early_stopping'] = True
    
    config = prepare_config(base_config, strict=False)
    
    # Reconstruct bucket
    fastgm, bucket = _reconstruct_bucket(problem_idx, bucket_label, config, device)
    message_size = int(bucket.get_message_size())
    
    # Create net and trainer
    net = Net(bucket, hidden_sizes=config['hidden_sizes'])
    trainer = Trainer(net, bucket)
    
    # Load data
    all_data = trainer.dataloader.load_all()[0]
    x_all, y_all, bw_all = all_data['x'], all_data['y'], all_data['bw']
    
    batches = []
    for i in range(0, len(x_all), batch_size):
        end_idx = min(i + batch_size, len(x_all))
        batches.append({
            'x': x_all[i:end_idx],
            'y': y_all[i:end_idx],
            'bw': bw_all[i:end_idx] if bw_all is not None else None,
        })
    
    # Precompute exact_contribution
    exact_contribution = (exact_fw * exact_bw).sum_all_entries()
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train one epoch
        loss = trainer.train_epoch(batches)
        
        # Error tracking if enabled
        if track_every_epoch:
            with torch.no_grad():
                approx_factor = FactorNN(net, trainer.data_preprocessor)
                approx_exact = approx_factor.to_exact()
                approx_contribution = (approx_exact * exact_bw).sum_all_entries()
                log_z_err = approx_contribution - exact_contribution
    
    elapsed = time.time() - start_time
    return elapsed, message_size, batch_size


if __name__ == '__main__':
    # Use smokers_20 bucket 400 (small, fast)
    pt_path = 'data/hard_buckets/smokers_20_bucket_400.pt'
    
    if not os.path.exists(pt_path):
        print(f"ERROR: {pt_path} not found")
        print("Run scripts/select_hard_buckets.py first to generate hard buckets")
        sys.exit(1)
    
    device = 'cpu'
    num_epochs = 10
    batch_size = 100
    
    print("="*60)
    print("Error Tracking Overhead Test")
    print("="*60)
    print(f"Bucket: {pt_path}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    print()
    
    print("1. Training WITHOUT error tracking...")
    time_without, msg_size, batch_size_used = measure_training_time(
        pt_path, device, num_epochs, batch_size, track_every_epoch=False
    )
    print(f"   Time: {time_without:.2f}s")
    print(f"   Message size: {msg_size}, Batch size: {batch_size_used}")
    print(f"   Ratio: {msg_size / batch_size_used:.1f}x")
    
    print(f"\n2. Training WITH error tracking every epoch...")
    time_with, _, _ = measure_training_time(
        pt_path, device, num_epochs, batch_size, track_every_epoch=True
    )
    print(f"   Time: {time_with:.2f}s")
    
    overhead = time_with / time_without
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Time without error tracking: {time_without:.2f}s")
    print(f"Time with error tracking:    {time_with:.2f}s")
    print(f"Overhead:                    {overhead:.2f}x")
    print(f"\nMessage size / batch size:   {msg_size / batch_size_used:.1f}x")
    print()
    
    if overhead < 2.0:
        print("✅ DECISION: Use Plan A (track every epoch)")
        print("   Error tracking adds <2x overhead")
    else:
        print("✅ DECISION: Use Plan B (scale checkpoints by batch_size/message_size)")
        print(f"   Error tracking adds {overhead:.2f}x overhead (>2x threshold)")
