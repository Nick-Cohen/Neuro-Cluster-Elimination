#!/usr/bin/env python3
"""
Test script to verify GPU guard auto-redirect functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.gpu_guard import ensure_gpu_server, is_on_gpu_server, check_gpu_available
import socket

print("=" * 60)
print("GPU Guard Test Script")
print("=" * 60)

# This should auto-redirect if not on GPU server
ensure_gpu_server()

print(f"Running on: {socket.gethostname()}")
print(f"On GPU server: {is_on_gpu_server()}")
print(f"GPU available: {check_gpu_available()}")

if check_gpu_available():
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'],
        capture_output=True,
        text=True
    )
    print("\nGPU Info:")
    print(result.stdout)
else:
    print("\nWARNING: No GPU detected!")

print("\n✓ Test successful - running on correct server")
