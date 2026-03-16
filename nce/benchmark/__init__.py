"""Benchmark training harness for NCE neural network factors.

Provides single-bucket training with wall-clock time limits,
checkpoint error tracking, and structured output (plots + metrics).
"""

from .training import train_single_bucket

__all__ = ['train_single_bucket']
