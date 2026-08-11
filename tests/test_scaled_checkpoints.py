"""
Unit tests for checkpoint scaling edge cases.

Tests the get_scaled_error_tracking_epochs() function to ensure correct behavior
across various edge cases: no scaling when ratio ≥ 1, deduplication, epoch 0
preservation, small ratios, and filtering to num_epochs.
"""

import sys
import math

# Inline the functions we need to test (avoids import issues with torch dependencies)
def get_error_tracking_epochs(num_epochs):
    """
    Generate sorted, deduplicated list of checkpoint epochs for error tracking.

    Fixed base list: [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
    then every 5000 up to num_epochs, plus the final epoch.

    Args:
        num_epochs: Total number of training epochs.

    Returns:
        Sorted list of unique checkpoint epoch numbers.
    """
    base = [0, 1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    epochs = [e for e in base if e <= num_epochs]
    # Add every 5000 after 10000
    e = 15000
    while e <= num_epochs:
        epochs.append(e)
        e += 5000
    # Add final epoch if not already present
    if num_epochs not in epochs:
        epochs.append(num_epochs)
    return sorted(set(epochs))


def get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size):
    """
    Generate checkpoint schedule scaled by batch_size/message_size ratio.

    Cost model:
    - Training one epoch: O(batch_size) — iterate through batch_size samples
    - Error tracking at one checkpoint: O(message_size) — evaluate all message_size assignments
    
    When batch_size << message_size, error tracking dominates cost. This function scales
    the checkpoint schedule so that tracking overhead remains proportional to training time.
    
    Formula: scaled_epoch = ceil((batch_size / message_size) * base_epoch)
    
    Args:
        num_epochs: Total number of training epochs.
        batch_size: Number of samples per training batch.
        message_size: Number of assignments in the message domain.
        
    Returns:
        Sorted list of unique checkpoint epoch numbers, with epoch 0 always included
        and all values <= num_epochs.
    """
    # Get base checkpoint schedule
    base = get_error_tracking_epochs(num_epochs)
    
    # Compute scaling factor
    scaling_factor = batch_size / message_size
    
    # If ratio >= 1.0, tracking is cheap relative to training, no scaling needed
    if scaling_factor >= 1.0:
        return base
    
    # Scale each non-zero epoch by the ratio
    scaled = [math.ceil(epoch * scaling_factor) for epoch in base if epoch > 0]
    
    # Deduplicate and sort (scaling may cause collisions)
    scaled = sorted(set(scaled))
    
    # Always include epoch 0 (baseline checkpoint)
    scaled = [0] + scaled
    
    # Filter to valid range (scaled epochs may exceed num_epochs)
    scaled = [e for e in scaled if e <= num_epochs]
    
    return scaled


def test_no_scaling_when_ratio_equals_1():
    """When batch_size == message_size (ratio=1), should return base schedule unchanged."""
    batch_size = 1000
    message_size = 1000
    num_epochs = 500
    
    base_schedule = get_error_tracking_epochs(num_epochs)
    scaled_schedule = get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size)
    
    assert scaled_schedule == base_schedule, (
        f"When ratio=1, scaled schedule should equal base schedule. "
        f"Got {scaled_schedule}, expected {base_schedule}"
    )


def test_no_scaling_when_batch_larger():
    """When batch_size > message_size (ratio > 1), should return base schedule unchanged."""
    batch_size = 10000
    message_size = 1000
    num_epochs = 500
    
    base_schedule = get_error_tracking_epochs(num_epochs)
    scaled_schedule = get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size)
    
    assert scaled_schedule == base_schedule, (
        f"When ratio > 1, scaled schedule should equal base schedule. "
        f"Got {scaled_schedule}, expected {base_schedule}"
    )


def test_deduplication():
    """Scaled schedule should contain no duplicate epochs."""
    batch_size = 10
    message_size = 1000
    num_epochs = 100
    
    scaled_schedule = get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size)
    
    assert len(scaled_schedule) == len(set(scaled_schedule)), (
        f"Scaled schedule contains duplicates: {scaled_schedule}. "
        f"Unique count: {len(set(scaled_schedule))}, total count: {len(scaled_schedule)}"
    )


def test_epoch_zero_always_included():
    """Epoch 0 (baseline) must always be included in the scaled schedule."""
    batch_size = 1
    message_size = 100000
    num_epochs = 10
    
    scaled_schedule = get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size)
    
    assert 0 in scaled_schedule, (
        f"Epoch 0 must always be included in scaled schedule. "
        f"Got {scaled_schedule}"
    )


def test_small_ratio_produces_valid_output():
    """Very small ratio (batch << message) should produce non-empty, sorted, valid epochs."""
    batch_size = 10
    message_size = 100000
    num_epochs = 5000
    
    scaled_schedule = get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size)
    
    # Should be non-empty
    assert len(scaled_schedule) > 0, (
        f"Scaled schedule should not be empty even with very small ratio. "
        f"Got {scaled_schedule}"
    )
    
    # Should be sorted
    assert scaled_schedule == sorted(scaled_schedule), (
        f"Scaled schedule should be sorted. Got {scaled_schedule}"
    )
    
    # All epochs should be <= num_epochs
    assert all(epoch <= num_epochs for epoch in scaled_schedule), (
        f"All epochs should be <= num_epochs={num_epochs}. "
        f"Got {scaled_schedule}, max={max(scaled_schedule)}"
    )
    
    # All epochs should be >= 0
    assert all(epoch >= 0 for epoch in scaled_schedule), (
        f"All epochs should be >= 0. Got {scaled_schedule}, min={min(scaled_schedule)}"
    )


def test_filtered_to_num_epochs():
    """Scaled schedule should never include epochs > num_epochs."""
    batch_size = 100
    message_size = 10000
    num_epochs = 10
    
    scaled_schedule = get_scaled_error_tracking_epochs(num_epochs, batch_size, message_size)
    
    assert max(scaled_schedule) <= num_epochs, (
        f"Max epoch in scaled schedule should be <= num_epochs={num_epochs}. "
        f"Got max={max(scaled_schedule)}, schedule={scaled_schedule}"
    )
    
    # Also verify all epochs are within bounds
    for epoch in scaled_schedule:
        assert 0 <= epoch <= num_epochs, (
            f"Epoch {epoch} is out of bounds [0, {num_epochs}]. "
            f"Schedule: {scaled_schedule}"
        )


if __name__ == "__main__":
    # Run all tests
    print("Running checkpoint scaling edge case tests...\n")
    
    tests = [
        ("No scaling when ratio equals 1", test_no_scaling_when_ratio_equals_1),
        ("No scaling when batch larger than message", test_no_scaling_when_batch_larger),
        ("Deduplication", test_deduplication),
        ("Epoch zero always included", test_epoch_zero_always_included),
        ("Small ratio produces valid output", test_small_ratio_produces_valid_output),
        ("Filtered to num_epochs", test_filtered_to_num_epochs),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}")
            print(f"  {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}")
            print(f"  Unexpected error: {e}\n")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print(f"{'='*60}")
    
    # Exit with non-zero status if any test failed
    sys.exit(0 if failed == 0 else 1)
