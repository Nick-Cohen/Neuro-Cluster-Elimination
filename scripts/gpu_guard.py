"""
GPU Execution Guard for NCE Project

This module ensures GPU-requiring scripts run on the deepreasoning server.
Import and call ensure_gpu_server() at the top of any script that needs GPUs.

Example:
    from nce.utils.gpu_guard import ensure_gpu_server
    ensure_gpu_server()
    # ... rest of your script
"""

import os
import sys
import socket
import subprocess
from pathlib import Path


def is_on_gpu_server():
    """Check if currently running on the GPU server."""
    hostname = socket.gethostname()
    return hostname.startswith('deepreasoning')


def get_project_root():
    """Get NCE project root directory."""
    # gpu_guard.py is in scripts/, so parent is project root
    return Path(__file__).parent.parent


def ensure_gpu_server(auto_redirect=True):
    """
    Ensure the script is running on the GPU server.
    
    Args:
        auto_redirect: If True and not on GPU server, attempt to re-execute
                      the script on deepreasoning via SSH. If False, just
                      print an error and exit.
    
    Returns:
        None if on GPU server, exits otherwise.
    """
    if is_on_gpu_server():
        return
    
    if not auto_redirect:
        print("ERROR: This script requires GPU resources.", file=sys.stderr)
        print("Please run on deepreasoning server:", file=sys.stderr)
        print(f"  ssh deepreasoning", file=sys.stderr)
        print(f"  cd {get_project_root()}", file=sys.stderr)
        print(f"  source venv/bin/activate", file=sys.stderr)
        print(f"  python {' '.join(sys.argv)}", file=sys.stderr)
        sys.exit(1)
    
    # Auto-redirect to GPU server
    project_root = get_project_root().resolve()
    script_path = Path(sys.argv[0]).resolve()
    
    # Ensure script is within project
    try:
        rel_path = script_path.relative_to(project_root)
    except ValueError:
        print(f"ERROR: Script {script_path} is not within project {project_root}", file=sys.stderr)
        sys.exit(1)
    
    print(f"==> Not on GPU server (current: {socket.gethostname()})", file=sys.stderr)
    print(f"==> Auto-redirecting to deepreasoning...", file=sys.stderr)
    
    # Use the run_on_gpu.sh wrapper
    wrapper_script = project_root / "scripts" / "run_on_gpu.sh"
    
    if not wrapper_script.exists():
        print(f"ERROR: Wrapper script not found: {wrapper_script}", file=sys.stderr)
        print(f"Please run manually:", file=sys.stderr)
        print(f"  ssh deepreasoning 'cd {project_root} && source venv/bin/activate && python {rel_path} {' '.join(sys.argv[1:])}'", file=sys.stderr)
        sys.exit(1)
    
    # Execute via wrapper and exit
    cmd = [str(wrapper_script), str(script_path)] + sys.argv[1:]
    os.execvp(str(wrapper_script), cmd)


def check_gpu_available():
    """
    Check if GPU is available via nvidia-smi.
    
    Returns:
        bool: True if GPU detected, False otherwise.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


if __name__ == "__main__":
    print(f"Hostname: {socket.gethostname()}")
    print(f"On GPU server: {is_on_gpu_server()}")
    print(f"GPU available: {check_gpu_available()}")
    print(f"Project root: {get_project_root()}")
