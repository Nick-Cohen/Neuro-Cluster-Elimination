#!/usr/bin/bash
# Wrapper to run Python with correct library paths

# Add system library path for libffi.so.8 (which we'll symlink as .7)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Activate NCE-Frozen venv
source /home/cohenn1/NCE-Frozen-1-15-26/bin/activate

# Run Python with all arguments
exec python "$@"
