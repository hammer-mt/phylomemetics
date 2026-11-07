#!/bin/bash
# Run one memory-heavy analysis at a time on this machine: bash v2/heavy.sh <command ...>
# (the large models' matrices take several GB each; two such steps at once exhaust a 15 GB machine)
mkdir -p v2/cache
exec flock v2/cache/heavy.lock "$@"
