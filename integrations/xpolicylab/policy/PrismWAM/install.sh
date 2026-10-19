#!/usr/bin/env bash
# Add the XPolicyLab transport to the active PrismWAM environment.
# Install PrismWAM first (PrismWAM/docs/installation.md), activate it, then run this.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
require_xpl
require_prismwam
run_command python -m pip install -r "$PRISMWAM_ROOT/environments/xpolicylab.txt"
run_command python -m pip install --no-deps -e "$XPL_ROOT"
run_command python -m pip check
