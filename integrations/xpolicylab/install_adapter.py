#!/usr/bin/env python3
"""Copy the small policy adapter into an existing XPolicyLab source checkout."""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xpolicylab-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.xpolicylab_root.expanduser().resolve()
    for filename in (
        "setup_policy_server.py",
        "model_template.py",
        "utils/checkpoint_resolver.py",
    ):
        if not (root / filename).is_file():
            parser.error(f"Not an XPolicyLab checkout: missing {root / filename}")
    source = Path(__file__).resolve().parent / "policy" / "PrismWAM"
    target = root / "policy" / "PrismWAM"
    files = sorted(path for path in source.iterdir() if path.is_file())
    if target.is_symlink():
        parser.error(f"Refusing a symbolic-link destination: {target}")
    conflicts = [
        path.name
        for path in files
        if (target / path.name).exists()
        and (target / path.name).read_bytes() != path.read_bytes()
    ]
    if conflicts:
        parser.error(
            "Existing adapter files differ; review or relocate them first: "
            + ", ".join(conflicts)
        )
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=True)
        for path in files:
            if not (target / path.name).exists():
                shutil.copy2(path, target / path.name)
    print(
        json.dumps(
            dict(
                source=str(source),
                target=str(target),
                files=[path.name for path in files],
                dry_run=args.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
