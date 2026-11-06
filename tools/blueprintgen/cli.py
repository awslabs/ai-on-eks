# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI: aoe-blueprint gen <entry.yaml> [--target vllm|dynamo] [-o out.yaml]."""
from __future__ import annotations

import argparse
import sys

import yaml

from tools.blueprintgen.generate import generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aoe-blueprint")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("gen", help="generate a blueprint from a registry entry")
    gen.add_argument("entry", help="path to registry/models/<name>.yaml")
    gen.add_argument("--target", choices=["vllm", "dynamo"], default="vllm")
    gen.add_argument("-o", "--out", help="write to file instead of stdout")

    args = parser.parse_args(argv)
    if args.command == "gen":
        with open(args.entry) as fh:
            entry = yaml.safe_load(fh)
        manifest = generate(entry, target=args.target)
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(manifest)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(manifest)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
