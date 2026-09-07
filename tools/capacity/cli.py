# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI: aoe-capacity find <instance_type> [--regions all|r1,r2] [--json]."""
from __future__ import annotations

import argparse
import json
import sys

from tools.capacity.scan import scan


def _parse_regions(value: str | None) -> list[str] | None:
    if value is None or value == "all":
        return None
    return [r.strip() for r in value.split(",") if r.strip()]


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No capacity found.")
        return
    header = f"{'REGION':<16}{'INSTANCE':<18}{'SPOT $/hr':>10}{'OD $/hr':>10}{'SCORE':>7}  AVAIL"
    print(header)
    for r in rows:
        spot = "-" if r["spot_price_usd_hr"] is None else f"{r['spot_price_usd_hr']:.2f}"
        od = "-" if r["on_demand_price_usd_hr"] is None else f"{r['on_demand_price_usd_hr']:.2f}"
        score = "-" if r["spot_placement_score"] is None else str(r["spot_placement_score"])
        print(f"{r['region']:<16}{r['instance_type']:<18}{spot:>10}{od:>10}{score:>7}  {r['availability']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aoe-capacity")
    sub = parser.add_subparsers(dest="command", required=True)
    find = sub.add_parser("find", help="find capacity for an instance type")
    find.add_argument("instance_type")
    find.add_argument("--regions", default="all", help="all | comma,separated,regions")
    find.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    args = parser.parse_args(argv)
    if args.command == "find":
        rows = scan([args.instance_type], regions=_parse_regions(args.regions))
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            _print_table(rows)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
