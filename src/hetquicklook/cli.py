"""Command-line interface for basic quick-look operations."""

from __future__ import annotations

import argparse
from datetime import date
import json

from .config import QuicklookConfig
from .discovery import discover_observations, inventory_members
from .instrument import Instrument


def _discover_command(args: argparse.Namespace) -> int:
    observations = discover_observations(
        QuicklookConfig(args.root), instrument=args.instrument, date=args.date
    )
    records = [
        {
            "observation_id": item.observation_id,
            "archive_path": str(item.archive_path),
            "outer_tar_member": item.outer_tar_member,
            "date": item.date.isoformat(),
            "instrument": item.instrument.value,
        }
        for item in observations
    ]
    if args.inventory:
        for record, observation in zip(records, observations):
            members = inventory_members(observation)
            record["members"] = [
                {
                    "member": member.member_name,
                    "basename": member.member_basename,
                    "size": member.size,
                    "outer_tar_member": member.outer_tar_member,
                    "identity": (
                        {
                            "exposure_id": member.identity.exposure_id,
                            "amp_token": member.identity.amplifier_token,
                            "ifuslot": member.identity.ifu_slot,
                            "amp": member.identity.amplifier,
                            "frame_type": member.identity.frame_type,
                        }
                        if member.identity is not None
                        else None
                    ),
                    "parse_error": member.parse_error,
                }
                for member in members
            ]
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for record in records:
            print(f"{record['date']} {record['instrument']} {record['archive_path']}")
            if args.inventory:
                for member in record["members"]:
                    print(f"  {member['size']:>10} {member['member']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(prog="hetquicklook")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover = subparsers.add_parser("discover", help="list observation archives")
    discover.add_argument("root", help="root containing date directories")
    discover.add_argument("--instrument", choices=[item.value for item in Instrument])
    discover.add_argument("--date", help="date as YYYYMMDD or YYYY-MM-DD")
    discover.add_argument("--json", action="store_true", help="emit JSON records")
    discover.add_argument(
        "--inventory",
        action="store_true",
        help="include literal archive members and parsed identities",
    )
    discover.set_defaults(handler=_discover_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - exercised by the module entry point
    raise SystemExit(main())
