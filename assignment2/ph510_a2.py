#!/usr/bin/env python3
"""
PH510 Assignment 2 (Vector OOP + Geometry + Hansen checks)

Python: 3.10+

"""

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PH510 Assignment 2")
    p.add_argument("--h", type=float, default=1e-5)
    p.add_argument("--points", nargs="*", default=["0,0,0"])
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    print("PH510 A2 scaffold running")


if __name__ == "__main__":
    main()
