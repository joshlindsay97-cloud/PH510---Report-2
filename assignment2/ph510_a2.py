#!/usr/bin/env python3
"""
PH510 Assignment 2 (Vector OOP + Geometry + Hansen checks)

Python: 3.10+
"""

import argparse
import math
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar, Union

Number = Union[float, complex]
TNum = TypeVar("TNum", float, complex)


@dataclass(frozen=True, slots=True)
class Vector3(Generic[TNum]):
    """Simple 3D Cartesian vector."""

    _x: TNum
    _y: TNum
    _z: TNum

    @property
    def x(self) -> TNum:
        return self._x

    @property
    def y(self) -> TNum:
        return self._y

    @property
    def z(self) -> TNum:
        return self._z

    def __iter__(self) -> Iterable[TNum]:
        yield self._x
        yield self._y
        yield self._z

    def __str__(self) -> str:
        return f"({self._x}, {self._y}, {self._z})"

    def magnitude(self) -> float:
        return math.sqrt(abs(self._x) ** 2 + abs(self._y) ** 2 + abs(self._z) ** 2)

    def __add__(self, other: "Vector3[TNum]") -> "Vector3[TNum]":
        return self.__class__(self._x + other._x, self._y + other._y, self._z + other._z)

    def __sub__(self, other: "Vector3[TNum]") -> "Vector3[TNum]":
        return self.__class__(self._x - other._x, self._y - other._y, self._z - other._z)

    def dot(self, other: "Vector3[TNum]") -> TNum:
        return self._x * other._x + self._y * other._y + self._z * other._z

    def cross(self, other: "Vector3[TNum]") -> "Vector3[TNum]":
        cx = self._y * other._z - self._z * other._y
        cy = self._z * other._x - self._x * other._z
        cz = self._x * other._y - self._y * other._x
        return self.__class__(cx, cy, cz)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PH510 Assignment 2")
    p.add_argument("--h", type=float, default=1e-5)
    p.add_argument("--points", nargs="*", default=["0,0,0"])
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    v = Vector3(1.0, 0.0, 0.0)
    w = Vector3(0.0, 1.0, 0.0)
    print("v =", v, "|v| =", v.magnitude())
    print("v·w =", v.dot(w), "v×w =", v.cross(w))


if __name__ == "__main__":
    main()
