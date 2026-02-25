#!/usr/bin/env python3
"""
PH510 Assignment 2 (Vector OOP + Geometry + Hansen checks)

Commit 5: Add finite-difference operators (central difference, divergence, curl).
Task 2 output is still the default run target.

Python: 3.10+
"""

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, List, Sequence, Tuple, TypeVar, Union

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self._x!r}, y={self._y!r}, z={self._z!r})"

    def __str__(self) -> str:
        return f"({self._x}, {self._y}, {self._z})"

    def magnitude(self) -> float:
        """Euclidean norm (works for complex components via abs())."""
        return math.sqrt(abs(self._x) ** 2 + abs(self._y) ** 2 + abs(self._z) ** 2)

    def __add__(self, other: "Vector3[TNum]") -> "Vector3[TNum]":
        return self.__class__(self._x + other._x, self._y + other._y, self._z + other._z)

    def __sub__(self, other: "Vector3[TNum]") -> "Vector3[TNum]":
        return self.__class__(self._x - other._x, self._y - other._y, self._z - other._z)

    def __neg__(self) -> "Vector3[TNum]":
        return self.__class__(-self._x, -self._y, -self._z)

    def dot(self, other: "Vector3[TNum]") -> TNum:
        return self._x * other._x + self._y * other._y + self._z * other._z

    def cross(self, other: "Vector3[TNum]") -> "Vector3[TNum]":
        cx = self._y * other._z - self._z * other._y
        cy = self._z * other._x - self._x * other._z
        cz = self._x * other._y - self._y * other._x
        return self.__class__(cx, cy, cz)

    def scaled(self, scalar: TNum) -> "Vector3[TNum]":
        return self.__class__(self._x * scalar, self._y * scalar, self._z * scalar)

    def __mul__(self, scalar: TNum) -> "Vector3[TNum]":
        return self.scaled(scalar)

    def __rmul__(self, scalar: TNum) -> "Vector3[TNum]":
        return self.scaled(scalar)

    def __truediv__(self, scalar: TNum) -> "Vector3[TNum]":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero.")
        return self.__class__(self._x / scalar, self._y / scalar, self._z / scalar)


@dataclass(frozen=True, slots=True)
class ComplexVector3(Vector3[complex]):
    """Complex 3D vector using the Hermitian dot product a* · b."""

    _x: complex
    _y: complex
    _z: complex

    def dot(self, other: "ComplexVector3") -> complex:
        return (
            self._x.conjugate() * other._x
            + self._y.conjugate() * other._y
            + self._z.conjugate() * other._z
        )


def triangle_area(a: Vector3[float], b: Vector3[float], c: Vector3[float]) -> float:
    """Area = 0.5 * ||(b-a) x (c-a)||."""
    return 0.5 * (b - a).cross(c - a).magnitude()


def _clamped_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def triangle_angles_radians(
    a: Vector3[float], b: Vector3[float], c: Vector3[float]
) -> Tuple[float, float, float]:
    """Internal angles at A, B, C (radians)."""
    ab, ac = b - a, c - a
    ba, bc = a - b, c - b
    ca, cb = a - c, b - c

    def angle(u: Vector3[float], v: Vector3[float]) -> float:
        denom = u.magnitude() * v.magnitude()
        if denom == 0:
            raise ValueError("Degenerate triangle: zero edge length.")
        return _clamped_acos(float(u.dot(v) / denom))

    return angle(ab, ac), angle(ba, bc), angle(ca, cb)


def radians_to_degrees(vals: Sequence[float]) -> Tuple[float, ...]:
    return tuple(v * 180.0 / math.pi for v in vals)


# =========================
# Finite differences (Part 3)
# =========================

def central_diff_scalar(
    f: Callable[[Vector3[float]], complex], x: Vector3[float], axis: int, h: float
) -> complex:
    """Central difference approximation for a scalar function."""
    if axis == 0:
        xp, xm = Vector3(x.x + h, x.y, x.z), Vector3(x.x - h, x.y, x.z)
    elif axis == 1:
        xp, xm = Vector3(x.x, x.y + h, x.z), Vector3(x.x, x.y - h, x.z)
    elif axis == 2:
        xp, xm = Vector3(x.x, x.y, x.z + h), Vector3(x.x, x.y, x.z - h)
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return (f(xp) - f(xm)) / (2.0 * h)


def divergence(field: Callable[[Vector3[float]], ComplexVector3], x: Vector3[float], h: float) -> complex:
    """Numerical divergence ∇·F using central differences."""
    fx = lambda p: field(p).x
    fy = lambda p: field(p).y
    fz = lambda p: field(p).z
    return central_diff_scalar(fx, x, 0, h) + central_diff_scalar(fy, x, 1, h) + central_diff_scalar(fz, x, 2, h)


def curl(field: Callable[[Vector3[float]], ComplexVector3], x: Vector3[float], h: float) -> ComplexVector3:
    """Numerical curl ∇×F using central differences."""
    fx = lambda p: field(p).x
    fy = lambda p: field(p).y
    fz = lambda p: field(p).z

    dFz_dy = central_diff_scalar(fz, x, 1, h)
    dFy_dz = central_diff_scalar(fy, x, 2, h)

    dFx_dz = central_diff_scalar(fx, x, 2, h)
    dFz_dx = central_diff_scalar(fz, x, 0, h)

    dFy_dx = central_diff_scalar(fy, x, 0, h)
    dFx_dy = central_diff_scalar(fx, x, 1, h)

    return ComplexVector3(dFz_dy - dFy_dz, dFx_dz - dFz_dx, dFy_dx - dFx_dy)


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def line(sep: str = "-") -> str:
        return "+".join(sep * (w + 2) for w in widths)

    def fmt_row(r: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    print(line("-"))
    print(" " + fmt_row(headers))
    print(line("="))
    for r in rows:
        print(" " + fmt_row(r))
    print(line("-"))


def fmt_vec3(v: Vector3[Number]) -> str:
    return f"({v.x}, {v.y}, {v.z})"


def run_task2() -> None:
    triangles = [
        (Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0)),
        (Vector3(-1.0, -1.0, -1.0), Vector3(0.0, -1.0, -1.0), Vector3(-1.0, 0.0, -1.0)),
        (Vector3(1.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0), Vector3(0.0, 0.0, 0.0)),
        (Vector3(0.0, 0.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(0.0, 0.0, 1.0)),
    ]

    headers = ["Tri", "A", "B", "C", "Area", "Angles (deg)"]
    rows: List[List[str]] = []

    for i, (a, b, c) in enumerate(triangles, start=1):
        area = triangle_area(a, b, c)
        ang_r = triangle_angles_radians(a, b, c)
        ang_d = radians_to_degrees(ang_r)
        rows.append(
            [
                str(i),
                fmt_vec3(a),
                fmt_vec3(b),
                fmt_vec3(c),
                f"{area:.12g}",
                f"({ang_d[0]:.6g}, {ang_d[1]:.6g}, {ang_d[2]:.6g})",
            ]
        )

    print("\n=== Task 2: triangle areas + internal angles ===\n")
    print_table(headers, rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PH510 Assignment 2 (up to finite differences).")
    p.add_argument("--h", type=float, default=1e-5, help="Central difference step.")
    p.add_argument("--points", nargs="*", default=["0,0,0"])
    return p.parse_args()


def main() -> None:
    _ = parse_args()
    run_task2()


if __name__ == "__main__":
    main()
