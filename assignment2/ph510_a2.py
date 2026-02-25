#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Josh Lindsay
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
PH510 Assignment 2 — Vector OOP + triangle geometry + Hansen checks.

What this script does:
- Task 2: triangle areas + internal angles
- Task 3: finite-difference divergence/curl checks for the Hansen fields

Python:
- Intended for Python 3.10+

Usage:
  python ph510_a2.py
  python ph510_a2.py --h 1e-6 --points 0,0,0 0.1,0.2,0.3 1,1,1
  python ph510_a2.py --out results.txt
"""

import argparse
import cmath
import contextlib
import io
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


# =========================
# Task 2: Geometry
# =========================

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
# Task 3: Finite differences
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
    return (
        central_diff_scalar(fx, x, 0, h)
        + central_diff_scalar(fy, x, 1, h)
        + central_diff_scalar(fz, x, 2, h)
    )


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


# =========================
# Hansen fields
# =========================

def hansen_setup():
    k = Vector3(0.0, 0.0, math.pi)
    k_mag = k.magnitude()

    def phase(x: Vector3[float]) -> complex:
        return cmath.exp(1j * k.dot(x))

    def M(x: Vector3[float]) -> ComplexVector3:
        ph = phase(x)
        return ComplexVector3(ph, 0.0 * ph, 0.0 * ph)

    def N(x: Vector3[float]) -> ComplexVector3:
        ph = phase(x)
        return ComplexVector3(0.0 * ph, ph, 0.0 * ph)

    def div_M_analytic(_: Vector3[float]) -> complex:
        return 0.0 + 0.0j

    def div_N_analytic(_: Vector3[float]) -> complex:
        return 0.0 + 0.0j

    def curl_M_analytic(x: Vector3[float]) -> ComplexVector3:
        ph = phase(x)
        return ComplexVector3(0.0j, 1j * k_mag * ph, 0.0j)

    def curl_N_analytic(x: Vector3[float]) -> ComplexVector3:
        ph = phase(x)
        return ComplexVector3(-1j * k_mag * ph, 0.0j, 0.0j)

    return k, k_mag, phase, M, N, div_M_analytic, div_N_analytic, curl_M_analytic, curl_N_analytic


# =========================
# Output helpers
# =========================

def fmt_complex(z: complex, prec: int = 6) -> str:
    a, b = z.real, z.imag
    if abs(b) < 10 ** (-(prec + 1)):
        return f"{a:.{prec}g}"
    if abs(a) < 10 ** (-(prec + 1)):
        return f"{b:.{prec}g}j"
    return f"{a:.{prec}g}{b:+.{prec}g}j"


def fmt_vec3(v: Vector3[Number], prec: int = 6) -> str:
    def f(x: Number) -> str:
        return fmt_complex(x, prec) if isinstance(x, complex) else f"{float(x):.{prec}g}"
    return f"({f(v.x)}, {f(v.y)}, {f(v.z)})"


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


# =========================
# Task runners
# =========================

def run_task2() -> None:
    triangles = [
        (Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0)),
        (Vector3(-1.0, -1.0, -1.0), Vector3(0.0, -1.0, -1.0), Vector3(-1.0, 0.0, -1.0)),
        (Vector3(1.0, 0.0, 0.0), Vector3(0.0, 0.0, 1.0), Vector3(0.0, 0.0, 0.0)),
        (Vector3(0.0, 0.0, 0.0), Vector3(1.0, -1.0, 0.0), Vector3(0.0, 0.0, 1.0)),
    ]

    headers = ["Tri", "A", "B", "C", "Area", "Angles (rad)", "Angles (deg)"]
    rows: List[List[str]] = []

    for i, (a, b, c) in enumerate(triangles, start=1):
        area = triangle_area(a, b, c)
        ang_r = triangle_angles_radians(a, b, c)
        ang_d = radians_to_degrees(ang_r)
        rows.append(
            [
                str(i),
                fmt_vec3(a, 6),
                fmt_vec3(b, 6),
                fmt_vec3(c, 6),
                f"{area:.12g}",
                f"({ang_r[0]:.6g}, {ang_r[1]:.6g}, {ang_r[2]:.6g})",
                f"({ang_d[0]:.6g}, {ang_d[1]:.6g}, {ang_d[2]:.6g})",
            ]
        )

    print("\n=== Task 2: triangle areas + internal angles ===\n")
    print_table(headers, rows)


def run_task3(points: Sequence[Vector3[float]], h: float) -> None:
    k, k_mag, phase, M, N, div_M_a, div_N_a, curl_M_a, curl_N_a = hansen_setup()

    headers = [
        "Point x",
        "phase",
        "divM_num",
        "divM_an",
        "divN_num",
        "divN_an",
        "||curlN_num-curlN_an||",
        "||curlM_num-curlM_an||",
        "||curlN_num-M/|k|||",
        "||curlM_num-N/|k|||",
    ]
    rows: List[List[str]] = []

    print("\n=== Task 3: Hansen checks (finite diff vs analytic) ===\n")
    print(f"k = {fmt_vec3(k, 6)}  |k| = {k_mag:.12g}  h = {h:g}\n")

    for x in points:
        ph = phase(x)

        divM_num = divergence(M, x, h)
        divN_num = divergence(N, x, h)
        divM_an = div_M_a(x)
        divN_an = div_N_a(x)

        curlN_num = curl(N, x, h)
        curlM_num = curl(M, x, h)
        curlN_an = curl_N_a(x)
        curlM_an = curl_M_a(x)

        rhs_curlN_assign = M(x) / k_mag
        rhs_curlM_assign = N(x) / k_mag

        res_curlN_vs_an = (curlN_num - curlN_an).magnitude()
        res_curlM_vs_an = (curlM_num - curlM_an).magnitude()
        res_curlN_vs_assign = (curlN_num - rhs_curlN_assign).magnitude()
        res_curlM_vs_assign = (curlM_num - rhs_curlM_assign).magnitude()

        rows.append(
            [
                fmt_vec3(x, 6),
                fmt_complex(ph, 6),
                fmt_complex(divM_num, 6),
                fmt_complex(divM_an, 6),
                fmt_complex(divN_num, 6),
                fmt_complex(divN_an, 6),
                f"{res_curlN_vs_an:.6g}",
                f"{res_curlM_vs_an:.6g}",
                f"{res_curlN_vs_assign:.6g}",
                f"{res_curlM_vs_assign:.6g}",
            ]
        )

    print_table(headers, rows)

    x0 = points[0]
    print("\n--- Details at first point ---")
    print("x =", fmt_vec3(x0, 6), " phase =", fmt_complex(phase(x0), 12))
    print("M(x) =", fmt_vec3(M(x0), 12))
    print("N(x) =", fmt_vec3(N(x0), 12))
    print("curl(N)_num =", fmt_vec3(curl(N, x0, h), 12))
    print("curl(N)_an  =", fmt_vec3(curl_N_a(x0), 12))
    print("M/|k|       =", fmt_vec3(M(x0) / k_mag, 12))
    print("curl(M)_num =", fmt_vec3(curl(M, x0, h), 12))
    print("curl(M)_an  =", fmt_vec3(curl_M_a(x0), 12))
    print("N/|k|       =", fmt_vec3(N(x0) / k_mag, 12))


# =========================
# CLI
# =========================

def parse_point(s: str) -> Vector3[float]:
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Bad point '{s}'. Use x,y,z")
    x, y, z = (float(p.strip()) for p in parts)
    return Vector3(x, y, z)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PH510 A2: vectors + triangles + Hansen checks.")
    p.add_argument("--h", type=float, default=1e-5, help="Central difference step (default 1e-5).")
    p.add_argument(
        "--points",
        nargs="*",
        default=["0,0,0", "0.1,0.2,0.3", "1,1,1"],
        help="Points as x,y,z floats (space-separated).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Optional path to save the printed output to a text file (e.g. results.txt).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    points = [parse_point(s) for s in args.points]

    if args.out:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_task2()
            run_task3(points=points, h=float(args.h))
        output = buf.getvalue()

        print(output, end="")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\nSaved output to: {args.out}")
        return

    run_task2()
    run_task3(points=points, h=float(args.h))


if __name__ == "__main__":
    main()
