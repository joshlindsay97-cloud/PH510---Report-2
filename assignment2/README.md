# PH510 Assignment 2 — Vectors, Triangles, and Hansen Checks

This folder is my solution for PH510 Assignment 2.

The script builds a small 3D vector class and then uses it for:
- Task 2: triangle areas + internal angles from the given vertices
- Task 3: complex vectors + finite-difference divergence/curl checks for the Hansen fields

## Requirements
- Python 3.10+

## Run
From the repo root:

python assignment2/ph510_a2.py

Change finite-difference step:
python assignment2/ph510_a2.py --h 1e-6

Custom test points for Task 3:
python assignment2/ph510_a2.py --points 0,0,0 0.25,0,0.5 1,1,1

Save output to a file:
python assignment2/ph510_a2.py --out assignment2_results.txt

## Output notes
Task 2 prints a table of vertices, areas, and angles (radians + degrees).

Task 3 prints divergence/curl results and residuals comparing:
- numerical vs analytic curl
- numerical curl vs the assignment RHS

## License
MIT License (see header of ph510_a2.py).

Pylint score: 6.86/10
