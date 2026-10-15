# Numerical Analysis Teaching Code

Julia examples for the 2026 Numerical Analysis course at Shenzhen University.
Download the whole repository using **Code > Download ZIP** and extract it.
Keep `helper.jl` beside the lecture folders; downloading only a support file is
not enough. The LaTeX lecture sources are not required to run this code.

## Setup

Use Julia 1.10 or later. From the repository folder, run once in a terminal:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then start Julia with this project:

```sh
julia --project=.
```

`Plots` is the only external package. `LinearAlgebra`, `Printf`, and `Test` are
Julia standard libraries. A local `Manifest.toml` records the resolved versions
and is excluded from Git so students can use different supported Julia versions.

## Classroom activities

Load the relevant support file from the repository folder:

```julia
include("Lecture2/lecture2_support.jl")
# or
include("Lecture3/Lecture3_support.jl")
# or
include("Lecture4/Lecture4_support.jl")
```

Loading these files defines the shared functions without running the activities.
Select the code **inside** a `#= ... =#` block and run it, or remove those two
comment markers and load the file again. Run one activity at a time in lecture
order. Lecture 2 Activity 1 contains three related parts; run them together.

If Julia is already open, activate the project with `using Pkg; Pkg.activate(".")`
from the repository folder before loading a file. If you start inside a lecture
folder instead, use `julia --project=..` and include its support file by name.

| Lecture | Student file | Contents |
| --- | --- | --- |
| 1 | `Lecture1/test.jl` | Newton iteration and floating-point precision |
| 2 | `Lecture2/lecture2_support.jl` | Monomial and Lagrange bases, divided differences, error bounds, Hermite interpolation |
| 3 | `Lecture3/Lecture3_support.jl` | Five activities on Hermite, Runge, and piecewise interpolation |
| 4 | `Lecture4/Lecture4_support.jl` | Seven activities on cubic splines, B-splines, Chebyshev nodes, and least squares |

Lecture 4 Activities 1–3 cover natural and clamped cubic splines, 4–5 cover
B-spline bases, 6 compares equispaced and Chebyshev nodes, and 7 fits a
least-squares cubic. Its unnumbered illustration blocks reproduce the single
cubic B-spline and ReLU Remez plots.

The Lecture 2 support file previously had a `.txt` extension. Its Julia filename
now matches the lecture slides. `Lecture3/hermite_comparison.jl` remains a small
compatibility entry point for loading the shared plotting functions.

## Notation and interpretation

- Mathematical node `x_i` is stored at Julia index `i+1`.
- `L(nodes,j,z)` uses the mathematical basis index `j=0,...,n`.
  `lagrange_basis(nodes,j,z)` retains the one-based interface used in the
  Lecture 2 instructor checks.
- For B-splines, `p` is the degree and `nsegments` counts intervals.
  `bspline_basis(x,i,p,knots)` uses Julia index `i` for mathematical `N_(i-1,p)`.
- `spline_slopes` returns first derivatives `m_i=S'(x_i)`, not second derivatives.
- Errors sampled on a plotting grid are not certified uniform error bounds.

## Checks and figure generation

Run the regression tests from the repository folder:

```sh
julia --project=. test/runtests.jl
```

The tests cover the retained helper functions and run the classroom activities
both in order and independently. They use headless plotting and do not export
figures.

Instructor scripts can also be run separately:

```sh
julia --project=. Lecture2/lecture2_activities.jl
julia --project=. Lecture3/lecture3_activities.jl
julia --project=. Lecture4/lecture4_spline_checks.jl
```

The latter two commands export figures to `figures/Lecture3/` and
`figures/Lecture4/`. Optional exports in the classroom files use the same
`figures/` folder. Generated figures and local metadata are excluded from Git.
No script needs the original Dropbox folder or writes into the lecture sources.
