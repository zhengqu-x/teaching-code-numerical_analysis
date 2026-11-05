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

`Plots` is the only external package. `LinearAlgebra`, `Printf`, `Random`, and `Test` are
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
# or
include("Lecture7/Lecture7_support.jl")
# or
include("Lecture8/Lecture8_support.jl")
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
| 7 | `Lecture7/Lecture7_support.jl` | Four activities on composite quadrature, endpoint corrections, Romberg, and periodic integration |
| 8 | `Lecture8/Lecture8_support.jl` | Eight activities on adaptive Simpson, Gaussian rules, integral equations, tensor quadrature and Monte Carlo, plus a 784-dimensional research example |

Lecture 4 Activities 1–3 cover natural and clamped cubic splines, 4–5 cover
B-spline bases, 6 compares equispaced and Chebyshev nodes, and 7 fits a
least-squares cubic. Its unnumbered illustration blocks reproduce the single
cubic B-spline and ReLU Remez plots.

The Lecture 2 support file previously had a `.txt` extension. Its Julia filename
now matches the lecture slides. `Lecture3/hermite_comparison.jl` remains a small
compatibility entry point for loading the shared plotting functions.

## Lecture 7

Load `include("Lecture7/Lecture7_support.jl")`. Run the four commented activity
blocks in the same order as the slides:

1. Error and Function Evaluations.
2. Correcting the Endpoints.
3. Romberg and a Smoothness Test.
4. Periodic Endpoints.

## Lecture 8 (6 November 2026)

Load `include("Lecture8/Lecture8_support.jl")`. This file includes `helper.jl`
and leaves all eight activities in separate commented blocks, in slide order:

1. Adaptive Simpson and a narrow peak that the initial samples miss.
2. Gaussian polynomial precision.
3. Gauss-Legendre versus composite Simpson at equal evaluation counts.
4. Chebyshev, Laguerre and Hermite weights.
5. A Gaussian average of a classifier output.
6. Nystrom discretization of an integral equation with an exact reference.
7. Tensor quadrature for a double integral.
8. Monte Carlo on an eight-dimensional cube, compared with three Gauss nodes per coordinate (6561 evaluations for each method): a smooth integrand and a discontinuous threshold probability. The latter includes 100 separate Monte Carlo runs to show variability.

All activities use the shared functions in `helper.jl`. Monte Carlo
confidence intervals are approximate; a small algebraic residual certifies
only the discrete linear solve, not the integral-equation approximation.
The research slides connect quadrature to neural operators (JMLR 2023)
and Monte Carlo class-probability estimation under Gaussian input noise
(Cohen et al., ICML 2019). A separate research example after Activity 8 samples
a 784-dimensional Gaussian integral for an illustrative linear classifier
and plots the estimates against an independent reference.
Jacobi and Gauss-Seidel follow next time.

## Notation and interpretation

- Mathematical node `x_i` is stored at Julia index `i+1`.
- For Gaussian nodes `x_0,...,x_n`, call `gauss_rule(family,n+1)`;
  the precision is `2n+1`. The integration helpers also take `n+1` nodes.
- `L(nodes,j,z)` uses the mathematical basis index `j=0,...,n`.
  `lagrange_basis(nodes,j,z)` retains the one-based interface used in the
  Lecture 2 instructor checks.
- For B-splines, `p` is the degree and `nsegments` counts intervals.
  `bspline_basis(x,i,p,knots)` uses Julia index `i` for mathematical `N_(i-1,p)`.
- `spline_slopes` returns first derivatives `m_i=S'(x_i)`, not second derivatives.
- Errors sampled on a plotting grid are not certified uniform error bounds.
- `adaptive_simpson(f,a,b,n;atol=1e-8)` uses an even `n >= 2` on each current
  interval, compares `S_n` with `S_{2n}`, and keeps `n` fixed when bisecting.
  It reuses all sampled values. Omitting `n` gives the previous default `n=2`.
  Lecture 8 Activity 1 integrates two interior peaks on a smooth background,
  uses `n=4`, and compares `n=2,4,8`. Its five-point missed-peak example uses `n=2`.
- `tensor_quadrature(f,rules)` accepts a vector of one-dimensional rules and
  evaluates a vector-input integrand at every tensor node. `fill(rule,d)` uses
  the same rule in all coordinates; three nodes give `3^d` evaluations.
- `uniform_simpson(f,a,b,n;atol=1e-8)` doubles the global subinterval count from
  an initial even `n`, reuses every sample, and stops using the Simpson-difference
  estimate. It returns the corrected value and a history of evaluation counts.
  Activity 1 compares both methods at the same tolerance with the exact integral.

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
