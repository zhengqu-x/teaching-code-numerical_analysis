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
# or
include("Lecture9/Lecture9_support.jl")
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
| 8 | `Lecture8/Lecture8_support.jl` | Six activities on adaptive Simpson, Gaussian rules, Gaussian averages and integral equations |
| 9 | `Lecture9/Lecture9_support.jl` | Six activities on tensor quadrature, Monte Carlo, Jacobi, Gauss-Seidel and graph propagation |

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

## Lectures 8 and 9

Lecture 8 has six activities: adaptive Simpson, Gaussian precision, Gauss versus
composite Simpson, weighted Gaussian rules, Gaussian averages, and Nystrom
integral equations. The residual of the discrete solve and the error against
the known continuous solution are reported separately.

Lecture 9 has six activities in slide order:

1. Tensor Simpson and Gauss quadrature on a rectangle.
2. Monte Carlo and tensor Gauss for a smooth eight-dimensional integral.
3. A threshold probability, including variation across 100 separate runs.
4. Jacobi and Gauss-Seidel on a four-equation system.
5. Spectral radii, a positive definite matrix where Jacobi diverges, and residual versus error.
6. Graph-prediction propagation and two solvers for the same fixed-point system.

An optional example after Activity 3 samples all 784 coordinates of a noisy
classifier input. Monte Carlo confidence intervals are approximate. Graph
propagation is an illustrative numerical example with supplied initial scores,
not a trained APPNP model. It uses row normalization; the cited APPNP paper
uses symmetric normalization.

The shared `stationary_solve` stops on `norm(b-A*x,Inf) <= atol + rtol*norm(b,Inf)`.
Its flag records a residual test, not a forward-error certificate. Iterates are
stored as columns for classroom plots; column 1 is the initial vector.
Use `stationary_matrix` to inspect either iteration matrix. The notation is
`A=D-L-U`; no explicit inverse is formed by these helpers.


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

## Additional Lecture 9 research readings

- Klicpera, Bojchevski and Guennemann, [Predict then Propagate](https://arxiv.org/abs/1810.05997), ICLR 2019.
- Jie Chen, [Graph Neural Preconditioners for Iterative Solutions of Sparse Linear Systems](https://proceedings.iclr.cc/paper_files/paper/2025/file/3bb7e33e4da1fd19ebc6f0899e45a9dc-Paper-Conference.pdf), ICLR 2025.
