# Numerical Analysis: shared functions for Lectures 2-8.
# Each lecture support file loads this with:
#     include(joinpath(@__DIR__, "..", "helper.jl"))
# Keep helper.jl at the repository root, beside the lecture folders.
# Loading this file imports the packages and defines functions only.
# It does not run activities, benchmarks, checks, or figure exports.
# Mathematical nodes x_0,...,x_n are stored at Julia indices 1,...,n+1.
# L uses j=0,...,n; lagrange_basis retains the older one-based interface.
using LinearAlgebra
using Plots
using Printf
using Random

# INTEGRAL EQUATIONS AND MULTIPLE INTEGRALS (LECTURE 8)
"""Gauss-Legendre nodes and weights on [a,b]; pass n+1 for nodes x_0,...,x_n."""
function mapped_gauss_rule(a, b, num_nodes::Integer)
    isfinite(a) && isfinite(b) && a < b || throw(ArgumentError("Require finite a < b."))
    rule = gauss_rule(:legendre,num_nodes)
    c,d = (a+b)/2,(b-a)/2
    return (nodes=c .+ d.*rule.nodes,weights=d.*rule.weights)
end

"""Composite Simpson nodes/weights; n is the even subinterval count."""
function simpson_rule(a, b, n::Integer)
    isfinite(a) && isfinite(b) && a < b || throw(ArgumentError("Require finite a < b."))
    n >= 2 && iseven(n) || throw(ArgumentError("n must be positive and even."))
    h = (b-a)/n
    coefficients = [i == 0 || i == n ? 1.0 : isodd(i) ? 4.0 : 2.0 for i in 0:n]
    return (nodes=collect(range(a,b,length=n+1)),weights=h/3 .* coefficients)
end

"""Tensor quadrature in two variables. Each rule has nodes and weights."""
function tensor_quadrature(f, rx, ry)
    value = sum(rx.weights[i]*ry.weights[j]*f(rx.nodes[i],ry.nodes[j])
                for i in eachindex(rx.nodes), j in eachindex(ry.nodes))
    return (value=value,evaluations=length(rx.nodes)*length(ry.nodes))
end

"""Tensor quadrature for a vector-valued input x in d coordinates.
Pass one nodes/weights rule per coordinate, for example fill(rule,d).
Evaluate f(x) once at every tensor node, without exploiting separability.
The three-argument method tensor_quadrature(f,rx,ry) still accepts f(x,y)."""
function tensor_quadrature(f, rules::AbstractVector)
    isempty(rules) && throw(ArgumentError("Provide at least one coordinate rule."))
    all(length(r.nodes) == length(r.weights) > 0 for r in rules) ||
        throw(DimensionMismatch("Each rule needs equally many nodes and weights."))
    indices = Iterators.product((eachindex(r.nodes) for r in rules)...)
    value = sum(prod(rules[k].weights[idx[k]] for k in eachindex(rules)) *
                f([rules[k].nodes[idx[k]] for k in eachindex(rules)]) for idx in indices)
    return (value=value,evaluations=prod(length(r.nodes) for r in rules))
end


"""Nystrom discretization of y(t)=integral k(t,s)y(s)ds+f(t).
The supplied weights include the integration interval. Returns the nodal
solution, algebraic residual, and continuous Nystrom extension evaluate(t).
A small residual certifies only the discrete solve, not the quadrature error."""
function nystrom_solve(kernel, f, rule)
    nodes,weights = rule.nodes,rule.weights
    num_nodes = length(nodes)
    num_nodes == length(weights) && num_nodes > 0 || throw(DimensionMismatch("Check nodes and weights."))
    B = [weights[j]*kernel(nodes[i],nodes[j]) for i in 1:num_nodes,j in 1:num_nodes]
    A = Matrix{Float64}(I,num_nodes,num_nodes)-B
    rhs = f.(nodes)
    Y = A\rhs
    evaluate(t) = f(t)+sum(weights[j]*kernel(t,nodes[j])*Y[j] for j in 1:num_nodes)
    return (;nodes,weights,Y,evaluate,residual=norm(A*Y-rhs,Inf))
end

"""Monte Carlo integration on [0,1]^d using N iid uniform vectors.
Uses an explicit seed and Welford's sample-variance calculation.
The 95% interval is a CLT approximation, not a deterministic error bound."""
function mc_unitcube(f, d::Integer, N::Integer; seed::Integer=2026)
    d >= 1 && N >= 2 || throw(ArgumentError("Require d >= 1 and N >= 2."))
    rng = MersenneTwister(seed)
    average, M2 = 0.0,0.0
    for j in 1:N
        value = Float64(f(rand(rng,d)))
        isfinite(value) || throw(DomainError(value,"Nonfinite integrand value."))
        delta = value-average
        average += delta/j
        M2 += delta*(value-average)
    end
    variance = max(M2/(N-1),0.0)
    stderr = sqrt(variance/N)
    return (value=average,stderr=stderr,ci=(average-1.96stderr,average+1.96stderr),
            evaluations=N)
end

# FURTHER QUADRATURE (LECTURE 7)
"""Nested trapezoidal refinement and Richardson extrapolation.
Only the lower triangle of table is defined; upper entries are NaN.
levels rows use exactly 2^(levels-1)+1 function evaluations."""
function romberg_table(f, a, b; levels::Integer=6)
    isfinite(a) && isfinite(b) && a < b || throw(ArgumentError("Require finite a < b."))
    1 <= levels <= 20 || throw(ArgumentError("Use 1 to 20 levels."))
    R = fill(NaN,levels,levels)
    R[1,1] = (b-a)*(f(a)+f(b))/2
    evaluations = 2
    for k in 2:levels
        nold = 2^(k-2)
        h = (b-a)/nold
        R[k,1] = R[k-1,1]/2 + h/2*sum(f(a+(j-0.5)*h) for j in 1:nold)
        evaluations += nold
        for j in 2:k
            R[k,j] = R[k,j-1]+(R[k,j-1]-R[k-1,j-1])/(4^(j-1)-1)
        end
    end
    return (table=R,evaluations=evaluations)
end

"""Print |R[k,j]-exact| for the defined lower triangle of a Romberg table.
Undefined upper-triangle cells are blank. The numerical table is unchanged."""
function print_romberg_errors(table::AbstractMatrix, exact; io::IO=stdout)
    rows, cols = size(table)
    @printf(io,"%3s","k")
    for j in 1:cols
        @printf(io,"%12s","j=$j")
    end
    println(io)
    for k in 1:rows
        @printf(io,"%3d",k)
        for j in 1:cols
            if j <= k
                @printf(io,"%12.4e",abs(table[k,j]-exact))
            else
                @printf(io,"%12s","")
            end
        end
        println(io)
    end
    return nothing
end

"""adaptive_simpson(f, a, b, n=2; atol=1e-8, maxdepth=20)

Adaptive composite Simpson with n even subintervals on each current interval.
Compare S_n(a,b) with S_n(a,c)+S_n(c,b)=S_{2n}(a,b), where c=(a+b)/2.
Keep n fixed in both recursive calls and split the absolute tolerance in half.
Reuse all coarse-grid values; each refinement samples only n new midpoints.
The default n=2 gives the usual three-node versus five-node Simpson test.
Returns the Richardson-corrected value, the sum of |S2-S1|/15 estimates,
function-evaluation count, accepted leaf intervals, and a convergence flag.
The error estimate is heuristic: an unsampled peak can fool it."""
function adaptive_simpson(f, a, b, n::Integer=2; atol=1e-8, maxdepth::Integer=20)
    isfinite(a) && isfinite(b) && a < b || throw(ArgumentError("Require finite a < b."))
    n >= 2 && iseven(n) || throw(ArgumentError("n must be at least 2 and even."))
    isfinite(atol) && atol > 0 || throw(ArgumentError("atol must be finite and positive."))
    0 <= maxdepth <= 50 || throw(ArgumentError("Use maxdepth between 0 and 50."))
    evaluations = Ref(0)
    function sample(x)
        y = f(x)
        isfinite(y) || throw(DomainError(x,"Nonfinite integrand value."))
        evaluations[] += 1
        return y
    end
    # Simpson weights for n+1 cached values on an equally spaced grid.
    function composite(a,b,y)
        total = y[1]+y[end]
        for j in 1:n-1
            total += (isodd(j) ? 4 : 2)*y[j+1]
        end
        return (b-a)/(3n)*total
    end
    intervals = Tuple{Float64,Float64}[]
    function refine(x,y,S,tol,depth)
        local a,b = first(x),last(x)
        # Insert one midpoint between consecutive cached nodes.
        midpoints = [x[j]+(x[j+1]-x[j])/2 for j in 1:n]
        if !all(x[j] < midpoints[j] < x[j+1] for j in 1:n)
            push!(intervals,(a,b))
            return (S,Inf,false)
        end
        fine_x = Vector{Float64}(undef,2n+1)
        fine_y = similar(y,2n+1)
        fine_x[1:2:end],fine_x[2:2:end] = x,midpoints
        fine_y[1:2:end],fine_y[2:2:end] = y,sample.(midpoints)
        # The first and last n+1 entries are the two child grids.
        xl,xr = view(fine_x,1:n+1),view(fine_x,n+1:2n+1)
        yl,yr = view(fine_y,1:n+1),view(fine_y,n+1:2n+1)
        local c = fine_x[n+1]
        SL, SR = composite(a,c,yl),composite(c,b,yr)
        delta = SL+SR-S
        estimate = abs(delta)/15
        if estimate <= tol || depth == 0
            push!(intervals,(a,b))
            return (SL+SR+delta/15,estimate,estimate <= tol)
        end
        left = refine(xl,yl,SL,tol/2,depth-1)
        right = refine(xr,yr,SR,tol/2,depth-1)
        return (left[1]+right[1],left[2]+right[2],left[3] && right[3])
    end
    a,b = Float64(a),Float64(b)
    x = collect(range(a,b,length=n+1))
    if !all(x[j] < x[j+1] for j in 1:n)
        # There are too few representable points for the requested grid.
        value = (b-a)/2*(sample(a)+sample(b))
        return (value=value,estimated_error=Inf,evaluations=evaluations[],
                intervals=[(a,b)],converged=false)
    end
    y = sample.(x)
    value,estimated_error,converged = refine(x,y,composite(a,b,y),Float64(atol),maxdepth)
    return (;value,estimated_error,evaluations=evaluations[],intervals,converged)
end

"""uniform_simpson(f, a, b, n=4; atol=1e-8, maxlevels=20)

Double an initially even global subinterval count n until |S_2n-S_n|/15 <= atol.
Reuse all sampled values and return the Richardson-corrected approximation.
The history records cumulative evaluations and estimated errors at each level.
As with adaptive_simpson, the estimate is heuristic, not a certified bound."""
function uniform_simpson(f, a, b, n::Integer=4; atol=1e-8, maxlevels::Integer=20)
    isfinite(a) && isfinite(b) && a < b || throw(ArgumentError("Require finite a < b."))
    n >= 2 && iseven(n) || throw(ArgumentError("n must be at least 2 and even."))
    isfinite(atol) && atol > 0 || throw(ArgumentError("atol must be finite and positive."))
    maxlevels >= 1 || throw(ArgumentError("maxlevels must be positive."))
    evaluations = 0
    function sample(x)
        value = f(x)
        isfinite(value) || throw(DomainError(x,"Nonfinite integrand value."))
        evaluations += 1
        return value
    end
    function from_values(y)
        m = length(y)-1
        total = y[1]+y[end]
        for j in 1:m-1
            total += (isodd(j) ? 4 : 2)*y[j+1]
        end
        return (b-a)*total/(3m)
    end
    y = sample.(range(Float64(a),Float64(b),length=n+1))
    coarse = from_values(y)
    history = NamedTuple[]
    for level in 1:maxlevels
        refined = Vector{Float64}(undef,2n+1)
        refined[1:2:end] = y
        refined[2:2:end] = [sample(a+(j-0.5)*(b-a)/n) for j in 1:n]
        fine = from_values(refined)
        delta = fine-coarse
        estimated_error = abs(delta)/15
        value = fine+delta/15
        n *= 2
        push!(history,(n=n,calls=evaluations,estimated_error=estimated_error,value=value))
        if estimated_error <= atol || level == maxlevels
            return (value=value,estimated_error=estimated_error,evaluations=evaluations,
                    subintervals=n,converged=estimated_error <= atol,history=history)
        end
        y, coarse = refined, fine
    end
end


"""Gaussian nodes and weights for a standard positive weight.
Pass n+1 as num_nodes for nodes x_0,...,x_n (precision 2n+1).
:legendre on [-1,1], :chebyshev with 1/sqrt(1-x^2),
:laguerre with exp(-x) on [0,Inf), :hermite with exp(-x^2) on R.
The function values multiplied by the weights must NOT include the weight again.
Uses the symmetric Jacobi matrix (Golub-Welsch); no extra package needed."""
function gauss_rule(family::Symbol, num_nodes::Integer)
    num_nodes >= 1 || throw(ArgumentError("num_nodes is the positive node count."))
    family in (:legendre,:chebyshev,:laguerre,:hermite) ||
        throw(ArgumentError("Unknown Gaussian weight family."))
    if family == :chebyshev
        nodes = sort([cos((2j-1)*pi/(2num_nodes)) for j in 1:num_nodes])
        return (nodes=nodes,weights=fill(pi/num_nodes,num_nodes))
    end
    diagonal = family == :laguerre ? [2.0k+1 for k in 0:num_nodes-1] : zeros(num_nodes)
    off = family == :legendre ? [k/sqrt(4.0k^2-1) for k in 1:num_nodes-1] :
          family == :laguerre ? Float64.(1:num_nodes-1) : [sqrt(k/2) for k in 1:num_nodes-1]
    mass = family == :legendre ? 2.0 : family == :laguerre ? 1.0 : sqrt(pi)
    E = eigen(SymTridiagonal(diagonal,off))
    return (nodes=E.values,weights=mass.*E.vectors[1,:].^2)
end

"""Gauss-Legendre integration on [a,b]; pass n+1 for nodes x_0,...,x_n."""
function gauss_legendre(f, a, b, num_nodes::Integer)
    isfinite(a) && isfinite(b) && a < b || throw(ArgumentError("Require finite a < b."))
    rule = gauss_rule(:legendre,num_nodes)
    c, d = (a+b)/2,(b-a)/2
    return d*sum(rule.weights.*f.(c .+ d.*rule.nodes))
end

"""Lanczos Gaussian quadrature for z' f(A) z, with real symmetric A.
For log, A must be positive definite. Stores at most m Lanczos vectors and
uses two passes of reorthogonalization for this classroom example.
Returns the approximation and the Gaussian rule for the probe's spectral measure."""
function lanczos_quadrature(f, A::AbstractMatrix{<:Real}, z::AbstractVector{<:Real}, m::Integer)
    size(A,1) == size(A,2) == length(z) || throw(DimensionMismatch("A and z must agree."))
    issymmetric(A) || throw(ArgumentError("A must be real symmetric."))
    1 <= m <= length(z) || throw(ArgumentError("Use 1 <= m <= length(z)."))
    mass = dot(z,z)
    isfinite(mass) && mass > 0 || throw(ArgumentError("z must be nonzero and finite."))
    Q = zeros(length(z),m)
    q = Float64.(z)/sqrt(mass)
    previous = zeros(length(z))
    alpha, beta = Float64[], Float64[]
    beta_previous = 0.0
    for j in 1:m
        Q[:,j] = q
        Aq = A*q
        r = Aq-beta_previous*previous
        aj = dot(q,r)
        r -= aj*q
        # Two passes reduce loss of orthogonality in small finite-precision examples.
        for pass in 1:2
            for i in 1:j
                qi = view(Q,:,i)
                r -= dot(qi,r)*qi
            end
        end
        push!(alpha,aj)
        bj = norm(r)
        if j == m || bj <= 100eps(Float64)*max(norm(Aq),1.0)
            break
        end
        push!(beta,bj)
        previous,q = q,r/bj
        beta_previous = bj
    end
    E = eigen(SymTridiagonal(alpha,beta))
    weights = mass.*E.vectors[1,:].^2
    return (value=sum(weights.*f.(E.values)),nodes=E.values,weights=weights)
end

# APPROXIMATION, DIFFERENTIATION, AND QUADRATURE (LECTURES 5-6)
# These routines use no additional external packages.
"""Composite trapezoidal rule with n equal subintervals."""
function trapezoid(f, a, b, n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive."))
    a < b || throw(ArgumentError("Require a < b."))
    h = (b-a)/n
    return h*((f(a)+f(b))/2 + sum((f(a+i*h) for i in 1:n-1); init=zero(f(a))))
end

"""Composite Simpson rule. n counts subintervals and must be positive and even."""
function simpson(f, a, b, n::Integer)
    n >= 2 && iseven(n) || throw(ArgumentError("n must be positive and even."))
    a < b || throw(ArgumentError("Require a < b."))
    h = (b-a)/n
    return h/3*(f(a)+f(b)+
        4sum(f(a+i*h) for i in 1:2:n-1)+
        2sum((f(a+i*h) for i in 2:2:n-2); init=zero(f(a))))
end

"""Standard normalization: Legendre P_n, Chebyshev T_n, Laguerre L_n,
or physicists' Hermite H_n. n is the degree, not the number of terms."""
function orthopoly(family::Symbol, n::Integer, x)
    n >= 0 || throw(ArgumentError("The degree must be nonnegative."))
    family in (:legendre,:chebyshev,:laguerre,:hermite) ||
        throw(ArgumentError("Unknown polynomial family."))
    p0 = one(x)
    n == 0 && return p0
    p1 = family == :laguerre ? one(x)-x : family == :hermite ? 2x : x
    for k in 1:n-1
        p2 = if family == :legendre
            ((2k+1)*x*p1-k*p0)/(k+1)
        elseif family == :chebyshev
            2x*p1-p0
        elseif family == :laguerre
            ((2k+1-x)*p1-k*p0)/(k+1)
        else
            2x*p1-2k*p0
        end
        p0,p1 = p1,p2
    end
    return p1
end

function polynomial_design(nodes, n::Integer; basis=:monomial)
    n >= 0 || throw(ArgumentError("The degree must be nonnegative."))
    return [basis == :monomial ? x^j : orthopoly(basis,j,x)
            for x in nodes, j in 0:n]
end

"""Numerical continuous projection on [-1,1]. Legendre uses weight 1;
Chebyshev uses weight 1/sqrt(1-x^2), evaluated by x=cos(theta).
The coefficients approximate the exact inner products using Simpson quadrature."""
function orthogonal_projection(f, n::Integer; family=:legendre, panels=2000)
    n >= 0 || throw(ArgumentError("The degree must be nonnegative."))
    family in (:legendre,:chebyshev) ||
        throw(ArgumentError("Projection supports Legendre and Chebyshev."))
    if family == :legendre
        return [(2j+1)/2*simpson(x -> f(x)*orthopoly(:legendre,j,x),
                               -1.0,1.0,panels) for j in 0:n]
    end
    return [(j == 0 ? 1/pi : 2/pi)*
            simpson(t -> f(cos(t))*cos(j*t),0.0,Float64(pi),panels) for j in 0:n]
end

function orthogonal_value(coefficients, x; family=:legendre)
    isempty(coefficients) && throw(ArgumentError("At least one coefficient is required."))
    return sum(coefficients[j+1]*orthopoly(family,j,x)
               for j in 0:length(coefficients)-1)
end

"""Update the two orthonormal shifted-Legendre coefficients on [0,T].
The input u is held constant on [t,T); c initially represents [0,t].
This is the exact degree-1 HiPPO-LegS update for that held input, up to
roundoff, not an exact integral of an arbitrary signal between samples.
At t=0 the first interval is constant, so initialize c=[u,0].
Only the current two floating-point coefficients are stored and overwritten.
"""
function hippo_linear_update!(c::AbstractVector, u::Real, t::Real, T::Real)
    length(c) == 2 || throw(DimensionMismatch("Use c=[c_0,c_1]."))
    isfinite(t) && isfinite(T) && 0 <= t < T ||
        throw(ArgumentError("Require finite times with 0 <= t < T."))
    if iszero(t)
        c[1],c[2] = u,zero(u)
        return c
    end
    alpha = t/T
    c0,c1 = c  # Both right-hand sides use the OLD coefficients.
    c[1] = alpha*c0+(1-alpha)*u
    c[2] = alpha^2*c1+sqrt(3)*alpha*(1-alpha)*(u-c0)
    return c
end

"""Online HiPPO-LegS update for any number of orthonormal coefficients.
Hold the current input u constant on [t,T), and overwrite the current c.
For zero-based indices i,j, c' = (-A*c+b*u)/t, where A_ii=i+1,
A_ij=sqrt((2i+1)*(2j+1)) for i>j, A_ij=0 for i<j, b_i=sqrt(2i+1).
Since A*e_0=b, the exact held-input step is
c(T)=u*e_0+exp(-log(T/t)*A)*(c(t)-u*e_0).
The matrix exponential accounts for the expanding interval and changing basis.
At t=0, initialize [u,0,...,0]. No past samples or full integrals are needed.
This teaching implementation forms small matrices at each step. Its working
storage depends on length(c), never on the number of samples already seen.
Source: Gu et al., HiPPO (2020), Appendix D.3, https://arxiv.org/abs/2008.07669.
"""
function hippo_update!(c::AbstractVector, u::Real, t::Real, T::Real)
    isempty(c) && throw(ArgumentError("Store at least one coefficient."))
    isfinite(t) && isfinite(T) && 0 <= t < T ||
        throw(ArgumentError("Require finite times with 0 <= t < T."))
    if iszero(t)
        fill!(c,zero(eltype(c)))
        c[1] = u
        return c
    end
    m = length(c)
    A = [i == j ? Float64(i) : i > j ? sqrt((2i-1)*(2j-1)) : 0.0
         for i in 1:m, j in 1:m]
    shifted = copy(c)
    shifted[1] -= u
    c .= exp(-log1p((T-t)/t)*A)*shifted
    c[1] += u
    return c
end

"""Reconstruct the stored history at s in [0,t], with t>0.
c contains coefficients of sqrt(2j+1)*P_j(2s/t-1), j=0,...,length(c)-1.
"""
function hippo_value(c::AbstractVector, s::Real, t::Real)
    isempty(c) && throw(ArgumentError("Store at least one coefficient."))
    isfinite(t) && t > 0 && 0 <= s <= t ||
        throw(ArgumentError("Require 0 <= s <= t with finite t > 0."))
    return sum(c[j+1]*sqrt(2j+1)*orthopoly(:legendre,j,2s/t-1)
               for j in 0:length(c)-1)
end

"""Padé [n/m] from Taylor coefficients a_0,...,a_(n+m).
Returns (p,q) in increasing powers, with q_0=1. A singular system may
mean that this normalized construction is unavailable; it is not regularized."""
function pade_coefficients(a, n::Integer, m::Integer)
    n >= 0 && m >= 0 || throw(ArgumentError("Degrees must be nonnegative."))
    length(a) >= n+m+1 || throw(DimensionMismatch("Need n+m+1 Taylor coefficients."))
    ak(k) = k < 0 ? zero(a[1]) : a[k+1]
    if m == 0
        return (p=collect(a[1:n+1]),q=[one(a[1])])
    end
    A = [ak(n+r-j) for r in 1:m, j in 1:m]
    tail = A \ [-ak(n+r) for r in 1:m]
    q = vcat(one(eltype(tail)),tail)
    p = [sum(q[j+1]*ak(k-j) for j in 0:min(k,m)) for k in 0:n]
    return (p=p,q=q)
end

"""Small-stencil weights: sum_j w_j f(x+h*s_j)/h^order.
Solves polynomial moment equations; not intended for large ill-conditioned stencils."""
function finite_difference_weights(offsets, order::Integer=1)
    N = length(offsets)
    0 <= order < N || throw(ArgumentError("Require 0 <= order < number of nodes."))
    length(unique(offsets)) == N || throw(ArgumentError("Offsets must be distinct."))
    A = [s^k for k in 0:N-1, s in float.(offsets)]
    b = zeros(eltype(A),N)
    b[order+1] = factorial(order)
    return A \ b
end

function difference(f, x, h; scheme=:central)
    h > 0 || throw(ArgumentError("h must be positive."))
    scheme == :forward && return (f(x+h)-f(x))/h
    scheme == :backward && return (f(x)-f(x-h))/h
    scheme == :central && return (f(x+h)-f(x-h))/(2h)
    scheme == :second && return (f(x+h)-2f(x)+f(x-h))/h^2
    throw(ArgumentError("Unknown difference scheme."))
end

"""Richardson table for central first differences at h, h/2, ...
Column j has formal order 2j for a sufficiently smooth f before roundoff dominates."""
function richardson_table(f, x, h; levels=5)
    levels >= 1 || throw(ArgumentError("levels must be positive."))
    h > 0 || throw(ArgumentError("h must be positive."))
    R = fill(NaN,levels,levels)
    for k in 1:levels
        R[k,1] = difference(f,x,h/2^(k-1))
        for j in 2:k
            R[k,j] = R[k,j-1]+(R[k,j-1]-R[k-1,j-1])/(4^(j-1)-1)
        end
    end
    return R
end

"""Integrated gradients along a straight path. Supply an analytic or AD
gradient. This helper performs quadrature only; it does not differentiate F."""
function integrated_gradients(gradient, x, baseline, n::Integer; rule=:simpson)
    length(x) == length(baseline) || throw(DimensionMismatch("Input and baseline sizes differ."))
    rule in (:simpson,:trapezoid) || throw(ArgumentError("Unknown quadrature rule."))
    delta = x-baseline
    integrator = rule == :simpson ? simpson : trapezoid
    return [delta[i]*integrator(t -> gradient(baseline+t*delta)[i],0.0,1.0,n)
            for i in eachindex(x)]
end

# DATA CHECKS AND GLOBAL POLYNOMIAL INTERPOLATION
function check_nodes(xx)
    isempty(xx) && throw(ArgumentError("At least one node is required."))
    all(isfinite, xx) || throw(ArgumentError("Nodes must be finite."))
    length(unique(xx)) == length(xx) ||
        throw(ArgumentError("Interpolation nodes must be distinct."))
    return nothing
end

function check_data(xx, yy)
    check_nodes(xx)
    length(xx) == length(yy) ||
        throw(DimensionMismatch("Each node needs one value."))
    all(isfinite, yy) || throw(ArgumentError("Values must be finite."))
    return nothing
end

# L_(n,j): mathematical index j=0,...,n (stored at Julia index j+1).
function L(nodes, j, z)
    check_nodes(nodes)
    n = length(nodes)-1
    0 <= j <= n || throw(ArgumentError("Use j=0,...,n."))
    value = one(z-nodes[1])
    for i in eachindex(nodes)
        i == j+1 && continue
        value *= (z-nodes[i])/(nodes[j+1]-nodes[i])
    end
    return value
end

# Compatibility with the instructor checks, whose basis index starts at 1.
function lagrange_basis(nodes, j, z)
    j in eachindex(nodes) || throw(BoundsError(nodes,j))
    return L(nodes,j-1,z)
end

function lagrange(nodes, values, z)
    check_data(nodes,values)
    return sum(values[j+1]*L(nodes,j,z) for j in 0:length(nodes)-1)
end

# Newton coefficients and nested evaluation.

function divdiff(xx, yy)
    check_data(xx, yy)
    cc = collect(float.(yy))
    for order in 1:(length(xx)-1)
        # Backward updates preserve the previous-order entries cc[j-1].
        for j in length(xx):-1:(order+1)
            cc[j] = (cc[j] - cc[j-1]) / (xx[j] - xx[j-order])
        end
    end
    return cc
end

function newton_value(xx, cc, z)
    check_data(xx, cc)
    value = cc[end]
    for j in (length(cc)-1):-1:1
        value = cc[j] + (z - xx[j]) * value
    end
    return value
end

# Global Hermite polynomial H_(2n+1), using values and first derivatives.

function hermite(nodes, values, slopes, z)
    length(nodes) == length(values) == length(slopes) ||
        throw(DimensionMismatch("One value and one slope per node are required."))
    check_data(nodes,values)
    length(nodes) == 1 && return values[1]+slopes[1]*(z-nodes[1])
    return sum(begin
        Lj = L(nodes,j,z)
        xj = nodes[j+1]
        dLj = sum(1/(xj-nodes[i+1]) for i in 0:length(nodes)-1 if i != j)
        Hj = (1-2*(z-xj)*dLj)*Lj^2
        Hhatj = (z-xj)*Lj^2
        values[j+1]*Hj + slopes[j+1]*Hhatj
    end for j in 0:length(nodes)-1)
end

# Independent small-degree coefficient solve, for checking the Hermite formula.

function hermite_coefficients(nodes, values, slopes)
    m = length(nodes)
    A = zeros(2m,2m)
    rhs = zeros(2m)
    for i in 1:m
        for k in 0:(2m-1)
            A[2i-1,k+1] = nodes[i]^k
            A[2i,k+1] = k == 0 ? 0.0 : k*nodes[i]^(k-1)
        end
        rhs[2i-1] = values[i]
        rhs[2i] = slopes[i]
    end
    return A \ rhs
end

# DIVIDED-DIFFERENCE COMPLEXITY EXPERIMENTS
# Direct recursion repeats overlapping subproblems; the table uses O(n^2) updates.

function dd_last_recursive(xx, yy, lo=1, hi=length(xx))
    lo == hi && return yy[lo]
    return (dd_last_recursive(xx,yy,lo+1,hi) -
            dd_last_recursive(xx,yy,lo,hi-1)) / (xx[hi]-xx[lo])
end

function dd_last_table(xx, yy)
    cc = copy(yy)
    nlocal = length(xx)-1
    for k in 1:nlocal
        for j in (nlocal+1):-1:(k+1)
            cc[j] = (cc[j]-cc[j-1]) / (xx[j]-xx[j-k])
        end
    end
    return cc[end]
end

function dd_seconds(fun, xx, yy; batch=10)
    fun(xx,yy)
    best = Inf
    for trial in 1:5
        elapsed = @elapsed begin
            for repetition in 1:batch
                fun(xx,yy)
            end
        end
        best = min(best, elapsed/batch)
    end
    return best
end

function dd_cost_experiment(ns=[4,8,12,16,20])
    # Keep the exponential experiment small enough for a live class.
    all(k -> 1 <= k <= 20, ns) ||
        throw(ArgumentError("Use orders between 1 and 20 for direct recursion."))
    println("n | recursive updates | table updates | recursive ms | table ms")
    results = NamedTuple[]
    for order in ns
        # Generate data outside the timing. Spacing 1 avoids tiny denominators.
        xx = Float64.(0:order)
        yy = sin.(xx)
        recursive_value = dd_last_recursive(xx,yy)
        table_value = dd_last_table(xx,yy)
        @assert isapprox(recursive_value,table_value; atol=1e-14,rtol=1e-10)
        recursive_ms = 1000*dd_seconds(dd_last_recursive,xx,yy)
        table_ms = 1000*dd_seconds(dd_last_table,xx,yy)
        row = (n=order, recursive_updates=2^order-1,
               table_updates=order*(order+1)÷2,
               recursive_ms=recursive_ms, table_ms=table_ms)
        push!(results,row)
        println(order, " | ", row.recursive_updates, " | ", row.table_updates,
                " | ", round(recursive_ms,sigdigits=3),
                " | ", round(table_ms,sigdigits=3))
    end
    return results
end

function dd_table_scaling(ns=[128,256,512,1024])
    previous_ms = nothing
    for order in ns
        xx = Float64.(0:order)
        yy = sin.(xx)
        ms = 1000*dd_seconds(dd_last_table,xx,yy; batch=20)
        ratio = isnothing(previous_ms) ? missing : ms/previous_ms
        println((n=order, milliseconds=ms, ratio_to_previous=ratio))
        previous_ms = ms
    end
end

# PIECEWISE INTERPOLATION AND CUBIC SPLINES
# Locate the interval containing z, including the rightmost endpoint.

function interval_index(nodes,z)
    first(nodes) <= z <= last(nodes) || throw(DomainError(z,"Outside the node interval."))
    return clamp(searchsortedlast(nodes,z),1,length(nodes)-1)
end

function piecewise_linear(xx, yy, z)
    check_data(xx, yy)
    length(xx) >= 2 || throw(ArgumentError("At least two nodes are required."))
    issorted(xx) || throw(ArgumentError("Nodes must be strictly increasing."))
    i = interval_index(xx,z)
    t = (z - xx[i]) / (xx[i+1] - xx[i])
    return (1-t)*yy[i] + t*yy[i+1]
end

# The local Hermite formula also evaluates a cubic spline from its nodal slopes.

function piecewise_hermite(nodes,values,slopes,z)
    i = interval_index(nodes,z)
    delta = nodes[i+1]-nodes[i]
    t = (z-nodes[i])/delta
    return (1+2t)*(1-t)^2*values[i]+t^2*(3-2t)*values[i+1] +
           delta*t*(1-t)^2*slopes[i]+delta*t^2*(t-1)*slopes[i+1]
end

# Solve for m_i = S'(x_i), not second derivatives. Julia entry i+1 stores m_i.

function spline_slopes(nodes,values; boundary=:natural, endpoint_slopes=nothing)
    length(nodes) == length(values) >= 2 || throw(DimensionMismatch("Need matching data at >=2 nodes."))
    all(diff(nodes) .> 0) || throw(ArgumentError("Nodes must be strictly increasing."))
    N = length(nodes)
    delta = diff(nodes)
    d = diff(values)./delta
    lower, upper = zeros(N-1), zeros(N-1)
    diagonal, rhs = fill(2.0,N), zeros(N)
    for i in 2:N-1
        lambda = delta[i-1]/(delta[i-1]+delta[i])
        lower[i-1], upper[i] = 1-lambda, lambda
        rhs[i] = 3*((1-lambda)*d[i-1]+lambda*d[i])
    end
    if boundary == :natural
        # 2m_0 + m_1 = 3d_0; m_(n-1) + 2m_n = 3d_(n-1).
        upper[1], lower[end] = 1, 1
        rhs[1], rhs[end] = 3d[1], 3d[end]
    elseif boundary == :clamped
        endpoint_slopes === nothing &&
            throw(ArgumentError("Supply endpoint_slopes=(fprime(a),fprime(b))."))
        diagonal[1] = diagonal[end] = 1
        rhs[1], rhs[end] = endpoint_slopes
    else
        throw(ArgumentError("Use :natural or :clamped."))
    end
    return Tridiagonal(lower,diagonal,upper) \ rhs
end

# B-SPLINES AND BARYCENTRIC INTERPOLATION
# Knots are strictly increasing, as assumed in the lectures.
# Julia index i represents N_(i-1,p), where p is the degree.
# With distinct knots, degree p >= 1 gives C^(p-1) continuity.
function bspline_basis(x, i, p, knots)
    p >= 0 || throw(ArgumentError("Degree must be nonnegative."))
    n_basis = length(knots)-p-1
    1 <= i <= n_basis || return 0.0
    if p == 0
        return knots[i] <= x < knots[i+1] ? 1.0 : 0.0
    end

    left_den = knots[i+p]-knots[i]
    right_den = knots[i+p+1]-knots[i+1]
    return (x-knots[i])/left_den * bspline_basis(x,i,p-1,knots) +
           (knots[i+p+1]-x)/right_den * bspline_basis(x,i+1,p-1,knots)
end

# nsegments intervals on [a,b], plus p distinct exterior knots on each side.
# Slides: p is the degree and n=nsegments; a=t_p and b=t_(n+p).
# Knots t_0,...,t_(n+2p) give n+p basis functions spanning S_p^(p-1).
# No endpoints are repeated; natural/clamped conditions are imposed separately.
function uniform_knots(nsegments, p; a=0.0, b=1.0)
    nsegments >= 1 || throw(ArgumentError("Need at least one interval."))
    p >= 0 || throw(ArgumentError("Degree must be nonnegative."))
    a < b || throw(ArgumentError("Require a < b."))
    h = (b-a)/nsegments
    interior = collect(range(a,b,length=nsegments+1))
    return vcat(a .- h .* (p:-1:1),interior,b .+ h .* (1:p))
end

function chebyshev_roots(n)
    n >= 1 || throw(ArgumentError("Need n >= 1."))
    return [-cos((2j-1)*pi/(2n)) for j in 1:n]
end

function barycentric_weights(nodes)
    return [inv(prod(nodes[i]-nodes[j] for j in eachindex(nodes) if j != i))
            for i in eachindex(nodes)]
end

function barycentric_value(nodes, values, weights, x)
    for i in eachindex(nodes)
        x == nodes[i] && return values[i]
    end
    terms = weights ./ (x .- nodes)
    return sum(terms .* values)/sum(terms)
end

# REMEZ APPROXIMATION OF RELU ON [-1,1]
# ReLU(x) = (x + abs(x))/2. The unique minimax polynomial of degree <= 7
# therefore has the form x/2 + c0 + c2*x^2 + c4*x^4 + c6*x^6.
# Remez exchange on [0,1] uses five alternating points in the even basis.
# Reflecting those points gives nine alternating extrema on [-1,1].
function remez_relu7(; tol=1e-13, maxiter=25)
    tol > 0 || throw(ArgumentError("Require tol > 0."))
    maxiter >= 1 || throw(ArgumentError("Require maxiter >= 1."))
    reference = [(1-cospi(j/4))/2 for j in 0:4]
    alternating_signs = [(-1.0)^j for j in 0:4]

    for iteration in 1:maxiter
        # Enforce x/2 - sum(c[j+1]*x^(2j)) = (-1)^k * signed_level.
        A = hcat([x^(2j) for x in reference, j in 0:3], alternating_signs)
        solution = A \ (reference ./ 2)
        c, signed_level = solution[1:4], solution[5]

        # On (0,1), the error derivative is
        # 1/2 - 2c2*x - 4c4*x^3 - 6c6*x^5.
        # Its roots are the eigenvalues of this companion matrix.
        derivative_coeffs = [0.5,-2c[2],0.0,-4c[3],0.0,-6c[4]]
        companion = zeros(5,5)
        companion[2:5,1:4] .= Matrix{Float64}(I,4,4)
        companion[:,5] .= -derivative_coeffs[1:5] ./ derivative_coeffs[6]
        critical = sort([real(z) for z in eigvals(companion)
                         if abs(imag(z)) < 1e-10 && 0 < real(z) < 1])
        length(critical) == 3 || error("Expected three interior error extrema.")
        reference = vcat(0.0,critical,1.0)
        positive_errors = [x/2-evalpoly(x^2,c) for x in reference]
        residual = maximum(abs.(positive_errors .- signed_level .* alternating_signs))

        if residual <= tol
            # Ascending powers, including the zero coefficient of x^7.
            coefficients = [c[1],0.5,c[2],0.0,c[3],0.0,c[4],0.0]
            extrema = vcat(-reverse(reference[2:end]),reference)
            errors = [max(0.0,x)-evalpoly(x,coefficients) for x in extrema]
            return (coefficients=coefficients, max_error=maximum(abs,errors),
                    extrema=extrema, errors=errors,
                    equioscillation_residual=residual, iterations=iteration)
        end
    end
    error("Remez exchange did not converge within maxiter iterations.")
end

function plot_relu_remez7(result=remez_relu7())
    grid = sort(unique(vcat(collect(range(-1.0,1.0,length=4001)),result.extrema)))
    target = max.(0.0,grid)
    approximation = [evalpoly(x,result.coefficients) for x in grid]
    E = result.max_error
    common = (xlabel="x",xlims=(-1.04,1.04),xticks=-1:0.5:1,
              guidefontsize=18,tickfontsize=15,titlefontsize=19,
              legendfontsize=14)
    values = plot(grid,target;common...,label="ReLU(x)",color=:royalblue,
                  linewidth=3,ylabel="Function value",legend=:topleft,
                  ylims=(-0.08,1.08),title="ReLU on [-1,1]: degree ≤ 7")
    plot!(values,grid,approximation;label="Remez polynomial P*(x)",
          color=:darkorange,linewidth=2.5,linestyle=:dash)

    errors = plot(grid,target.-approximation;common...,legend=false,
                  color=:purple,linewidth=2.5,ylabel="ReLU(x) - P*(x)",
                  ylims=(-1.32E,1.32E),yticks=([-E,0,E],["-E","0","+E"]),
                  title="9 alternating extrema: E ≈ $(round(E,digits=8))")
    hline!(errors,[-E,E];label="",color=:firebrick,linestyle=:dash,linewidth=1.5)
    hline!(errors,[0.0];label="",color=:gray,linewidth=0.7)
    scatter!(errors,result.extrema,result.errors;label="",color=:firebrick,
             markershape=:circle,markersize=5,markerstrokewidth=0.7)
    return plot(values,errors;layout=(1,2),size=(1600,620),
                left_margin=11Plots.mm,right_margin=6Plots.mm,
                top_margin=7Plots.mm,bottom_margin=10Plots.mm)
end

# RUNGE EXAMPLES AND PLOTTING HELPERS
# Lecture 3 uses R on [-5,5]; Lecture 4 uses runge(x)=R(5x) on [-1,1].

R(z) = 1/(1+z^2)
dR(z) = -2z/(1+z^2)^2
runge(x) = R(5x)

function runge_errors(n; high_precision=false, piecewise=false)
    T = high_precision ? BigFloat : Float64
    # Construct directly in the requested precision, not from rounded Float64 data.
    xx = collect(range(T(-5), T(5), length=n+1))
    grid = range(T(-5), T(5), length=2001)
    yy = R.(xx)
    evaluate = piecewise ? piecewise_linear : lagrange
    nodal = maximum(abs(evaluate(xx, yy, z)-R(z)) for z in xx)
    sampled = maximum(abs(evaluate(xx, yy, z)-R(z)) for z in grid)
    return (degree_or_intervals=n, nodal_residual=nodal, grid_error=sampled)
end

function plot_hermite_sine(m)
    nodes = collect(range(0.0,pi/2,length=m))
    values = sin.(nodes)
    slopes = cos.(nodes)
    grid = range(0.0,pi/2,length=1001)
    interpolated = [hermite(nodes,values,slopes,z) for z in grid]
    signed_error = sin.(grid) .- interpolated
    degree_bound = 2m-1

    # Verify the expanded polynomial and its analytic derivative.
    coeffs = hermite_coefficients(nodes,values,slopes)
    derivative_coeffs = [k*coeffs[k+1] for k in 1:degree_bound]
    value_residual = maximum(abs(evalpoly(z,coeffs)-sin(z)) for z in nodes)
    slope_residual = maximum(abs(evalpoly(z,derivative_coeffs)-cos(z)) for z in nodes)
    @assert value_residual < 1e-11
    @assert slope_residual < 1e-11
    @assert maximum(abs(evalpoly(z,coeffs)-hermite(nodes,values,slopes,z))
                    for z in grid) < 1e-10
    @assert maximum(abs(hermite(nodes,values,slopes,z)-sin(z))
                    for z in nodes) < 1e-12
    println((nodes=m, degree_at_most=degree_bound,
             coefficients=coeffs, value_residual=value_residual,
             derivative_residual=slope_residual,
             maximum_sampled_error=maximum(abs.(signed_error))))

    tick_positions = m == 2 ? [0,pi/2] : m == 3 ? [0,pi/4,pi/2] : [0,pi/6,pi/3,pi/2]
    tick_labels = m == 2 ? ["0","pi/2"] : m == 3 ? ["0","pi/4","pi/2"] : ["0","pi/6","pi/3","pi/2"]
    curve_plot = plot(grid,sin.(grid);
        label="f(x) = sin(x)", color=:blue, linewidth=2.5,
        xlabel="x", ylabel="Function value", title="$m nodes: H_$degree_bound(x)",
        xticks=(tick_positions,tick_labels),
        xlims=(-0.03,pi/2+0.03), ylims=(-0.04,1.08),
        legend=:bottom, margin=5 * Plots.mm)
    plot!(curve_plot,grid,interpolated;
        label="Hermite H_$degree_bound(x)", color=:darkorange,
        linestyle=:dash, linewidth=2.5)
    scatter!(curve_plot,nodes,values;
        label="Nodes", color=:black, markershape=:circle, markersize=5)

    error_plot = plot(grid,signed_error;
        label=false, color=:purple, linewidth=2,
        xlabel="x", ylabel="f(x) - H_$degree_bound(x)",
        title="Interpolation error",
        xticks=(tick_positions,tick_labels),
        xlims=(-0.03,pi/2+0.03), margin=5 * Plots.mm)
    hline!(error_plot,[0.0]; label=false, color=:gray, linestyle=:dot)
    scatter!(error_plot,nodes,zeros(m);
        label=false, color=:black, markershape=:circle, markersize=4)

    result = plot(curve_plot,error_plot;
        layout=(1,2), size=(1150,420), titlefontsize=12,
        guidefontsize=10, tickfontsize=9, legendfontsize=9)
    display(result)
    return result
end

function plot_hermite_lagrange_sine(m)
    m in (2,3,4) || throw(ArgumentError("Use 2, 3, or 4 nodes."))
    nodes = collect(range(0.0,pi/2,length=m))
    values, slopes = sin.(nodes), cos.(nodes)
    grid = range(0.0,pi/2,length=1001)
    hv = [hermite(nodes,values,slopes,z) for z in grid]
    pv = [lagrange(nodes,values,z) for z in grid]
    eh, ep = sin.(grid).-hv, sin.(grid).-pv
    dh, dp = 2m-1, m-1
    ticks = m == 2 ? ["0","pi/2"] :
            m == 3 ? ["0","pi/4","pi/2"] : ["0","pi/6","pi/3","pi/2"]
    @assert maximum(abs(hermite(nodes,values,slopes,z)-sin(z)) for z in nodes) < 1e-12
    @assert maximum(abs(lagrange(nodes,values,z)-sin(z)) for z in nodes) < 1e-12

    curve = plot(grid,sin.(grid);
        label="sin(x)",color=:blue,linewidth=2.5,
        title="$m nodes: sin(x) and H_$dh(x)",xlabel="x",ylabel="value",
        xticks=(nodes,ticks),xlims=(-0.03,pi/2+0.03),ylims=(-0.04,1.08),
        legend=:bottomright)
    plot!(curve,grid,hv;label="Hermite H_$dh(x)",color=:darkorange,
          linestyle=:dash,linewidth=2.5)
    scatter!(curve,nodes,values;label="Nodes",color=:black,markersize=4)
    ph = plot(grid,eh;label=false,color=:purple,linewidth=2.5,
        title="Hermite error: sin(x) - H_$dh(x)",xlabel="x",ylabel="signed error",
        xticks=(nodes,ticks),xlims=(-0.03,pi/2+0.03),yformatter=:scientific)
    pp = plot(grid,ep;label=false,color=:forestgreen,linewidth=2.5,
        title="Lagrange error: sin(x) - P_$dp(x)",xlabel="x",ylabel="signed error",
        xticks=(nodes,ticks),xlims=(-0.03,pi/2+0.03),yformatter=:scientific)
    for panel in (ph,pp)
        hline!(panel,[0.0];label=false,color=:gray,linestyle=:dot)
        scatter!(panel,nodes,zeros(m);label=false,color=:black,markersize=4)
    end
    # Set margins on the individual panels before composition: otherwise GR
    # can clip guide text at the left and bottom of the exported PDF canvas.
    for panel in (curve,ph,pp)
        plot!(panel;left_margin=10Plots.mm,right_margin=5Plots.mm,
              bottom_margin=10Plots.mm,top_margin=7Plots.mm)
    end
    # Separate linear y-scales preserve the shape of both errors.
    result = plot(curve,ph,pp;layout=(1,3),size=(1500,510),
        titlefontsize=12,guidefontsize=11,tickfontsize=10,legendfontsize=10)
    println((nodes=m,H_degree=dh,P_degree=dp,
             hermite_max_sampled_error=maximum(abs.(eh)),
             lagrange_max_sampled_error=maximum(abs.(ep))))
    return result
end

function plot_runge_hermite(n=10)
    n isa Integer && n >= 1 || throw(ArgumentError("n must be a positive integer."))
    nodes = collect(range(-5.0,5.0,length=n+1))
    values, slopes = R.(nodes), dR.(nodes)
    grid = range(-5.0,5.0,length=4001)
    hv = [hermite(nodes,values,slopes,z) for z in grid]
    rv = maximum(abs(hermite(nodes,values,slopes,z)-R(z)) for z in nodes)
    # Complex-step differentiation checks slopes without subtracting nearby values.
    step = 1e-20
    rd = maximum(abs(imag(hermite(nodes,values,slopes,z+im*step))/step-dR(z)) for z in nodes)
    @assert rv < 1e-10
    @assert rd < 1e-9
    println((n=n,degree_at_most=2n+1,value_residual=rv,slope_residual=rd,
             maximum_sampled_error=maximum(abs.(R.(grid).-hv))))
    p = plot(grid,R.(grid);label="R(x)",color=:blue,linewidth=2.5,
        xlabel="x",ylabel="value",title="n = $n: global Hermite H_$(2n+1)",
        legend=:top,size=(1000,600),left_margin=12Plots.mm,
        right_margin=6Plots.mm,bottom_margin=10Plots.mm,top_margin=7Plots.mm)
    plot!(p,grid,hv;label="Hermite",color=:darkorange,linestyle=:dash,linewidth=2.5)
    scatter!(p,nodes,values;label="Nodes",color=:black,markershape=:circle,markersize=4)
    return p
end

# A single cubic B-spline with five equally spaced local knots on [0,1].
# The slide gives its explicit four-piece formula with u=(x-1/2)/(1/4).
# Julia index 1 here represents the mathematical N_{i,3} on these local knots.
function plot_cubic_bspline()
    knots = collect(range(0.0,1.0,length=5))
    grid = range(-0.08,1.08,length=1161)
    values = [bspline_basis(x,1,3,knots) for x in grid]
    knot_values = [bspline_basis(x,1,3,knots) for x in knots]
    fig = plot(grid,values;label="",color=:blue,linewidth=3,
               xlabel="x",ylabel="Nᵢ,₃(x)",legend=false,
               xticks=(knots,["0","1/4","1/2","3/4","1"]),
               yticks=([0,1/6,2/3],["0","1/6","2/3"]),
               xlims=(-0.08,1.08),ylims=(-0.035,0.73),
               guidefontsize=18,tickfontsize=15,size=(1100,380),
               left_margin=10Plots.mm,right_margin=5Plots.mm,
               bottom_margin=9Plots.mm,top_margin=3Plots.mm)
    vline!(fig,knots;label="",color=:gray,alpha=0.35,linestyle=:dash)
    scatter!(fig,knots,knot_values;label="",color=:darkorange,
             markershape=:circle,markersize=6,markerstrokewidth=1)
    return fig
end

# All cubic B-splines for nsegments original intervals on [0,1].
# For nsegments=2: nine knots t_0,...,t_8 and five basis functions N_0,3,...,N_4,3.
# By default show the restriction to [0,1]. With full_support=true, show
# every support [t_i,t_(i+4)] and shade the original interval [0,1].
# Circles mark partition nodes, or the five local knots in the full view.
function plot_cubic_bspline_basis(nsegments=2; full_support=false)
    p = 3
    knots = uniform_knots(nsegments,p)
    nodes = collect(range(0.0,1.0,length=nsegments+1))
    xmin, xmax = full_support ? (first(knots),last(knots)) : (0.0,1.0)
    grid = range(xmin,xmax,length=full_support ? 1601 : 801)
    tick_nodes = full_support ? knots : nodes
    padding = 0.02*(xmax-xmin)
    nbasis = length(knots)-p-1
    colors = palette(:tab10)
    fig = plot(;xlabel="x",ylabel="Nᵢ,₃(x)",legend=:outerright,
               legendtitle="Basis index",legendfontsize=14,legendtitlefontsize=14,
               xticks=tick_nodes,yticks=([0,1/6,2/3],["0","1/6","2/3"]),
               xlims=(xmin-padding,xmax+padding),
               ylims=(-0.035,full_support ? 0.78 : 0.73),
               guidefontsize=18,tickfontsize=15,size=(1200,460),
               left_margin=10Plots.mm,right_margin=5Plots.mm,
               bottom_margin=9Plots.mm,top_margin=3Plots.mm)
    if full_support
        vspan!(fig,[0.0,1.0];label="",color=:gold,fillalpha=0.18,linewidth=0)
        vline!(fig,[0.0,1.0];label="",color=:gray,alpha=0.8,
               linewidth=1.5,linestyle=:dash)
        annotate!(fig,0.5,0.735,text("[0,1]",16,:black))
    end
    vline!(fig,tick_nodes;label="",color=:gray,alpha=0.25,linestyle=:dash)
    for i in 1:nbasis
        color = colors[mod1(i,length(colors))]
        plot!(fig,grid,[bspline_basis(x,i,p,knots) for x in grid];
              label="i = $(i-1)",color=color,linewidth=3)
        marker_nodes = full_support ? knots[i:i+p+1] : nodes
        scatter!(fig,marker_nodes,[bspline_basis(x,i,p,knots) for x in marker_nodes];
                 label="",color=color,markershape=:circle,markersize=5,
                 markerstrokewidth=0.5)
    end
    return fig
end

# One representative basis function for each degree p=1,2,3,4.
# Select mathematical i=p (Julia index p+1), so t_p=0 and the full support
# is [0,(p+1)h], where h=1/nsegments. Use identical axes in all four panels.
function plot_bspline_degrees(nsegments=6)
    nsegments >= 1 || throw(ArgumentError("Need at least one interval."))
    xmax = max(1.0,5/nsegments)  # Include the degree-4 support for small n too.
    xx = range(0.0,xmax,length=1201)
    basis_labels = ("N₁,₁(x)","N₂,₂(x)","N₃,₃(x)","N₄,₄(x)")
    smoothness_labels = ("C⁰","C¹","C²","C³")
    colors = palette(:tab10)
    panels = map(1:4) do p
        knots = uniform_knots(nsegments,p)
        i = p+1  # Mathematical N_(p,p).
        local_knots = knots[i:i+p+1]
        panel = plot(;xlabel=p >= 3 ? "x" : "",ylabel=basis_labels[p],
                 title="p = $p: $(smoothness_labels[p])",
                 titlefontsize=22,guidefontsize=20,tickfontsize=16,
                 legend=false,xlims=(-0.02xmax,1.02xmax),ylims=(-0.06,1.08),
                 xticks=range(0.0,xmax,length=5),yticks=[0,0.5,1])
        vline!(panel,local_knots;label="",color=:gray,alpha=0.3,linestyle=:dash)
        plot!(panel,xx,[bspline_basis(x,i,p,knots) for x in xx];
              label=basis_labels[p],color=colors[p],linewidth=3)
        scatter!(panel,local_knots,[bspline_basis(x,i,p,knots) for x in local_knots];
                 label="",color=colors[p],markershape=:circle,markersize=5,
                 markerstrokewidth=0.5)
        panel
    end
    return plot(panels...;layout=(2,2),size=(1600,900),left_margin=9Plots.mm,
                right_margin=4Plots.mm,bottom_margin=7Plots.mm,top_margin=6Plots.mm)
end
