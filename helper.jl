# Numerical Analysis: shared functions for Lectures 2-4.
# Each lecture support file loads this with:
#     include(joinpath(@__DIR__, "..", "helper.jl"))
# Keep helper.jl at the repository root, beside Lecture2/, Lecture3/, and Lecture4/.
# Loading this file imports the packages and defines functions only.
# It does not run activities, benchmarks, checks, or figure exports.
# Mathematical nodes x_0,...,x_n are stored at Julia indices 1,...,n+1.
# L uses j=0,...,n; lagrange_basis retains the older one-based interface.
using LinearAlgebra
using Plots

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
