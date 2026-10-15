# Lecture 2: Julia commands for live demonstrations
# From this folder: include("lecture2_support.jl").
# Loading imports the helpers only. Select the code inside one #= ... =# block,
# or uncomment that block and load the file again. Activity 1 includes parts (a)-(c).
# Activity 6 retains the Hermite examples that continue into Lecture 3.
# Shared functions and package imports; the path is relative to this file.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture2_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture2"))


# ACTIVITY 1: Monomial basis, a linear combination, and interpolation coefficients
# Part (a): Visualize the monomial basis on [0,1].
#=
x = range(0.0, 1.0, length=201)

# Dots apply operations element by element to the sampled x values.
b0 = ones(length(x))
b1 = x
b2 = x.^2
b3 = x.^3

basis_plot = plot(x, b0;
    label="1", linewidth=2,
    xlabel="x", ylabel="Basis value",
    title="Monomial basis on [0,1]",
    xlims=(0,1), ylims=(0,1.05),
    legend=:topleft, margin=6 * Plots.mm)
plot!(basis_plot, x, b1; label="x", linewidth=2)
plot!(basis_plot, x, b2; label="x^2", linewidth=2)
plot!(basis_plot, x, b3; label="x^3", linewidth=2)
display(basis_plot)

# Rows correspond to x = 0, 1/2, 1; columns to 1, x, x^2, x^3.
check_x = [0.0, 0.5, 1.0]
basis_values = [z^j for z in check_x, j in 0:3]
display(basis_values)

# Part (b): A linear combination of the basis functions
# P_3(x) = a_0 + a_1*x + a_2*x^2 + a_3*x^3.

a0 = 1.0
a1 = -2.0
a2 = 1.0
a3 = 1.0

P3(z) = a0 + a1*z + a2*z^2 + a3*z^3
p_values = a0 .* b0 .+ a1 .* b1 .+ a2 .* b2 .+ a3 .* b3

combination_plot = plot(x, p_values;label="P_3(x)", linewidth=3, color=:black,
    xlabel="x", ylabel="Polynomial value",
    title="Coefficients: ($a0, $a1, $a2, $a3)",
    xlims=(0,1), legend=:topright, margin=6 * Plots.mm)
scatter!(combination_plot, check_x, P3.(check_x);
    label="Check values", markersize=5)
display(combination_plot)
display(P3.(check_x))
# With the initial coefficients: [1.0, 0.375, 1.0].

# View the basis and its linear combination side by side.
# activity1_plot = plot(basis_plot, combination_plot; layout=(1,2), size=(1100,430))
# display(activity1_plot)


# Optional: save the comparison after uncommenting its two lines above.
# mkpath(lecture2_figure_dir)
# savefig(activity1_plot, joinpath(lecture2_figure_dir,"activity1_monomial_basis.png"))


# Part (c): Recover coefficients from a Vandermonde system.
xx = [-1.0, 0.0, 1.0];
V = [x^j for x in xx, j in 0:2];
s=1.0;
yy = [s*x^2 for x in xx];
aa = V \ yy
BigFloat.(aa)
=#


# ACTIVITY 2: Lagrange basis, n=4, equally spaced nodes on [0,1]
# This section can run independently of Activity 1.
#=
n = 4
nodes = collect(range(0.0, 1.0, length=n+1))
grid = range(0.0, 1.0, length=401)

# Mathematical index j runs from 0 to n.
# Julia entry nodes[j+1] stores the mathematical node x_j.


# Each column contains one basis function evaluated on the plotting grid.
L_values = [L(nodes,j,z) for z in grid, j in 0:n]
lagrange_plot = plot(grid, L_values;
    label=permutedims(["L_4,$j" for j in 0:n]),
    linewidth=2, xlabel="x", ylabel="Basis value",
    title="Lagrange basis: five equally spaced nodes",
    xlims=(-0.02,1.02), xticks=nodes,
    legend=:outerright, margin=6 * Plots.mm,
    size=(950,450))
vline!(lagrange_plot, nodes;
    label=false, color=:gray, linestyle=:dot, linewidth=1)
# Circular markers show L_(4,j)(x_i): one at its own node, zero at the others.
scatter!(lagrange_plot, nodes, [L(nodes,j,z) for z in nodes, j in 0:n];
    label=false, color=:black, markershape=:circle, markersize=4)
display(lagrange_plot)

# Check 1: Rows correspond to nodes x_i, columns to basis functions L_(4,j).
# The result should be the 5-by-5 identity matrix.
L_at_nodes = [L(nodes,j,z) for z in nodes, j in 0:n]
display(L_at_nodes)
@assert L_at_nodes == Matrix{Float64}(I,n+1,n+1)

# Check 2: Sum the five basis functions at every sampled point.
basis_sum = vec(sum(L_values, dims=2))
sum_error = maximum(abs.(basis_sum .- 1.0))
println("Maximum sampled deviation of sum(L_4,j) from 1: ", sum_error)
@assert sum_error < 1e-12

sum_plot = plot(grid, basis_sum;
    label="Sum of L_4,j", linewidth=3,
    xlabel="x", ylabel="Sum",
    title="Sum of the Lagrange basis functions",
    xlims=(0,1), ylims=(0.9,1.1),
    legend=:bottomright, margin=6 * Plots.mm)
hline!(sum_plot, [1.0];
    label="1", color=:black, linestyle=:dash, linewidth=2)
display(sum_plot)
=#


# ACTIVITY 3: The central Lagrange basis function for n=20
# Load this file's helper include before running the activity.
#=
nodes20 = collect(range(0.0, 1.0, length=21))
grid20 = range(0.0, 1.0, length=8001)
# Mathematical j=10 corresponds to Julia entry 11.
L20_10(z) = L(nodes20,10,z)
values20 = L20_10.(grid20)
node_values20 = L20_10.(nodes20)
@assert node_values20 == [i == 10 ? 1.0 : 0.0 for i in 0:20]

lagrange20_plot = plot(grid20, values20;
    label="L_20,10(x)", linewidth=2,
    xlabel="x", ylabel="Basis value",
    title="Central Lagrange basis: n=20, j=10",
    xlims=(-0.01,1.01), xticks=0:0.1:1,
    legend=:topright, margin=6 * Plots.mm, size=(950,450))
scatter!(lagrange20_plot, nodes20, node_values20;
    label="Values at nodes", color=:black,
    markershape=:circle, markersize=4)
display(lagrange20_plot)

# L_(20,10)(1/2)=1 and it is zero at all other nodes.
# Compare the vertical scale with n=4: values between nodes can be much larger.
# Optional: export the current plot inside this repository.
# mkpath(lecture2_figure_dir)
# savefig(lagrange20_plot, joinpath(lecture2_figure_dir,"Lagrange-basis-n20-j10.pdf"))
=#

# ACTIVITY 4: Cost of computing f[x_0,...,x_n]
# Load this file's helper include, then run this section independently.
# Inputs below are Float64 vectors of equal length with distinct nodes.

# dd_last_recursive, dd_last_table, and dd_cost_experiment are in helper.jl.
# The table stores one column; the recursive method repeats subproblems.
#=
cost_results = dd_cost_experiment([4,8,12,16,20])

# One update means two subtractions and one division.
# Recursive update count: C(0)=0, C(n)=2C(n-1)+1, so C(n)=2^n-1.
# Table update count: n+(n-1)+...+1 = n(n+1)/2.
# Both implementations use O(n) auxiliary space (stack or a stored column).
# These are costs of these algorithms, not a lower bound for the problem.
# Runtime is noisy at small n. The operation counts explain the growth.
# Both methods receive precomputed data; evaluating sin is excluded.

# Optional: larger n makes the quadratic timing trend easier to see.
# Do NOT use direct recursion for these orders.

# dd_table_scaling()
# When n doubles, the table update count approaches a factor of 4.
=#

# ACTIVITY 5: Interpolate sin(x) and check the error bound
# Load this file's helper include before running the activity.

# Assemble the interpolant from the same basis function used in Activity 2.
# Julia yy[j+1] stores the mathematical value f_j.
#=
xx = [0.0, pi/4, pi/2]
yy = sin.(xx)
z = pi/6
sine_grid = range(0.0, pi/2, length=501)
p2_values = [lagrange(xx,yy,t) for t in sine_grid]

sine_plot = plot(sine_grid, sin.(sine_grid);
    label="f(x) = sin(x)", color=:blue, linewidth=2.5,
    xlabel="x", ylabel="Function value",
    xticks=([0,pi/6,pi/4,pi/2], ["0","pi/6","pi/4","pi/2"]),
    xlims=(-0.03,pi/2+0.03), ylims=(-0.04,1.08),
    legend=:bottomright, margin=6 * Plots.mm, size=(950,450))
plot!(sine_plot, sine_grid, p2_values;
    label="Lagrange interpolant P_2(x)", color=:darkorange,
    linewidth=2.5, linestyle=:dash)
scatter!(sine_plot, xx, yy;
    label="Interpolation nodes", color=:black,
    markershape=:circle, markersize=5)
vline!(sine_plot, [z];
    label="x = pi/6", color=:gray, linestyle=:dot, linewidth=1.5)
display(sine_plot)

# Optional: export the lecture figure without needing the LaTeX source tree.
# mkpath(lecture2_figure_dir)
# savefig(sine_plot, joinpath(lecture2_figure_dir,"Sine-interpolation.pdf"))

# The interpolant matches sin(x) at the three nodes.
@assert all(isapprox(lagrange(xx,yy,t),sin(t); atol=1e-14) for t in xx)

# At z=pi/6, |f'''(t)|=|cos(t)| <= 1 gives this theoretical bound.
err = abs(sin(z) - lagrange(xx,yy,z))
bound = abs(prod(z .- xx))/factorial(3)
println((actual_error=err, error_bound=bound, below_bound=err <= bound))
@assert isapprox(bound, pi^3/1296)
@assert err <= bound
# Expected: actual_error ≈ 0.01742825, error_bound ≈ 0.02392460.
# The bound is not the exact error: the remainder contains f'''(xi),
# whose magnitude we bounded by 1.
=#

# ACTIVITY 6: Global Hermite interpolation with two, three and four nodes
# Load this file's helper include before running the activity.
# These are single global polynomials, not piecewise cubic interpolants.

# hermite, hermite_coefficients, and plot_hermite_sine are in helper.jl.
#=
# Explicit cubic for the endpoint value and slope conditions.
H3(z) = z + (12-4pi)/pi^2*z^2 + (4pi-16)/pi^3*z^3
dH3(z) = 1 + 2*(12-4pi)/pi^2*z + 3*(4pi-16)/pi^3*z^2
@assert isapprox([H3(0),H3(pi/2),dH3(0),dH3(pi/2)], [0,1,1,0]; atol=1e-14)
@assert all(isapprox(H3(z),hermite([0.0,pi/2],[0.0,1.0],[1.0,0.0],z); atol=1e-14)
            for z in range(0.0,pi/2,length=101))

hermite2_plot = plot_hermite_sine(2)
hermite3_plot = plot_hermite_sine(3)
hermite4_plot = plot_hermite_sine(4)
display(hermite2_plot)
display(hermite3_plot)
display(hermite4_plot)
# Optional exports:
# mkpath(lecture2_figure_dir)
# savefig(hermite2_plot, joinpath(lecture2_figure_dir,"Hermite-sine-2-nodes.pdf"))
# savefig(hermite3_plot, joinpath(lecture2_figure_dir,"Hermite-sine-3-nodes.pdf"))
# savefig(hermite4_plot, joinpath(lecture2_figure_dir,"Hermite-sine-4-nodes.pdf"))
# Two nodes impose four conditions, so the degree is at most 3.
# Three nodes impose six conditions, so the degree is at most 5.
# Four nodes impose eight conditions, so the degree is at most 7.
# Symmetry can make the actual degree lower. The subscript is a degree bound.
# A sampled maximum error is not a certified error bound on the whole interval.
=#
