# Lecture 10 (20 November 2026): spectral radius, nonlinear equations,
# bisection, fixed-point iteration, Newton and equilibrium networks.
# Start in the repository with julia --project=.
# include("Lecture10/Lecture10_support.jl")
# Loading imports the shared helper only; run one #= ... =# activity block at a time.
# All eight numbered activities run independently after loading this file.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture10_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture10"))
# Before optional figure exports, run mkpath(lecture10_figure_dir).

# ACTIVITY 1: Is Positive Definiteness Enough?
#=
A = [1. .75 .75; .75 1. .75; .75 .75 1.]
xstar = ones(3)
b = A*xstar
Tj = stationary_matrix(A;method=:jacobi)
Tg = stationary_matrix(A;method=:gauss_seidel)
rhoj = maximum(abs.(eigvals(Tj)))
rhog = maximum(abs.(eigvals(Tg)))
println((eigenvalues_A=eigvals(Symmetric(A)),rho_Jacobi=rhoj,rho_Gauss_Seidel=rhog))
jac = stationary_solve(A,b;method=:jacobi,maxiter=30,rtol=1e-12)
gs = stationary_solve(A,b;method=:gauss_seidel,maxiter=150,rtol=1e-12)
for (name,r) in (("Jacobi",jac),("Gauss-Seidel",gs))
    println((method=name,iterations=r.iterations,relative_residual=r.residuals[end]/norm(b,Inf),
             actual_error=norm(r.x-xstar,Inf),converged=r.converged))
end
resplot = plot(;xlabel="iteration k",ylabel="relative residual",yscale=:log10,
    title="A is symmetric positive definite",legend=:topright)
for (name,r) in (("Jacobi",jac),("Gauss-Seidel",gs))
    plot!(resplot,0:r.iterations,max.(r.residuals./norm(b,Inf),1e-16);label=name,linewidth=3)
end
rhoplot = bar(["Jacobi","Gauss-Seidel"],[rhoj,rhog];label=false,color=[:orange,:teal],
    ylabel="spectral radius",title="Convergence requires rho(T) < 1",ylims=(0,1.8))
hline!(rhoplot,[1.0];label=false,color=:black,linestyle=:dash,linewidth=2)
fig = plot(resplot,rhoplot;layout=(1,2),size=(1450,550),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=13,left_margin=12Plots.mm,
    bottom_margin=12Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
display(fig)
# Optional export: savefig(fig,joinpath(lecture10_figure_dir,"activity1.pdf"))
=#

# ACTIVITY 2: Recognizing Convergence Orders.
# These are model error sequences, not outputs of a particular solver.
# BigFloat keeps the rapid quadratic sequence away from Float64 underflow.
#=
ks = collect(0:8)
orders = [big"1.0",big"1.2",big"1.618",big"2.0"]
constants = [big"0.8",big"1.0",big"1.0",big"1.0"]
labels = ["Linear: 0.8 E", "Order 1.2: E^1.2", "Order 1.618: E^1.618", "Quadratic: E^2"]
error_sequences = Vector{BigFloat}[]
for (p,C) in zip(orders,constants)
    local errors = [big"0.5"]
    for k in 1:8
        push!(errors,C*errors[end]^p)
    end
    push!(error_sequences,errors)
end
println("Model errors E_k; columns are linear, order 1.2, order 1.618, and quadratic.")
@printf("%2s %12s %12s %12s %12s\n","k","Linear","Order 1.2","Order 1.618","Quadratic")
for k in eachindex(ks)
    @printf("%2d %12.5e %12.5e %12.5e %12.5e\n",ks[k],
        Float64(error_sequences[1][k]),Float64(error_sequences[2][k]),
        Float64(error_sequences[3][k]),Float64(error_sequences[4][k]))
end
fig = plot(;xlabel="iteration k",ylabel="log10(error E_k)",
    title="Convergence orders: model error sequences",legend=:bottomleft,
    size=(1100,520),guidefontsize=17,tickfontsize=13,titlefontsize=18,legendfontsize=13,
    left_margin=12Plots.mm,bottom_margin=11Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
for (label,E) in zip(labels,error_sequences)
    plot!(fig,ks,Float64.(log10.(E));label=label,markershape=:circle,linewidth=2)
end
display(fig)
# Optional export: savefig(fig,joinpath(lecture10_figure_dir,"activity2.pdf"))
=#

# ACTIVITY 3: Bisection and Its Guaranteed Error Bound.
# The sign-changing interval is [2,3], as in the lecture animation.
#=
f(x) = x^3-2x-5
r = bisect_root(f,2.0,3.0;atol=1e-8)
println("Bisection stops when the bracket half-width is at most atol (or finds an exact root).")
println((root=r.x,bracket_reductions=r.iterations,midpoint_evaluations=length(r.iterates),
         calls=r.calls,bound=r.bound,
         interval=r.interval,residual=abs(f(r.x)),status=r.status))
ks = collect(0:r.iterations)
fig = plot(ks,2 .* r.bounds;label="Recorded bracket width",markershape=:circle,
    yscale=:log10,xlabel="bracket reductions k",ylabel="interval width",
    title="Bisection halves the interval",size=(1100,500),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=13,left_margin=12Plots.mm,
    bottom_margin=11Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
plot!(fig,ks,2.0 .^ (-ks);label="Initial width / 2^k",linestyle=:dash,linewidth=2)
display(fig)
# Optional export: savefig(fig,joinpath(lecture10_figure_dir,"activity3.pdf"))
=#

# ACTIVITY 4: Newton and Bisection.
# Compare both solvers on the same cubic as the bisection animation.
#=
f(x) = x^3-2x-5
df(x) = 3x^2-2
nr = newton_solve(f,df,2.5;atol=1e-12)
br = bisect_root(f,2.0,3.0;atol=1e-12)
reference = setprecision(256) do
    z = big"2.5"
    for k in 1:12
        z -= (z^3-2z-5)/(3z^2-2)
    end
    z
end
println("Newton converged means its residual test passed; bisection reports an interval error bound.")
println((Newton=nr.x,iterations=nr.iterations,residual=nr.residuals[end],
         Bisection=br.x,reductions=br.iterations,bound=br.bound))
newton_errors = [Float64(abs(BigFloat(z)-reference)) for z in nr.iterates]
bisection_errors = [Float64(abs(BigFloat(z)-reference)) for z in br.iterates]
x0 = 2.5
x1 = x0-f(x0)/df(x0)
grid = range(1.9,2.7,length=401)
tangent(z) = f(x0)+df(x0)*(z-x0)
tangentplot = plot(grid,f.(grid);label="f(x)",linewidth=2,
    xlabel="x",ylabel="function / tangent value",title="The first Newton step",legend=:topleft)
plot!(tangentplot,grid,tangent.(grid);label="Tangent at x0=2.5",linestyle=:dash,linewidth=2)
hline!(tangentplot,[0.];label=false,color=:gray)
scatter!(tangentplot,[x0,x1],[f(x0),0.];label="Tangent point / x-intercept",markersize=6)
errorplot = plot(0:br.iterations,max.(bisection_errors,1e-17);label="Bisection",markershape=:circle,
    xlabel="iteration / bracket reduction k",ylabel="actual absolute error",yscale=:log10,
    title="The same cubic: two solvers")
plot!(errorplot,0:nr.iterations,max.(newton_errors,1e-17);label="Newton",markershape=:diamond,linewidth=2)
fig = plot(tangentplot,errorplot;layout=(1,2),size=(1450,530),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=12,left_margin=12Plots.mm,
    bottom_margin=12Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
display(fig)
# Optional export: savefig(fig,joinpath(lecture10_figure_dir,"activity4.pdf"))
=#

# ACTIVITY 5: Try Different Starting Points.
# For f(x)=x^3-2x+2, predict the behavior before running.
# Then try an additional starting value.
#=
f(x) = x^3-2x+2
df(x) = 3x^2-2
xstar = -cbrt(1+sqrt(19/27)) - cbrt(1-sqrt(19/27))
println("Exact real root (evaluated numerically), and its residual:")
println((xstar=xstar,residual=abs(f(xstar))))
println("Each row: starting value, final iterate, iteration count, and residual-test flag.")
for x0 in [0.0,-1.0,-2.0,2.0]
    local r = newton_solve(f,df,x0;atol=1e-12,maxiter=30)
    println((x0=x0,x=r.x,steps=r.iterations,converged=r.converged))
end
cycle = newton_solve(f,df,0.0;atol=1e-12,maxiter=8)
println("Iterates starting from x0=0 (including the initial value):")
println(cycle.iterates)
# Which starts reach xstar? Why does the local theorem allow the cycle?
# Try one more x0 and compare. Inspect r.iterates for an individual run.
=#

# ACTIVITY 6: The Three-Variable Fixed-Point Example.
# This square-root branch corresponds to x[2] >= -0.1.
# The original F=0 has another solution on the other square-root branch.
# The supplied contraction q is a proven bound on [-1,1]^3; the helper cannot
# verify such a global hypothesis from sampled values.
#=
T(x) = [cos(x[2]*x[3])/3+1/6,
        sqrt(x[1]^2+sin(x[3])+1.06)/9-0.1,
        -exp(-x[1]*x[2])/20-(10pi-3)/60]
xstar = [0.5,0.0,-pi/6]
q = 2sin(1)/3
r = fixedpoint_solve(T,zeros(3);contraction=q,atol=1e-10)
println("bounds are certified error bounds conditional on the proven contraction and invariant-set hypotheses.")
println((root=r.x,iterations=r.iterations,bound=r.bounds[end],
         actual_error=norm(r.x-xstar,Inf),fixed_point_residual=r.residuals[end],status=r.status))
errors = [norm(r.iterates[:,k]-xstar,Inf) for k in axes(r.iterates,2)]
ks = collect(0:r.iterations)
fig = plot(ks,max.(errors,1e-16);label="Actual error",markershape=:circle,
    xlabel="iteration k",ylabel="infinity norm",yscale=:log10,
    title="Fixed-point iteration and its error bound",size=(1100,510),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=13,left_margin=12Plots.mm,
    bottom_margin=11Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
plot!(fig,ks,max.(r.bounds,1e-16);label="Contraction error bound",linestyle=:dash,linewidth=2)
display(fig)
# Optional export: savefig(fig,joinpath(lecture10_figure_dir,"activity6.pdf"))

# EXERCISE: Find the other solution (after the positive-branch plot).
# Copy the map T above to Tminus and change ONLY its second component
# to select the branch x[2] + 0.1 < 0. Keep the first and third components.
# Before reusing contraction=q, justify self-mapping on [-1,1]^3 and the
# same derivative bound for the modified map.
# Then complete and run:
# rminus = fixedpoint_solve(Tminus,zeros(3);contraction=q,atol=1e-10)
# (rminus.x, rminus.iterations, rminus.bounds[end])
# Check x[2] < -0.1 and substitute the computed vector into all three
# original equations. Report the solution and iteration count.
=#

# ACTIVITY 7: Newton for the Three-Variable System.
# The Jacobian solve uses DF(x) \ F(x), never inv(DF(x)).
#=
F(x) = [3x[1]-cos(x[2]*x[3])-0.5,
        x[1]^2-81(x[2]+0.1)^2+sin(x[3])+1.06,
        exp(-x[1]*x[2])+20x[3]+(10pi-3)/3]
DF(x) = [3.0 x[3]*sin(x[2]*x[3]) x[2]*sin(x[2]*x[3]);
         2x[1] -162(x[2]+0.1) cos(x[3]);
         -x[2]*exp(-x[1]*x[2]) -x[1]*exp(-x[1]*x[2]) 20.0]
T(x) = [cos(x[2]*x[3])/3+1/6,
        sqrt(x[1]^2+sin(x[3])+1.06)/9-0.1,
        -exp(-x[1]*x[2])/20-(10pi-3)/60]
x0 = [0.1,0.1,-0.1]
xstar = [0.5,0.0,-pi/6]
nr = newton_system(F,DF,x0;atol=1e-12)
fp = fixedpoint_solve(T,x0;contraction=2sin(1)/3,atol=1e-12)
println((Newton=nr.x,iterations=nr.iterations,residual=nr.residuals[end],status=nr.status))
println((Fixed_point=fp.x,iterations=fp.iterations,bound=fp.bounds[end],status=fp.status))
fig = plot(;xlabel="iteration k",ylabel="actual error in infinity norm",yscale=:log10,
    title="The original three-variable example",size=(1100,510),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=13,left_margin=12Plots.mm,
    bottom_margin=11Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
for (name,r) in (("Fixed point",fp),("Newton",nr))
    local errors = [norm(r.iterates[:,k]-xstar,Inf) for k in axes(r.iterates,2)]
    plot!(fig,0:r.iterations,max.(errors,1e-16);label=name,markershape=:circle,linewidth=2)
end
display(fig)
# Optional export: savefig(fig,joinpath(lecture10_figure_dir,"activity7.pdf"))
=#

# ACTIVITY 8: A Two-State Equilibrium Network.
# A numerical illustration of a DEQ, with fixed coefficients and no training.
# Bai, Kolter and Koltun, Deep Equilibrium Models, NeurIPS 2019:
# https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html
#=
W = [0.35 0.10;-0.05 0.30]
b = [1.0,-0.5]
u = 0.5
T(z) = tanh.(W*z+u*b)
F(z) = z-T(z)
DF(z) = I-Diagonal(1 .- T(z).^2)*W
q = opnorm(W,Inf) # 0.45: maximum absolute row sum
# Coordinatewise tanh is 1-Lipschitz, so q is a proven global contraction bound.
# contraction=q is optional. For k >= 1, it stops on the certified error bound
# q/(1-q)*norm(z_new-z_old,Inf) <= atol; the iteration update is unchanged.
# Omitting contraction instead stops on norm(T(z)-z,Inf) <= atol.
# For example: fixedpoint_solve(T,zeros(2);atol=1e-12)
fp = fixedpoint_solve(T,zeros(2);contraction=q,atol=1e-12)
nr = newton_system(F,DF,zeros(2);atol=1e-12)
println("Toy equilibrium state: z = tanh(W*z + u*b). No network is trained in this activity.")
println((q=q,fixed_point=fp.x,Newton=nr.x,agreement=norm(fp.x-nr.x,Inf)))
println((fixed_point_iterations=fp.iterations,Newton_iterations=nr.iterations,
         fixed_point_bound=fp.bounds[end],Newton_residual=nr.residuals[end]))
stateplot = plot(0:fp.iterations,transpose(fp.iterates);label=["state z_1" "state z_2"],
    xlabel="fixed-point iteration k",ylabel="state value",title="Equilibrium state",linewidth=2)
resplot = plot(;xlabel="iteration k",ylabel="nonlinear residual",yscale=:log10,
    title="Two solvers for the same equilibrium")
for (name,r) in (("Fixed point",fp),("Newton",nr))
    plot!(resplot,0:r.iterations,max.(r.residuals,1e-16);label=name,markershape=:circle,linewidth=2)
end
fig = plot(stateplot,resplot;layout=(1,2),size=(1400,520),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=13,left_margin=11Plots.mm,
    bottom_margin=11Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
display(fig)
# Optional export: savefig(fig,joinpath(lecture10_figure_dir,"activity8.pdf"))
=#
