# Lecture 8 (6 November 2026): adaptivity, Gaussian quadrature,
# and integral equations.
# Start in the repository with julia --project=.
# include("Lecture8/Lecture8_support.jl")
# Loading defines functions only. Run the CONTENTS of each #= ... =# block.
# Six numbered activities. Multiple-integral activities are in Lecture9_support.jl.
# Each activity runs independently after loading this support file.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture8_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture8"))

# ACTIVITY 1: Where Does Adaptivity Refine?
# Returns corrected values and a heuristic error estimate, not a certificate.
# n is the even subinterval count on each current interval, kept fixed on recursion.

#=
f(x) = 1 + sin(5x) + 1/(1+(1000*(x-0.3))^2) +
       0.5/(1+(100*(x-0.75))^2)
exact = 1 + (1-cos(5))/5 + (atan(700)+atan(300))/1000 +
        (atan(25)+atan(75))/200
n = 4
r = adaptive_simpson(f,0.,1.,n;atol=1e-8)
println("Adaptive Simpson on [0,1]: n=$n subintervals per current interval, atol=1e-8")
println("  value = corrected approximation; estimated = estimated absolute error; actual = abs(value-exact).")
println("  calls = total function evaluations; converged = all local stopping tests passed.")
println((value=r.value,estimated=r.estimated_error,actual=abs(r.value-exact),
         calls=r.evaluations,converged=r.converged))
# Compare n=2,4,8 at the same tolerance. More nodes per interval need not
# mean fewer total evaluations; the accepted intervals can also change.
println("\nAdaptive Simpson: compare n=2,4,8 at the same tolerance (1e-8).")
println("  n = subintervals per current interval, kept fixed during refinement.")
println("  calls = total function evaluations; leaves = accepted intervals; actual_error = abs(value-exact).")
for nlocal in (2,4,8)
    local result = adaptive_simpson(f,0.,1.,nlocal;atol=1e-8)
    println((n=nlocal,calls=result.evaluations,leaves=length(result.intervals),
             actual_error=abs(result.value-exact),converged=result.converged))
end
# Uniform refinement also uses the Simpson-difference estimate to stop.
# Cached values make its cumulative evaluation count equal to the final n+1.
u = uniform_simpson(f,0.,1.,4;atol=1e-8)
println("\nUniform Simpson: double the global grid until its estimated error is at most 1e-8.")
println("  n = equal subintervals on [0,1]; calls = cumulative evaluations, with all samples reused.")
println("  estimated = abs(S_n-S_(n/2))/15; actual = error of the Richardson-corrected value.")
for row in u.history
    println((n=row.n,calls=row.calls,estimated=row.estimated_error,
             actual=abs(row.value-exact)))
end
println("\nSame tolerance, cached samples, and Richardson correction in both methods:")
println((method="Adaptive",calls=r.evaluations,estimated=r.estimated_error,
         actual=abs(r.value-exact),converged=r.converged))
println((method="Uniform",calls=u.evaluations,estimated=u.estimated_error,
         actual=abs(u.value-exact),converged=u.converged))

grid = range(0.,1.,length=20001)
centers = [(a+b)/2 for (a,b) in r.intervals]
widths = [b-a for (a,b) in r.intervals]
curve = plot(grid,f.(grid);label="Full function",linewidth=2.5,color=:navy,
    xlabel="x",ylabel="f(x)",title="Two interior peaks",legend=:bottomleft,
    xlims=(0,1),ylims=(-0.08,3.25))
plot!(curve,grid,1 .+ sin.(5 .* grid);label="Background: 1 + sin(5x)",
    linewidth=1.7,linestyle=:dash,color=:gray)
vline!(curve,[0.3,0.75];label=false,color=:gray,linestyle=:dot,linewidth=1)
mesh = scatter(centers,widths;label=false,markersize=4,color=:teal,
    xlims=(0,1),yscale=:log10,xlabel="interval center",ylabel="interval width",
    title="Accepted adaptive intervals")
vline!(mesh,[0.3,0.75];label=false,color=:gray,linestyle=:dot,linewidth=1)
fig = plot(curve,mesh;layout=(1,2),size=(1400,550),guidefontsize=18,
    tickfontsize=14,titlefontsize=19,legendfontsize=14,
    left_margin=11Plots.mm,right_margin=5Plots.mm,
    top_margin=6Plots.mm,bottom_margin=10Plots.mm)
display(fig)

# Failure experiment: the compact peak has support [0.36,0.38].
# Use n=2: all five initial samples (0,1/4,1/2,3/4,1) are zero.
# Its exact area is 16w/15. Omitting n also defaults to n=2.
w, center = 0.01, 0.37
bump(x) = abs((x-center)/w) < 1 ? (1-((x-center)/w)^2)^2 : 0.0
missed = adaptive_simpson(bump,0.,1.,2;atol=1e-8)
println("\nMissed-peak example: adaptive Simpson with n=2; all five initial samples miss the peak.")
println("  reported_error = estimated absolute error; actual_error = the true absolute error.")
println("  reported_converged means the local stopping test passed, even though the peak was missed.")
println((missed_peak_value=missed.value,reported_error=missed.estimated_error,
         actual_error=16w/15,reported_converged=missed.converged))
# The flag means the local test passed. Known support/breakpoints or additional
# sampling information are needed to resolve a feature that no node detected.
# Plot the missed peak and the five initial sample points.
samples = [0.0,0.25,0.5,0.75,1.0]
full_grid = range(0.,1.,length=4001)
zoom_grid = range(0.34,0.40,length=1201)
full_peak = plot(full_grid,bump.(full_grid);label="b(x)",color=:navy,linewidth=3,
    fillrange=0,fillalpha=0.18,title="Full interval [0, 1]",xlabel="x",ylabel="b(x)",
    xlims=(-0.025,1.025),ylims=(-0.08,1.14),legend=:topright,
    xticks=(samples,["0","1/4","1/2","3/4","1"]))
scatter!(full_peak,samples,bump.(samples);label="Initial samples",color=:red,
    markersize=7,markerstrokecolor=:white,markerstrokewidth=1)
zoom_peak = plot(zoom_grid,bump.(zoom_grid);label=false,color=:navy,linewidth=3,
    fillrange=0,fillalpha=0.18,title="Zoom near the peak",xlabel="x",ylabel="b(x)",
    xlims=(0.34,0.40),ylims=(-0.08,1.14),xticks=[0.34,0.36,0.37,0.38,0.40])
vline!(zoom_peak,[center-w,center+w];label=false,color=:gray,linestyle=:dash)
scatter!(zoom_peak,[center],[1.0];label=false,color=:navy,markersize=5)
for panel in (full_peak,zoom_peak)
    plot!(panel;titlefontsize=21,guidefontsize=20,tickfontsize=17,legendfontsize=16,
        left_margin=9Plots.mm,right_margin=5Plots.mm,
        bottom_margin=10Plots.mm,top_margin=6Plots.mm)
end
missed_peak_fig = plot(full_peak,zoom_peak;layout=(1,2),size=(1500,540))
display(missed_peak_fig)
=#


# ACTIVITY 2: Testing Gaussian Precision.
# Nodes x_0,...,x_n: use n+1 function values, with precision 2n+1.

#=
n = 2
rule = gauss_rule(:legendre,n+1)
println(rule)
for k in 0:2n+2
    local exact = isodd(k) ? 0.0 : 2/(k+1)
    println((degree=k,residual=sum(rule.weights.*rule.nodes.^k)-exact))
end
=#


# ACTIVITY 3: Gauss and Composite Simpson.
# n+1 Gaussian nodes versus n Simpson subintervals: both use n+1 values.
# Take n even for the composite Simpson rule.

#=
exact = exp(4)-1
ns = [2,4,6,8]
eg, es = Float64[], Float64[]
for n in ns
    local g = gauss_legendre(exp,0.,4.,n+1)
    local s = simpson(exp,0.,4.,n)
    push!(eg,abs(g-exact))
    push!(es,abs(s-exact))
    println((calls=n+1, Gauss_error=abs(g-exact), Simpson_error=abs(s-exact)))
end
fig = plot(ns .+ 1,max.(eg,eps(Float64));label="Gauss-Legendre",yscale=:log10,
    markershape=:circle,xlabel="function evaluations per rule",ylabel="absolute error",
    size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,ns .+ 1,es;label="Composite Simpson",markershape=:square)
display(fig)
# Polynomial exactness alone does not guarantee accuracy for every function.
=#

# ACTIVITY 4: Weighted Gaussian Quadrature.
# The rho weight is already built into the weights. Do not multiply it twice.

#=
cheb = gauss_rule(:chebyshev,2+1)
lag = gauss_rule(:laguerre,1+1)
herm = gauss_rule(:hermite,2+1)
println(sum(cheb.weights.*cheb.nodes.^4)) # 3pi/8
println(sum(lag.weights.*lag.nodes.^3)) # 3! = 6
println(sum(herm.weights.*herm.nodes.^4)) # 3sqrt(pi)/4
normal_fourth = sum(herm.weights.*(sqrt(2).*herm.nodes).^4)/sqrt(pi)
println((normal_fourth=normal_fourth,exact=3.0))
fig = plot(;xlabel="node x",ylabel="weight",size=(850,420),
    left_margin=8Plots.mm,bottom_margin=8Plots.mm)
for (name,rule) in (("Chebyshev",cheb),("Laguerre",lag),("Hermite",herm))
    scatter!(fig,rule.nodes,rule.weights;label=name,markersize=6)
end
display(fig)
=#

# ACTIVITY 5: A Gaussian Average.
# Source: Hensman, Matthews and Ghahramani, AISTATS 2015,
# https://proceedings.mlr.press/v38/hensman15.html (Gaussian expected log-likelihoods).
# This classroom example averages a sigmoid output to isolate the integration step.
# No model training or machine-learning package is required.

#=
g(s) = 1/(1+exp(-s))
ns = [0,2,4,8,14]
averages = Float64[]
for n in ns
    local rule = gauss_rule(:hermite,n+1)
    local values = g.(1 .+ sqrt(2).*rule.nodes)
    local average = sum(rule.weights.*values)/sqrt(pi)
    push!(averages,average)
    println((nodes=n+1, average=average))
end
println((output_at_mean=g(1.0),))
# Independent numerical reference: integrate against the normal density.
# Since 0 < g < 1, truncating to [-10,10] omits less than 2e-23.
reference = adaptive_simpson(z -> g(1+z)*exp(-z^2/2)/sqrt(2pi),
                             -10.,10.;atol=1e-12)
println((reference=reference.value,reference_converged=reference.converged,
         last_error=abs(averages[end]-reference.value)))
fig = plot(ns .+ 1,averages;label="Gauss-Hermite average",markershape=:circle,
    xlabel="Gaussian node count n+1",ylabel="average model output",size=(850,420),
    left_margin=8Plots.mm,bottom_margin=8Plots.mm)
hline!(fig,[reference.value];label="Numerical reference",linestyle=:dash)
hline!(fig,[g(1.0)];label="Output at mean score",linestyle=:dot)
display(fig)
# These are integration examples, not a reproduction of the full research model.
=#


# ACTIVITY 6: Solving an Integral Equation.
# k(t,s)=t*s/2; integral_0^1 s*exp(s)ds=1, so f(t)=exp(t)-t/2.

#=
kernel(t,s) = t*s/2
f(t) = exp(t)-t/2
grid = range(0.,1.,length=401)
ns = [1,2,3,5]
errors = Float64[]
signed_error_plot = plot(;xlabel="t",ylabel="Y(t) - exp(t)",xlims=(0.,1.),
    title="Signed error: n=1 (2 nodes)",legend=false)
hline!(signed_error_plot,[0.0];color=:gray,linestyle=:dash,linewidth=1)
for n in ns
    local rule = mapped_gauss_rule(0.,1.,n+1)
    local r = nystrom_solve(kernel,f,rule)
    local signed_error = r.evaluate.(grid).-exp.(grid)
    local error = maximum(abs.(signed_error))
    push!(errors,error)
    println((nodes=n+1,residual=r.residual,error=error))
    n == 1 && plot!(signed_error_plot,grid,signed_error;color=:royalblue,linewidth=3)
end
errorplot = plot(ns .+ 1,max.(errors,eps(Float64));label=false,yscale=:log10,
    markershape=:circle,xlabel="Gaussian nodes n+1",ylabel="max error on test grid",
    title="Quadrature controls the error")
fig = plot(signed_error_plot,errorplot;layout=(1,2),size=(1400,520),
    guidefontsize=15,tickfontsize=12,titlefontsize=16,legendfontsize=12,
    left_margin=9Plots.mm,bottom_margin=10Plots.mm,right_margin=5Plots.mm)
display(fig)
# A tiny algebraic residual alone says nothing about discretization accuracy.
=#
