# Lecture 7: composite rules, Euler-Maclaurin, Romberg, adaptivity, Gaussian rules.
# Start Julia from the repository: julia --project=.
# Then include("Lecture7/Lecture7_support.jl"). Loading does not run activities.
# Select and run the CONTENTS of one #= ... =# block at a time, in slide order.
# Every activity also runs independently after loading this file.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture7_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture7"))

# ACTIVITY 1: Composite quadrature error versus distinct evaluations.
# n counts subintervals; both methods use n+1 distinct nodes.
#=
ns = [4,8,16,32,64]
exact = exp(4)-1
et = [abs(trapezoid(exp,0.,4.,n)-exact) for n in ns]
es = [abs(simpson(exp,0.,4.,n)-exact) for n in ns]
println((trapezoid_ratios=et[1:end-1]./et[2:end],
         Simpson_ratios=es[1:end-1]./es[2:end]))
fig = plot(ns,et;label="Trapezoidal",yscale=:log10,
    markershape=:circle,xlabel="number of subintervals n",ylabel="absolute error",
    size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,ns,es;label="Simpson",markershape=:square)
display(fig)
=#


# ACTIVITY 2: Euler-Maclaurin endpoint corrections for exp on [0,1].
# Predict ratios 4,16,64. At small errors Float64 roundoff obscures the ratios.

#=
ns = [2,4,8,16,32]
exact = exp(1)-1
h = 1.0 ./ ns
T = [trapezoid(exp,0.,1.,n) for n in ns]
C4 = T .- h.^2 .* exact/12
C6 = C4 .+ h.^4 .* exact/720
for (name,Q) in (("T",T),("C4",C4),("C6",C6))
    local e = abs.(Q.-exact)
    println((rule=name,ratios=e[1:end-1]./e[2:end]))
end
fig = plot(ns,abs.(T.-exact);label="Trapezoidal O(h^2)",
yscale=:log10,markershape=:circle,xlabel="subinterval count n",
    ylabel="absolute error",size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,ns,max.(abs.(C4.-exact),eps(Float64));label="Corrected O(h^4)",markershape=:square)
plot!(fig,ns,max.(abs.(C6.-exact),eps(Float64));label="Corrected O(h^6)",markershape=:diamond)
display(fig)
=#


# ACTIVITY 3: Romberg table and the smoothness assumption.
# levels=6 requires 2^5+1=33 calls, not the sum of six separate grids.

#=
result = romberg_table(exp,0.,1.;levels=6)
println("exp(x): absolute errors")
print_romberg_errors(result.table,exp(1)-1)
println((evaluations=result.evaluations,
         first_extrapolation=result.table[2,2],Simpson=simpson(exp,0.,1.,2)))
rough = romberg_table(sqrt,0.,1.;levels=6)
println("sqrt(x): absolute errors")
print_romberg_errors(rough.table,2/3)
println("sqrt is not C^1 at the left endpoint")
fig = plot(1:6,max.([abs(result.table[k,k]-(exp(1)-1)) for k in 1:6],eps(Float64));
    label="exp(x)",yscale=:log10,markershape=:circle,xlabel="Romberg row k",
    ylabel="diagonal error",size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,1:6,[abs(rough.table[k,k]-2/3) for k in 1:6];label="sqrt(x)",markershape=:square)
display(fig)
=#


# ACTIVITY 4: Periodic analytic integrands can converge geometrically.
# Source: Trefethen and Weideman, SIAM Review 2014, doi:10.1137/130932132.
# The closed-form reference is 2pi/3 (substitute tan(x/2), splitting at pi).
#=
f(x) = 1/(5-4cos(x))
ns = [4,8,12,16,20,24,32,40]
errors = [abs(trapezoid(f,0.,2pi,n)-2pi/3) for n in ns]
nonperiodic = [abs(trapezoid(exp,0.,1.,n)-(exp(1)-1)) for n in ns]
println(collect(zip(ns,errors)))
fig = plot(ns,max.(errors,eps(Float64));label="Periodic analytic",yscale=:log10,
    markershape=:circle,xlabel="subinterval count n",ylabel="absolute error",
    size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,ns,nonperiodic;label="exp(x) on [0,1]",markershape=:square)
display(fig)
# Optional: although f(0) != f(pi), every odd derivative vanishes at both.
# Euler-Maclaurin therefore predicts rapid convergence on this half period too.
println((half_period_errors=[abs(trapezoid(f,0.,pi,n)-pi/3) for n in ns],))
=#

# ACTIVITY 5: Adaptive Simpson concentrates nodes around a peak.
# Returns corrected values and a heuristic error estimate, not a certificate.
#=
f(x) = 1/(1+400(x-0.3)^2)
r = adaptive_simpson(f,0.,1.;atol=1e-8)
exact = (atan(14)+atan(6))/20
println((value=r.value,estimated=r.estimated_error,actual=abs(r.value-exact),
         calls=r.evaluations,converged=r.converged))
ns = [16,32,64,128,256,512]
println([(n=n,calls=n+1,error=abs(simpson(f,0.,1.,n)-exact)) for n in ns])
centers = [(a+b)/2 for (a,b) in r.intervals]
widths = [b-a for (a,b) in r.intervals]
curve = plot(range(0.,1.,length=1001),f;label="Integrand",xlabel="x",ylabel="f(x)")
mesh = scatter(centers,widths;label="Accepted interval widths",yscale=:log10,
    xlabel="interval center",ylabel="width")
fig = plot(curve,mesh;layout=(2,1),size=(850,650),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
display(fig)
# Failure experiment: the compact peak has support [0.36,0.38].
# All initial samples (0,1/4,1/2,3/4,1) are zero. Its exact area is 16w/15.
w, center = 0.01, 0.37
bump(x) = abs((x-center)/w) < 1 ? (1-((x-center)/w)^2)^2 : 0.0
missed = adaptive_simpson(bump,0.,1.;atol=1e-8)
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


# ACTIVITY 6: Three Gaussian nodes integrate degrees 0 through 5 exactly.
# q is the number of nodes; the slides also use q=n+1.

#=
rule = gauss_rule(:legendre,3)
println(rule)
for k in 0:6
    local exact = isodd(k) ? 0.0 : 2/(k+1)
    println((degree=k,residual=sum(rule.weights.*rule.nodes.^k)-exact))
end
=#

# ACTIVITY 7: Gauss and composite Simpson with equal evaluation counts.
# q Gaussian nodes versus q-1 Simpson subintervals: both use q function values.
# q is odd so that Simpson has an even number of subintervals.
#=
exact = exp(4)-1
qs = [3,5,7,9]
eg, es = Float64[], Float64[]
for q in qs
    local g = gauss_legendre(exp,0.,4.,q)
    local s = simpson(exp,0.,4.,q-1)
    push!(eg,abs(g-exact))
    push!(es,abs(s-exact))
    println((calls=q, Gauss_error=abs(g-exact), Simpson_error=abs(s-exact)))
end
fig = plot(qs,max.(eg,eps(Float64));label="Gauss-Legendre",yscale=:log10,
    markershape=:circle,xlabel="function evaluations per rule",ylabel="absolute error",
    size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,qs,es;label="Composite Simpson",markershape=:square)
display(fig)
# Polynomial exactness alone does not guarantee accuracy for every function.
=#

# ACTIVITY 8: Weighted rules, and a standard normal expectation.
# The rho weight is already built into the weights. Do not multiply it twice.
#=
qc = gauss_rule(:chebyshev,3)
ql = gauss_rule(:laguerre,2)
qh = gauss_rule(:hermite,3)
println((Chebyshev=sum(qc.weights.*qc.nodes.^4),exact=3pi/8))
println((Laguerre=sum(ql.weights.*ql.nodes.^3),exact=6.0))
println((Hermite=sum(qh.weights.*qh.nodes.^4),exact=3sqrt(pi)/4))
normal_fourth = sum(qh.weights.*(sqrt(2).*qh.nodes).^4)/sqrt(pi)
println((normal_fourth=normal_fourth,exact=3.0))
fig = plot(;xlabel="node x",ylabel="weight",size=(850,420),
    left_margin=8Plots.mm,bottom_margin=8Plots.mm)
for (name,rule) in (("Chebyshev",qc),("Laguerre",ql),("Hermite",qh))
    scatter!(fig,rule.nodes,rule.weights;label=name,markersize=6)
end
display(fig)
=#

# ACTIVITY 9: Optional Gauss-Hermite quadrature for a Gaussian average.
# Source: Hensman, Matthews and Ghahramani, AISTATS 2015,
# https://proceedings.mlr.press/v38/hensman15.html (Gaussian expected log-likelihoods).
# This classroom example averages a sigmoid output to isolate the integration step.
# No model training or machine-learning package is required.
#=
g(s) = 1/(1+exp(-s))
qs = [1,3,5,9,15]
averages = Float64[]
for q in qs
    local rule = gauss_rule(:hermite,q)
    local values = g.(1 .+ sqrt(2).*rule.nodes)
    local average = sum(rule.weights.*values)/sqrt(pi)
    push!(averages,average)
    println((nodes=q, average=average))
end
println((output_at_mean=g(1.0),))
# Independent numerical reference: integrate against the normal density.
# Since 0 < g < 1, truncating to [-10,10] omits less than 2e-23.
reference = adaptive_simpson(z -> g(1+z)*exp(-z^2/2)/sqrt(2pi),
                             -10.,10.;atol=1e-12)
println((reference=reference.value,reference_converged=reference.converged,
         last_error=abs(averages[end]-reference.value)))
# Exact check: symmetry g(z)+g(-z)=1 makes E[g(Z)]=1/2.
rule0 = gauss_rule(:hermite,9)
symmetric_average = sum(rule0.weights.*g.(sqrt(2).*rule0.nodes))/sqrt(pi)
println((zero_mean_average=symmetric_average,exact=0.5))
fig = plot(qs,averages;label="Gauss-Hermite average",markershape=:circle,
    xlabel="Gaussian node count q",ylabel="average model output",size=(850,420),
    left_margin=8Plots.mm,bottom_margin=8Plots.mm)
hline!(fig,[reference.value];label="Numerical reference",linestyle=:dash)
hline!(fig,[g(1.0)];label="Output at mean score",linestyle=:dot)
display(fig)
# These are integration examples, not a reproduction of the full research model.
=#

# Optional export after an activity:
# mkpath(lecture7_figure_dir)
# savefig(fig,joinpath(lecture7_figure_dir,"activity.png"))
