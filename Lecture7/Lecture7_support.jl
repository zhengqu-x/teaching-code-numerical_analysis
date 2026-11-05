# Lecture 7: composite quadrature, Euler-Maclaurin corrections,
# Romberg integration and periodic integrands. Four activities in slide order.
# Start Julia from the repository: julia --project=.
# Then include("Lecture7/Lecture7_support.jl"). Loading does not run activities.
# Select and run the CONTENTS of one #= ... =# block at a time, in slide order.
# Every activity also runs independently after loading this file.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture7_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture7"))

# ACTIVITY 1: Error and Function Evaluations.
# n counts subintervals; both methods use n+1 distinct nodes.
#=
ns = [4,8,16,32,64]
exact = exp(4)-1
et = [abs(trapezoid(exp,0.,4.,n)-exact) for n in ns]
es = [abs(simpson(exp,0.,4.,n)-exact) for n in ns]
println(et[1:end-1]./et[2:end])
println(es[1:end-1]./es[2:end])
fig = plot(ns,et;label="Trapezoidal",yscale=:log10,
    markershape=:circle,xlabel="number of subintervals n",ylabel="absolute error",
    size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,ns,es;label="Simpson",markershape=:square)
display(fig)
=#


# ACTIVITY 2: Correcting the Endpoints.
# For exp on [0,1], both endpoint derivative differences equal exp(1)-1.
# Predict ratios 4,16,64. At small errors Float64 roundoff obscures the ratios.

#=
ns = [2,4,8,16,32]
exact = exp(1)-1
h = 1.0 ./ ns
T = [trapezoid(exp,0.,1.,n) for n in ns]
C4 = T .- h.^2 .* exact/12
C6 = C4 .+ h.^4 .* exact/720
println(abs.(C4.-exact))
println(abs.(C6.-exact))
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


# ACTIVITY 3: Romberg and a Smoothness Test.
# levels=6 requires 2^5+1=33 calls, not the sum of six separate grids.

#=
result = romberg_table(exp,0.,1.;levels=6)
println("exp(x): absolute errors")
print_romberg_errors(result.table,exp(1)-1)
println((evaluations=result.evaluations,))
rough = romberg_table(sqrt,0.,1.;levels=6)
println("sqrt(x): absolute errors")
print_romberg_errors(rough.table,2/3)
# Check the first extrapolation against Simpson with two subintervals.
println((first_extrapolation=result.table[2,2],Simpson=simpson(exp,0.,1.,2)))
println("sqrt is not C^1 at the left endpoint")
fig = plot(1:6,max.([abs(result.table[k,k]-(exp(1)-1)) for k in 1:6],eps(Float64));
    label="exp(x)",yscale=:log10,markershape=:circle,xlabel="Romberg row k",
    ylabel="diagonal error",size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,1:6,[abs(rough.table[k,k]-2/3) for k in 1:6];label="sqrt(x)",markershape=:square)
display(fig)
=#


# ACTIVITY 4: Periodic Endpoints.
# Periodic analytic integrands can converge geometrically.
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
# Repeat on [0,pi]: although f(0) != f(pi), every odd derivative vanishes at both.
# Euler-Maclaurin therefore predicts rapid convergence on this half period too.
println((half_period_errors=[abs(trapezoid(f,0.,pi,n)-pi/3) for n in ns],))
=#

# Optional export after an activity:
# mkpath(lecture7_figure_dir)
# savefig(fig,joinpath(lecture7_figure_dir,"activity.png"))
