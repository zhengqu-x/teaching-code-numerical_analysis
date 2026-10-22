# Lecture 5: Chebyshev nodes, least squares, orthogonal polynomials,
# rational approximation, and an introduction to differentiation.
# From the repository: julia --project=.
# Then: include("Lecture5/Lecture5_support.jl")
# Loading defines helpers only. Run the contents of one #= ... =# block.
# Activities 1 and 2 continue material postponed from Lecture 4.
# Activity 4 is an optional online-memory demo, not a trained sequence model.
# Activity numbers and core commands follow the Lecture 5 slides.
# The unnumbered polynomial-family illustration belongs between Activities 2 and 3.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture5_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture5"))

# ACTIVITY 1: Chebyshev or Equispaced Nodes?
# Runge interpolation at equispaced and Chebyshev nodes.
# n is the degree, so both interpolants use n+1 nodes. Try n=4,12,20.
#=
n = 12
xeq = collect(range(-1.0,1.0,length=n+1))
xch = chebyshev_roots(n+1)
w = barycentric_weights(xch)
P(z) = barycentric_value(xch,runge.(xch),w,z)
weq = barycentric_weights(xeq)
grid = range(-1.0,1.0,length=2001)
pe = [barycentric_value(xeq,runge.(xeq),weq,x) for x in grid]
pc = P.(grid)
p1 = plot(grid,runge.(grid);label="Runge function",color=:black,xlabel="x",ylabel="value")
plot!(p1,grid,pe;label="Equispaced",color=:blue,linestyle=:dash)
plot!(p1,grid,pc;label="Chebyshev",color=:red,linestyle=:dot)
scatter!(p1,xch,runge.(xch);label="Chebyshev nodes",color=:red,markersize=3)
p2 = plot(grid,abs.(runge.(grid).-pe);label="Equispaced",color=:blue,
          xlabel="x",ylabel="absolute error")
plot!(p2,grid,abs.(runge.(grid).-pc);label="Chebyshev",color=:red)
fig = plot(p1,p2;layout=(1,2),size=(1050,420),
           left_margin=8Plots.mm,bottom_margin=8Plots.mm)
display(fig)
println((equispaced=maximum(abs.(runge.(grid).-pe)),
         chebyshev=maximum(abs.(runge.(grid).-pc))))
# These are sampled maxima, not certified bounds.
# Optional export of the plot used in the slides:
# mkpath(lecture5_figure_dir)
# savefig(fig,joinpath(lecture5_figure_dir,"activity1.pdf"))
=#

# ACTIVITY 2: Projecting x^3 onto P_2.
# Weight rho=1 on [-1,1], with the basis 1,x,x^2.
# By hand: G_ij = integral(x^(i+j)), b_i = integral(x^(i+3)), i,j=0,1,2.
# The integral of x^k is zero for odd k and 2/(k+1) for even k.
#=
G = [2.0 0 2/3; 0 2/3 0; 2/3 0 2/5]
b = [0.0, 2/5, 0.0]
coeffs = G\b
# coeffs contains [a_0,a_1,a_2], so S_2^*(x) = 3x/5.
# These are continuous inner products computed from exact moment formulas.
residual_moments = b-G*coeffs
squared_L2_error = 2/7 - 2*dot(coeffs,b) + dot(coeffs,G*coeffs)
println((coefficients=coeffs,residual_moments=residual_moments,
         squared_L2_error=squared_L2_error))  # Expected squared error: 8/175.
# Verify by hand that integral((x^3-3x/5)*x^i) = 0 for i=0,1,2.
grid = range(-1.0,1.0,length=601)
fig = plot(grid,grid.^3;label="f(x) = x³",color=:black,linewidth=2.5,
           xlabel="x",ylabel="function value",legend=:topleft,
           size=(850,440),guidefontsize=13,tickfontsize=11,legendfontsize=12,
           left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,grid,evalpoly.(grid,Ref(coeffs));label="Projection: 3x/5",
      color=:blue,linestyle=:dash,linewidth=2.5)
display(fig)
# Optional export of the plot used in the slides:
# mkpath(lecture5_figure_dir)
# savefig(fig,joinpath(lecture5_figure_dir,"activity2.pdf"))
=#

# ILLUSTRATION: Standard polynomial families (between Activities 2 and 3).
#=
family_plots = Dict{Symbol,Any}()
for family in (:legendre,:chebyshev,:laguerre,:hermite)
    local interval = family == :laguerre ? (0.0,6.0) :
                     family == :hermite ? (-2.0,2.0) : (-1.0,1.0)
    local degree = family in (:laguerre,:hermite) ? 3 : 4
    local grid = range(interval...,length=601)
    local fig = plot(;xlabel="x",ylabel="polynomial value",size=(850,420),
                     legendfontsize=12,guidefontsize=14,tickfontsize=11,
                     left_margin=9Plots.mm,bottom_margin=8Plots.mm)
    for n in 0:degree
        plot!(fig,grid,orthopoly.(family,n,grid);label="n = $n",linewidth=2)
    end
    family_plots[family] = fig
    display(fig)
end
# Optional export:
# mkpath(lecture5_figure_dir)
# for (family,fig) in family_plots
#     savefig(fig,joinpath(lecture5_figure_dir,"basis_$family.pdf"))
# end
=#

# ACTIVITY 3: Two Continuous Projections.
# Continuous cubic projections of exp(x), with two weights.
# Inner products are approximated numerically; Lecture 6 explains Simpson's rule.
# Chebyshev projection changes variable x=cos(theta) to avoid endpoint singularities.
#=
aL = orthogonal_projection(exp,3;family=:legendre)
aC = orthogonal_projection(exp,3;family=:chebyshev)
S(z) = orthogonal_value(aL,z;family=:legendre)
println((Legendre=aL,Chebyshev=aC))
grid = range(-1.0,1.0,length=601)
SL = S.(grid)
SC = [orthogonal_value(aC,x;family=:chebyshev) for x in grid]
fig = plot(grid,exp.(grid).-SL;label="Weight 1 (Legendre)",
           xlabel="x",ylabel="exp(x) - projection",size=(850,420),
           left_margin=9Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,grid,exp.(grid).-SC;label="Chebyshev weight",linestyle=:dash)
display(fig)
println([simpson(x -> (exp(x)-S(x))*
                 orthopoly(:legendre,j,x),-1.0,1.0,2000) for j in 0:3])
=#

# ACTIVITY 4: Online Polynomial Memory.
# Online HiPPO-LegS memories with 4, 8, and 12 coefficients.
# Hold each incoming sample constant until the next sampling time.
# The update is exact for this held input (up to roundoff). It approximates
# the original continuous signal, whose full history is never integrated.
# Source: Gu et al., NeurIPS 2020, https://arxiv.org/abs/2008.07669
#=
signal(s) = 1 + s + 0.4sin(6pi*s)
N = 200; h = 1/N  # Try N=400. Also try signal(s)=1+s or a constant signal.
memories = [zeros(m) for m in (4,8,12)]  # Degrees 3, 7, and 11, respectively.
# Each model retains only its current m coefficients, never their time history.
styles = (:dash,:dashdot,:dot)
colors = (:blue,:red,:darkgreen)
panels = Plots.Plot[]  # Optional classroom display, not part of the memory update.
for k in 0:N-1
    local t,T = k*h,(k+1)*h
    local u = signal(t)  # One arriving sample, held on [t,T).
    for c in memories
        hippo_update!(c,u,t,T)  # Generalization of the c_0,c_1 update in the slides.
    end
    if k+1 in (N÷2,N)
        # Verification ONLY: evaluate the known test signal for the plot.
        # None of these grid values or plots feed into the next update.
        local grid = range(0.0,T,length=601)
        local p = plot(grid,signal.(grid);label="Original signal",color=:black,
                       linewidth=2.5,xlabel="past time s",ylabel="value",
                       title="Current time t = $(round(T;digits=2))",
                       legend=:topleft,legend_column=2)
        for (i,c) in enumerate(memories)
            local reconstructed = [hippo_value(c,s,T) for s in grid]
            plot!(p,grid,reconstructed;label="$(length(c)) coefficients",
                  color=colors[i],linestyle=styles[i],linewidth=2.5)
            println((time=T,stored_coefficients=length(c),
                     sampled_RMS=sqrt(sum(abs2,signal.(grid).-reconstructed)/length(grid))))
        end
        push!(panels,p)
    end
end
fig = plot(panels...;layout=(1,2),size=(1200,520),guidefontsize=13,
           tickfontsize=11,legendfontsize=11,titlefontsize=14,
           left_margin=8Plots.mm,right_margin=5Plots.mm,bottom_margin=8Plots.mm)
display(fig)
# After the loop, each vector in memories represents [0,1]. Earlier vectors are gone.
# More coefficients increase polynomial degree; halving h refines the held input.
# No projection/integration helper is used. Keep hippo_linear_update! as the
# explicit two-coefficient example corresponding to the preceding derivation.
# Optional export of the figure used in the slides:
# mkpath(lecture5_figure_dir)
# savefig(fig,joinpath(lecture5_figure_dir,"hippo_online.pdf"))
=#

# ACTIVITY 5: Taylor or Pade?
# Pade [3/2] versus the degree-5 Taylor polynomial for exp(-x).
# Both match six Taylor coefficients at zero, but neither is a minimax approximation.
#=
a = [(-1.0)^k/factorial(k) for k in 0:5]  # a[k+1] = f^(k)(0)/k!.
pq = pade_coefficients(a,3,2)
println(pq)  # p=[1,-0.6,0.15,-1/60], q=[1,0.4,0.05].
r(x) = evalpoly(x,pq.p)/evalpoly(x,pq.q)
println((pq, exp(-3), evalpoly(3,a), r(3)))
grid = range(0.0,3.0,length=601)
fig = plot(grid,exp.(-grid);label="exp(-x)",color=:black,
           xlabel="x",ylabel="value",size=(850,420),
           left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,grid,evalpoly.(grid,Ref(a));label="Taylor degree 5",linestyle=:dash)
plot!(fig,grid,r.(grid);label="Pade [3/2]",linestyle=:dot)
display(fig)
for x in (0.1,1.0,2.0,3.0)
    println((x=x,Taylor_error=abs(exp(-x)-evalpoly(x,a)),
             Pade_error=abs(exp(-x)-r(x))))
end
# Optional export of the plot used in the slides:
# mkpath(lecture5_figure_dir)
# savefig(fig,joinpath(lecture5_figure_dir,"pade_taylor.pdf"))
=#

# ACTIVITY 6: Predict the Error Ratios.
# First differences and convergence rates, before roundoff dominates.
# Predict error ratios 2 and 4 when h is halved.
#=
h = 0.1
ef(h) = abs(difference(sin,0.9,h;scheme=:forward)-cos(0.9))
ec(h) = abs(difference(sin,0.9,h)-cos(0.9))
println((ef(h)/ef(h/2), ec(h)/ec(h/2)))
hs = 0.2 ./ 2.0.^(0:8)
forward_errors = ef.(hs)
central_errors = ec.(hs)
println((forward_ratios=forward_errors[1:end-1]./forward_errors[2:end],
         central_ratios=central_errors[1:end-1]./central_errors[2:end]))
fig = plot(hs,forward_errors;label="Forward O(h)",xscale=:log10,yscale=:log10,
           xlabel="step h",ylabel="absolute error",markershape=:circle,
           size=(850,420),left_margin=8Plots.mm,bottom_margin=8Plots.mm)
plot!(fig,hs,central_errors;label="Central O(h^2)",markershape=:square)
display(fig)
=#

# Optional export after any activity:
# mkpath(lecture5_figure_dir)
# savefig(fig,joinpath(lecture5_figure_dir,"activity.png"))
