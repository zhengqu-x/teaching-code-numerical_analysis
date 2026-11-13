# Lecture 9 (13 November 2026): multiple integrals, Monte Carlo,
# Jacobi, Gauss-Seidel and convergence of linear iterations.
# Start in the repository with julia --project=.
# include("Lecture9/Lecture9_support.jl")
# Loading defines functions only. Run the contents of one #= ... =# block at a time.
# Six numbered activities run in slide order and independently after loading.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture9_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture9"))
# Before optional figure exports, run mkpath(lecture9_figure_dir).

# ACTIVITY 1: A Double Integral.
#=
f(x,y) = log(x+2y)
rx,ry = simpson_rule(1.,2.,4),simpson_rule(2.,3.,2)
S = tensor_quadrature(f,rx,ry)
n, m = 2, 2
gx,gy = mapped_gauss_rule(1.,2.,n+1),mapped_gauss_rule(2.,3.,m+1)
G = tensor_quadrature(f,gx,gy)
println((Simpson=S,Gauss=G))
# H''(z)=log(z). The corner formula below includes the factor 1/2.
H(z) = z^2*log(z)/2-3z^2/4
exact = (H(8)-H(7)-H(6)+H(5))/2
println((Simpson=S.value,calls=S.evaluations,error=abs(S.value-exact)))
println((Gauss=G.value,calls=G.evaluations,error=abs(G.value-exact)))
simpson_nodes = scatter([x for x in rx.nodes for y in ry.nodes],
    [y for x in rx.nodes for y in ry.nodes];label=false,
    title="Tensor Simpson: 5 x 3 = 15 nodes",xlabel="x",ylabel="y",
    markersize=6,xlims=(0.9,2.1),ylims=(1.9,3.1))
gauss_nodes = scatter([x for x in gx.nodes for y in gy.nodes],
    [y for x in gx.nodes for y in gy.nodes];label=false,
    title="Tensor Gauss: 3 x 3 = 9 nodes",xlabel="x",ylabel="y",
    markersize=6,xlims=(0.9,2.1),ylims=(1.9,3.1))
fig = plot(simpson_nodes,gauss_nodes;layout=(1,2),size=(1400,520),
    guidefontsize=16,tickfontsize=12,titlefontsize=16,
    left_margin=9Plots.mm,bottom_margin=10Plots.mm,right_margin=5Plots.mm)
display(fig)
# Neither comparison is at equal cost: report the counts alongside the errors.
println([(dimension=d,tensor_nodes=big(10)^d) for d in [2,4,8,12]])
=#

# ACTIVITY 2: Monte Carlo and Tensor Gauss for a Smooth Function.
#=
d = 8
f(x) = exp(sum(x)/length(x))
exact = (d*expm1(1/d))^d
sigma = sqrt((d/2*expm1(2/d))^d-exact^2)
Ns = [100,400,1600,6561]
results = [mc_unitcube(f,d,N;seed=2026) for N in Ns]
println("Monte Carlo on [0,1]^8: N evaluations; SE is the estimated standard error.")
for (N,r) in zip(Ns,results)
    println((N=N,estimate=r.value,SE=r.stderr,
             error=abs(r.value-exact),CI95=r.ci))
end

# Three Gauss-Legendre nodes in each coordinate: 3^8 = 6561 tensor nodes.
# Each weight is the product of eight one-dimensional weights.
# tensor_quadrature evaluates f at all 6561 nodes, even though this f is separable.
n = 2
rule = mapped_gauss_rule(0.,1.,n+1)
G = tensor_quadrature(f,fill(rule,d))
mc = results[end]
println("\nEqual-cost comparison: three Gauss nodes per coordinate, (n+1)^d = 6561 evaluations.")
println("  error = abs(estimate-exact). The Monte Carlo result uses seed 2026.")
println((method="Tensor Gauss",calls=G.evaluations,estimate=G.value,
         error=abs(G.value-exact)))
println((method="Monte Carlo",calls=mc.evaluations,estimate=mc.value,
         error=abs(mc.value-exact),SE=mc.stderr,CI95=mc.ci))

standard_errors = [r.stderr for r in results]
intervalplot = scatter(Ns,[r.value for r in results];
    yerror=[1.96r.stderr for r in results],xscale=:log10,label="Monte Carlo +/- 1.96 SE",
    xlabel="function evaluations N",ylabel="integral estimate",title="Monte Carlo and tensor Gauss")
hline!(intervalplot,[exact];label="Exact integral",linestyle=:dash)
scatter!(intervalplot,[G.evaluations],[G.value];label="Tensor Gauss (6561 nodes)",
    markershape=:diamond,markersize=7,color=:darkorange)
# On log-log axes, log(sigma/sqrt(N)) = log(sigma) - log(N)/2.
# Thus multiplying N by 4 divides the theoretical standard error by 2.
errorplot = plot(Ns,standard_errors;label="Estimate: s_N/sqrt(N)",markershape=:circle,
    linewidth=2,xscale=:log10,yscale=:log10,
    xlabel="Monte Carlo sample count N",ylabel="standard error",
    title="Standard error: log-log axes",legend=:topright,
    ylims=(0.18sigma/sqrt(maximum(Ns)),1.6sigma/sqrt(minimum(Ns))))
rate_grid = exp10.(range(log10(minimum(Ns)),log10(maximum(Ns)),length=120))
plot!(errorplot,rate_grid,sigma./sqrt.(rate_grid);
    label="Theory: sigma/sqrt(N), slope -1/2",linestyle=:dash,linewidth=2,color=:black)
# A guide triangle below the curves: run factor 4, drop factor 2.
guide_N = 400.0
guide_SE = 0.45sigma/sqrt(guide_N)
plot!(errorplot,[guide_N,4guide_N,4guide_N,guide_N],
    [guide_SE,guide_SE,guide_SE/2,guide_SE];label=false,color=:gray,linewidth=1.5)
annotate!(errorplot,2guide_N,1.22guide_SE,text("N x 4",11,:gray))
annotate!(errorplot,5guide_N,guide_SE/sqrt(2),text("SE / 2",11,:gray))
println("Rate guide: N x 4 gives standard error / 2; the log-log slope is -1/2.")
fig = plot(intervalplot,errorplot;layout=(1,2),size=(1400,550),
    guidefontsize=16,tickfontsize=13,titlefontsize=18,legendfontsize=11,
    left_margin=11Plots.mm,bottom_margin=11Plots.mm,right_margin=5Plots.mm)
display(fig)
# This smooth integrand favors Gaussian quadrature at this dimension and cost.
# The full tensor cost is 3^d; Monte Carlo permits any sample budget N.
# A particular Monte Carlo error need not decrease monotonically as N increases.
# The confidence intervals are approximate and rely on iid finite-variance samples.
=#

# ACTIVITY 3: A Threshold Probability.
#=
# Eight independent uniform loads exceed a total capacity of 4.5.
d = 8
n = 2
N = (n+1)^d
f(x) = sum(x) > 4.5 ? 1.0 : 0.0
# Symmetry gives P(sum X_i > 4.5) = P(sum X_i < 3.5).
# Inclusion-exclusion gives the volume of the clipped simplex in [0,1]^8.
exact = sum((-1)^k*binomial(d,k)*(3.5-k)^d for k in 0:3)/factorial(d)
rule = mapped_gauss_rule(0.,1.,n+1)
G = tensor_quadrature(f,fill(rule,d))
mc = mc_unitcube(f,d,N;seed=2026)
gauss_error = abs(G.value-exact)
println("\nThreshold probability P(X_1+...+X_8 > 4.5), with independent uniform X_i on [0,1].")
println("  Each method uses 6561 evaluations; error = abs(estimate-exact).")
println((exact=exact,method="Tensor Gauss",calls=G.evaluations,
         estimate=G.value,error=gauss_error))
println((method="Monte Carlo",seed=2026,calls=mc.evaluations,
         estimate=mc.value,error=abs(mc.value-exact),SE=mc.stderr,CI95=mc.ci))
println((theoretical_standard_error=sqrt(exact*(1-exact)/N),))

# A separate variability experiment: 100 runs, each with N evaluations.
# The comparison above remains a single Monte Carlo run at the same cost as Gauss.
seeds = 1:100
threshold_runs = [mc_unitcube(f,d,N;seed=seed) for seed in seeds]
threshold_estimates = [r.value for r in threshold_runs]
threshold_errors = abs.(threshold_estimates.-exact)
println("\nRepeated Monte Carlo runs: seeds 1:100, with 6561 evaluations per run.")
println((runs=length(seeds),mean_absolute_error=sum(threshold_errors)/length(seeds),
         largest_absolute_error=maximum(threshold_errors),
         more_accurate_than_Gauss=count(e -> e < gauss_error,threshold_errors)))
threshold_fig = scatter(collect(seeds),threshold_estimates;label="Monte Carlo",
    color=:royalblue,markersize=4,markerstrokewidth=0,
    xlabel="Monte Carlo seed",ylabel="estimated probability",
    title="Threshold probability: 6561 evaluations per run",
    xlims=(0,101),ylims=(0.23,0.295),legend=:outerright)
hline!(threshold_fig,[exact];label="Exact probability",color=:black,linewidth=2)
hline!(threshold_fig,[G.value];label="Tensor Gauss",color=:darkorange,
    linestyle=:dash,linewidth=2)
plot!(threshold_fig;size=(1400,530),guidefontsize=18,tickfontsize=14,
    titlefontsize=20,legendfontsize=15,left_margin=11Plots.mm,
    bottom_margin=11Plots.mm,right_margin=8Plots.mm,top_margin=6Plots.mm)
fig = threshold_fig
display(fig)
# Optional export: savefig(threshold_fig,joinpath(lecture9_figure_dir,"threshold_probability.pdf"))
=#

# RESEARCH EXAMPLE: Gaussian sampling in 784 dimensions.
# Optional extension after Activity 3; not a trained-model reproduction.
#=
d = 784
x = fill(1/(2sqrt(d)),d)
F(u) = sum(u) >= 0 ? 1 : 0
Ns = [100,400,1600,6400,25600]
rng = MersenneTwister(2026)
# Each call samples a complete d-dimensional N(0,I_d) noise vector.
outcomes = [F(x .+ randn(rng,d)) for r in 1:maximum(Ns)]
counts = cumsum(outcomes)
estimates = [counts[N]/N for N in Ns]
# The sample-variance formula for Bernoulli observations gives s_N/sqrt(N).
standard_errors = [sqrt(p*(1-p)/(N-1)) for (N,p) in zip(Ns,estimates)]
# p_1(x) = P(F(x+Z)=1) is the probability of predicting class 1.
# Since sum(x_i)=sqrt(d)/2 and sum(Z_i)/sqrt(d) is standard normal,
# p_1(x) = P(sum(Z_i)/sqrt(d) >= -1/2).
# Integrate its density from -1/2 to 10; the omitted tail is below 8e-24.
reference = adaptive_simpson(s -> exp(-s^2/2)/sqrt(2pi),-0.5,10.;atol=1e-12)
for (N,p,se) in zip(Ns,estimates,standard_errors)
    println((dimension=d,N=N,estimate=p,SE=se,
             CI95=(p-1.96se,p+1.96se),reference=reference.value))
end
fig = scatter(Ns,estimates;yerror=1.96 .* standard_errors,
    xscale=:log10,label="Monte Carlo +/- 1.96 SE",markersize=5,
    xlabel="number of noisy inputs N",ylabel="probability of class 1",
    title="Gaussian sampling in 784 dimensions",legend=:topright,
    size=(1100,430),guidefontsize=16,tickfontsize=13,titlefontsize=18,
    legendfontsize=12,left_margin=10Plots.mm,bottom_margin=10Plots.mm,
    right_margin=5Plots.mm,top_margin=5Plots.mm)
hline!(fig,[reference.value];label="Reference: 0.691462",linestyle=:dash,linewidth=2)
display(fig)
# The error bars are approximate 95% confidence intervals for this fixed class.
# Here the exact variance is p_1(x)*(1-p_1(x)), so the standard deviation
# of the Monte Carlo estimate is sqrt(p_1(x)*(1-p_1(x))/N).
=#

# ACTIVITY 4: Jacobi and Gauss-Seidel on the four-equation example.
# Iterates are columns; column 1 is x^(0), column 2 is the first sweep.
#=
A = [10. -1. 2. 0.; -1. 11. -1. 3.; 2. -1. 10. -1.; 0. 3. -1. 8.]
b = [6.,25.,-11.,15.]
xstar = [1.,2.,-1.,1.]
Tj = stationary_matrix(A;method=:jacobi)
Tg = stationary_matrix(A;method=:gauss_seidel)
println("Induced infinity norms (maximum absolute row sums), for the convergence bounds:")
println((q_Jacobi=opnorm(Tj,Inf),q_Gauss_Seidel=opnorm(Tg,Inf)))
jac = stationary_solve(A,b;method=:jacobi,rtol=1e-12)
gs = stationary_solve(A,b;method=:gauss_seidel,rtol=1e-12)
println("The first column is the initial zero vector; the second is the first sweep.")
println((Jacobi_first_sweep=jac.iterates[:,2],Gauss_Seidel_first_sweep=gs.iterates[:,2]))
println("converged means the relative residual test passed; actual_error uses the known solution.")
for (name,r) in (("Jacobi",jac),("Gauss-Seidel",gs))
    println((method=name,iterations=r.iterations,relative_residual=r.residuals[end]/norm(b,Inf),
             actual_error=norm(r.x-xstar,Inf),converged=r.converged))
end
resplot = plot(;xlabel="iteration k",ylabel="relative residual",yscale=:log10,
    title="Residual test",legend=:topright)
errplot = plot(;xlabel="iteration k",ylabel="actual error in infinity norm",yscale=:log10,
    title="Known solution: (1, 2, -1, 1)",legend=:topright)
for (name,r) in (("Jacobi",jac),("Gauss-Seidel",gs))
    plot!(resplot,0:r.iterations,max.(r.residuals./norm(b,Inf),1e-16);label=name,linewidth=3)
    errors = [norm(r.iterates[:,k]-xstar,Inf) for k in axes(r.iterates,2)]
    plot!(errplot,0:r.iterations,max.(errors,1e-16);label=name,linewidth=3)
end
fig = plot(resplot,errplot;layout=(1,2),size=(1450,550),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=13,left_margin=12Plots.mm,
    bottom_margin=12Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
display(fig)
# Optional export: savefig(fig,joinpath(lecture9_figure_dir,"activity4.pdf"))
=#

# ACTIVITY 5: When does iteration converge?
# Positive definiteness does not ensure convergence of Jacobi.
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

# A small residual alone need not imply a small solution error.
A_scaled = Diagonal([1.,1e-8])
b_scaled = [1.,1e-8]
x_approx = [1.,0.]
println("Residual and solution error for the scaled diagonal example:")
println((residual=norm(b_scaled-A_scaled*x_approx,Inf),
         actual_error=norm(x_approx-ones(2),Inf),condition_number=cond(A_scaled,Inf)))
# Optional export: savefig(fig,joinpath(lecture9_figure_dir,"activity5.pdf"))
=#

# ACTIVITY 6: A graph-prediction fixed point.
# This row-normalized classroom example illustrates the propagation principle
# of APPNP. The paper uses symmetric normalization; no model is trained here.
# Source: Klicpera, Bojchevski and Guennemann, ICLR 2019,
# https://arxiv.org/abs/1810.05997
#=
n = 8
W = Matrix{Float64}(I,n,n)
for i in 1:n-1
    W[i,i+1] = W[i+1,i] = 1.0
end
P = W ./ sum(W,dims=2)
scores = [1.,1.,0.,0.,0.,0.,-1.,-1.]
alpha = 0.2
A = Matrix{Float64}(I,n,n)-(1-alpha)*P
rhs = alpha*scores
xstar = A\rhs
z = copy(scores)
states = [copy(z)]
residuals = [norm(rhs-A*z,Inf)]
for k in 1:120
    z .= (1-alpha)*P*z+alpha*scores
    push!(states,copy(z))
    push!(residuals,norm(rhs-A*z,Inf))
end
jac = stationary_solve(A,rhs;method=:jacobi,x0=scores,rtol=1e-10,maxiter=400)
gs = stationary_solve(A,rhs;method=:gauss_seidel,x0=scores,rtol=1e-10,maxiter=400)
println("scores are initial scalar predictions; P averages each node with its neighbors.")
println((row_sums=vec(sum(P,dims=2)),contraction_bound=1-alpha,
         propagation_error=norm(z-xstar,Inf),Jacobi_iterations=jac.iterations,
         Gauss_Seidel_iterations=gs.iterations))
stateplot = plot(1:n,scores;label="Initial scores",linestyle=:dash,linewidth=2,
    xlabel="node index",ylabel="prediction score",title="Prediction propagation",legend=:topright)
for k in (1,5,20)
    plot!(stateplot,1:n,states[k+1];label="Propagation k=$k",linewidth=2)
end
plot!(stateplot,1:n,xstar;label="Linear solve",linewidth=3,color=:black,linestyle=:dot)
resplot = plot(0:120,max.(residuals./norm(rhs,Inf),1e-16);label="Propagation",linewidth=3,
    yscale=:log10,xlabel="iteration k",ylabel="relative residual",
    title="Same fixed point, different iterations",legend=:topright)
for (name,r) in (("Jacobi",jac),("Gauss-Seidel",gs))
    plot!(resplot,0:r.iterations,max.(r.residuals./norm(rhs,Inf),1e-16);label=name,linewidth=3)
end
fig = plot(stateplot,resplot;layout=(1,2),size=(1450,550),guidefontsize=17,
    tickfontsize=13,titlefontsize=18,legendfontsize=12,left_margin=12Plots.mm,
    bottom_margin=12Plots.mm,right_margin=6Plots.mm,top_margin=6Plots.mm)
display(fig)
# Optional export: savefig(fig,joinpath(lecture9_figure_dir,"activity6.pdf"))
=#
