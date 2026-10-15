# Lecture 4: cubic splines, B-splines, and polynomial approximation.
# From this folder, load with: include("Lecture4_support.jl")
# Shared functions come from ../helper.jl (LinearAlgebra and Plots).
# By default, loading only loads the helpers; all activity blocks are commented.
# Run a selected block, or uncomment it and load this file again.

# Activities 1-3: clamped/natural cubic splines.
# Activities 4-5: cubic B-spline bases and degrees 1-4.
# Activity 6: equispaced versus Chebyshev nodes for the Runge function.
# Activity 7: discrete least-squares approximation.
# Unnumbered ILLUSTRATION blocks reproduce figures shown in the slides.
# For cubic splines, Julia entry i+1 represents m_i = S'(x_i).
# Julia basis index i=1,2,... corresponds to mathematical N_{i-1,p}.
# p is the degree; smoothness is C^(p-1) for our distinct knots.
# n=nsegments counts intervals; t_0,...,t_(n+2p) gives n+p basis functions.
# Shared functions and package imports; the path is relative to this file.
include(joinpath(@__DIR__, "..", "helper.jl"))
lecture4_figure_dir = normpath(joinpath(@__DIR__, "..", "figures", "Lecture4"))

# CLASSROOM ACTIVITIES (in slide order)
# Select the code inside one #= ... =# block and run it after loading this file,
# or uncomment the desired block. Each activity can be run independently.

# ACTIVITY 1: Clamped spline through (0,0), (1,1), (2,0).
# First solve by hand: m_0=m_2=1 and m_0+4m_1+m_2=0.
# Thus (m_0,m_1,m_2)=(1,-1/2,1).

#=
xx, yy = [0.0,1.0,2.0], [0.0,1.0,0.0]
mc = spline_slopes(xx,yy;boundary=:clamped,endpoint_slopes=(1.0,1.0))
display(mc)  # Expected: [1,-0.5,1].
grid = range(0.0,2.0,length=1001)
p = plot(grid,[piecewise_hermite(xx,yy,mc,z) for z in grid];
         label="Clamped",xlabel="x",ylabel="S(x)")
scatter!(p,xx,yy;label="Nodes",color=:black)
display(p)
=#


# ACTIVITY 2: Natural spline through the same three points; compare with clamped.
# 2m_0+m_1=3, m_0+4m_1+m_2=0, m_1+2m_2=-3.
# Thus (m_0,m_1,m_2)=(3/2,0,-3/2).

#=
xx, yy = [0.0,1.0,2.0], [0.0,1.0,0.0]
mn = spline_slopes(xx,yy)
display(mn)  # Expected: [1.5,0,-1.5].
mc = spline_slopes(xx,yy;boundary=:clamped,endpoint_slopes=(1.0,1.0))
# Mathematical m_1 is Julia's second entry: boundary conditions change it too.
println((natural_m1=mn[2],clamped_m1=mc[2]))  # (0,-0.5).
grid = range(0.0,2.0,length=1001)
p = plot(grid,[piecewise_hermite(xx,yy,mn,z) for z in grid];
         label="Natural: m1 = 0",xlabel="x",ylabel="S(x)",legend=:bottomleft)
plot!(p,grid,[piecewise_hermite(xx,yy,mc,z) for z in grid];
      label="Clamped: m1 = -1/2",linestyle=:dash)
scatter!(p,xx,yy;label="Nodes",color=:black)
display(p)
=#

# ACTIVITY 3: Natural or clamped? sin(x) on [0,pi/2], with five nodes.
# The same piecewise_hermite evaluator is used; only the computed slopes differ.
# Predict: sin''(0)=0 but sin''(pi/2)=-1. Which natural condition conflicts?
#=
xx = collect(range(0.0,pi/2,length=5))
yy = sin.(xx)
mn = spline_slopes(xx,yy)
mc = spline_slopes(xx,yy;boundary=:clamped,endpoint_slopes=(1.0,0.0))
grid = range(0.0,pi/2,length=2001)
sn = [piecewise_hermite(xx,yy,mn,z) for z in grid]
sc = [piecewise_hermite(xx,yy,mc,z) for z in grid]
p = plot(grid,sin.(grid);label="sin(x)",xlabel="x",ylabel="value")
plot!(p,grid,sn;label="Natural",linestyle=:dash)
plot!(p,grid,sc;label="Clamped",linestyle=:dot)
scatter!(p,xx,yy;label="Nodes",color=:black)
e = plot(grid,sin.(grid).-sn;label="Natural",xlabel="x",ylabel="signed error")
plot!(e,grid,sin.(grid).-sc;label="Clamped")
comparison = plot(p,e;layout=(1,2),size=(1150,460),
                  left_margin=8Plots.mm,bottom_margin=8Plots.mm)
display(comparison)
(natural_error=maximum(abs.(sin.(grid).-sn)),
 clamped_error=maximum(abs.(sin.(grid).-sc)))
=#

# ILLUSTRATION: One cubic B-spline, immediately after the C2 spline-space slide.
# Local knots: 0, 1/4, 1/2, 3/4, 1. Circles mark the values at these knots.
# The exact piecewise formula appears on the slide; the plot uses the recursion.
#=
single_basis = plot_cubic_bspline()
display(single_basis)
# Optional export:
# mkpath(lecture4_figure_dir)
# savefig(single_basis,joinpath(lecture4_figure_dir,"Cubic-B-spline-single.pdf"))
=#

# ACTIVITY 4: Five cubic basis splines, immediately after "Cubic Basis Splines".
# On [0,1], n=2 intervals give h=1/2 and partition nodes 0, 1/2, 1.
# Extend by three knots on each side: t_k=(k-3)/2, k=0,...,8.
# There are nine knots and n+3=5 basis functions N_0,3,...,N_4,3.
# Plot the full supports, whose union is [-3/2,5/2], and shade [0,1].
# Circles mark each basis function's five local knots.
#=
n = 2
p = 3
knots = uniform_knots(n,p)
println((knots=knots, knot_count=length(knots), basis_functions=n+p))
fig = plot_cubic_bspline_basis(n; full_support=true)
display(fig)
# To compare with the restricted view: display(plot_cubic_bspline_basis(n))
# Full supports: [-1.5,0.5], [-1,1], [-0.5,1.5], [0,2], [0.5,2.5].
# Check: N_4,3 vanishes on (0,1/2); N_0,3 vanishes on (1/2,1).
# Optional export:
# mkpath(lecture4_figure_dir)
# savefig(fig,joinpath(lecture4_figure_dir,"Cubic-B-spline-basis-n2-full.pdf"))
=#

# ACTIVITY 5: Plot ONE B-spline for each degree p=1,2,3,4.
# The resulting figure is also used in the lecture slides.
# All knots are distinct, with p extra knots on either side of [0,1].
# Choose N_(p,p), whose full support starts at t_p=0 and ends at (p+1)h.
# For n=6, h=1/6: supports [0,1/3], [0,1/2], [0,2/3], [0,5/6].
# Four panels use the same axes. Circles mark the p+2 local knots.
# The smoothness increases from C0 through C3.
#=
nsegments = 6  # Number of intervals in [0,1].
for p in 1:4
    local knots = uniform_knots(nsegments,p)
    local i = p+1  # Julia index for mathematical N_(p,p).
    println((degree=p, smoothness=p-1, intervals=nsegments,
             knots=length(knots), basis_functions=nsegments+p,
             plotted_basis_index=p,
             support=(knots[i],knots[i+p+1])))
end
fig = plot_bspline_degrees(nsegments)
display(fig)
# Optional export:
# mkpath(lecture4_figure_dir)
# savefig(fig,joinpath(lecture4_figure_dir,"B-spline-basis.pdf"))
=#

# ILLUSTRATION: Degree-at-most-7 Remez approximation of ReLU on [-1,1].
# Symmetry makes the x^7 coefficient zero; the actual degree is 6.
# The nine alternating error extrema verify minimax optimality in P_7.
#=
result = remez_relu7()
display(result.coefficients)  # Coefficients of 1,x,...,x^7.
println((maximum_error=result.max_error, iterations=result.iterations))
fig = plot_relu_remez7(result)
display(fig)
# Optional export:
# mkpath(lecture4_figure_dir)
# savefig(fig,joinpath(lecture4_figure_dir,"ReLU-Remez-degree7.pdf"))
# savefig(fig,joinpath(lecture4_figure_dir,"ReLU-Remez-degree7.png"))
=#

# ACTIVITY 6: Compare equispaced and Chebyshev interpolation nodes for Runge.
# n is the polynomial degree, so each rule uses n+1 nodes.
#=
n = 12
xeq = collect(range(-1.0,1.0,length=n+1))
xch = chebyshev_roots(n+1)
yeq, ych = runge.(xeq), runge.(xch)
weq, wch = barycentric_weights(xeq), barycentric_weights(xch)
xx = range(-1.0,1.0,length=4001)
pe = [barycentric_value(xeq,yeq,weq,x) for x in xx]
pc = [barycentric_value(xch,ych,wch,x) for x in xx]
p = plot(xx,runge.(xx);label="Runge function",xlabel="x",ylabel="value")
plot!(p,xx,pe;label="Equispaced nodes",linestyle=:dash)
plot!(p,xx,pc;label="Chebyshev nodes",linestyle=:dot)
scatter!(p,xeq,yeq;label="Equispaced data",color=:blue,markersize=2)
scatter!(p,xch,ych;label="Chebyshev data",color=:red,markersize=2)
display(p)
(equispaced_error=maximum(abs.(runge.(xx).-pe)),
 chebyshev_error=maximum(abs.(runge.(xx).-pc)))
=#

# ACTIVITY 7: Discrete least-squares fit of degree 3 to smooth sampled data.
# The overdetermined solve uses Julia's QR-based backslash operation.
#=
nodes = collect(range(-1.0,1.0,length=25))
data = exp.(nodes) .+ 0.08sin.(17nodes)
A = [x^(j-1) for x in nodes, j in 1:4]
coeffs = A\data
xx = range(-1.0,1.0,length=1001)
fit = [sum(coeffs[j]*x^(j-1) for j in eachindex(coeffs)) for x in xx]
p = plot(xx,exp.(xx);label="exp(x)",xlabel="x",ylabel="value")
scatter!(p,nodes,data;label="Sampled data",color=:black,markersize=3)
plot!(p,xx,fit;label="Least-squares cubic",linestyle=:dash)
display(p)
residual = data-A*coeffs
(discrete_L2=sqrt(sum(abs2,residual)),discrete_Linf=maximum(abs.(residual)))
=#
